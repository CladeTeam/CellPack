"""Check fixed-example execution and read-only, file-local language analysis."""
import asyncio
import json
import unittest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer
from server import create_app, DOCS, SERVICE


class PlaygroundTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(TestServer(create_app(origins={'https://docs.example.org'}), handler_cancellation=True))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_fixed_examples_match_documentation(self):
        examples = json.loads((DOCS / 'public/api/examples.json').read_text())
        for name, example in examples.items():
            with self.subTest(example=name):
                response = await self.client.post('/run', json={'example': name})
                self.assertEqual(response.status, 200)
                result = await response.json()
                self.assertTrue(result['success'], result)
                self.assertEqual(result['stdout'].strip(), example['expected'])

    async def test_source_code_arguments_and_unknown_examples_are_rejected(self):
        for payload in [
            {'example': 'write', 'code': 'fn main() {}'},
            {'example': 'write', 'args': ['/etc/passwd']},
            {'example': '../write'}, {'example': 'unknown'},
            {'code': 'fn main() {}'}, [], {'example': None},
        ]:
            response = await self.client.post('/run', json=payload)
            self.assertEqual(response.status, 400, payload)
        self.assertFalse(self.client.server.app[SERVICE].processes)

    async def test_origins_and_analysis_file_validation(self):
        response = await self.client.get('/health', headers={'Origin': 'https://other.example.org'})
        self.assertEqual(response.status, 403)
        response = await self.client.get('/health', headers={'Origin': 'https://docs.example.org'})
        self.assertEqual(response.status, 200)
        for path in ['/lsp', '/lsp?example=write&file=../../Cargo.toml', '/lsp?example=unknown']:
            response = await self.client.get(path)
            self.assertEqual(response.status, 400)

    async def test_lsp_uses_fixed_source_and_keeps_definitions_in_current_file(self):
        service = self.client.server.app[SERVICE]
        async with self.client.ws_connect('/lsp?example=write&file=main.rs') as ws:
            counter = 0
            async def request(method, params, error=False):
                nonlocal counter
                counter += 1
                current = counter
                await ws.send_json({'jsonrpc': '2.0', 'id': current, 'method': method, 'params': params})
                while True:
                    message = await ws.receive_json(timeout=40)
                    if message.get('id') == current and 'method' not in message:
                        if error:
                            self.assertIn('error', message)
                            return message['error']
                        self.assertNotIn('error', message)
                        return message.get('result')
                    if 'id' in message and 'method' in message:
                        response = [None for _ in message.get('params', {}).get('items', [])] if message['method'] == 'workspace/configuration' else None
                        await ws.send_json({'jsonrpc': '2.0', 'id': message['id'], 'result': response})
            await request('initialize', {'rootUri': 'file:///work', 'capabilities': {}})
            await ws.send_json({'jsonrpc': '2.0', 'method': 'initialized', 'params': {}})
            uri = 'file:///work/src/main.rs'
            await ws.send_json({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {'textDocument': {'uri': uri, 'languageId': 'rust', 'version': 1, 'text': 'ignored client source'}}})
            code = service.examples['write']['code']
            def position(index):
                return {'line': code[:index].count('\n'), 'character': len(code[:index].rsplit('\n', 1)[-1])}
            imported = position(code.index('CpkWriter') + 3)
            hover = None
            for _ in range(25):
                hover = await request('textDocument/hover', {'textDocument': {'uri': uri}, 'position': imported})
                if hover:
                    break
                await asyncio.sleep(.2)
            self.assertIn('CpkWriter', json.dumps(hover))
            outside = await request('textDocument/definition', {'textDocument': {'uri': uri}, 'position': imported})
            self.assertEqual(outside, [])
            local = await request('textDocument/definition', {'textDocument': {'uri': uri}, 'position': position(code.index('values.iter') + 2)})
            self.assertTrue(local)
            self.assertTrue(all(location.get('uri', location.get('targetUri')) == uri for location in local))
            await request('cellpack/readFile', {'uri': 'file:///library/crates/cellpack/src/lib.rs'}, error=True)
            await ws.send_json({'jsonrpc': '2.0', 'method': 'textDocument/didChange', 'params': {'textDocument': {'uri': uri, 'version': 2}, 'contentChanges': [{'text': 'fn main() {}'}]}})
            closed = await ws.receive(timeout=10)
            self.assertEqual(closed.type, WSMsgType.CLOSE)
        for _ in range(30):
            if not service.processes:
                break
            await asyncio.sleep(.1)
        self.assertFalse(service.processes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
