import { Compartment, StateEffect } from '@codemirror/state';
import { hoverTooltip, keymap, EditorView } from '@codemirror/view';

const position = (doc, offset) => { const line = doc.lineAt(offset); return { line: line.number - 1, character: offset - line.from }; };
const offset = (doc, pos) => { const line = doc.line(Math.min(doc.lines, pos.line + 1)); return Math.min(line.to, line.from + pos.character); };

export async function connectAnalysis(view, endpoint, status, signal, example, file) {
  const uri = 'file:///work/src/' + file;
  const url = new URL(endpoint.replace(/\/$/, '') + '/lsp', window.location.href);
  url.searchParams.set('example', example); url.searchParams.set('file', file);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(url), compartment = new Compartment();
  view.dispatch({ effects: StateEffect.appendConfig.of(compartment.of([])) });
  let nextId = 0, disposed = false, openTimer, rejectOpen;
  const pending = new Map();
  const send = message => { if(socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ jsonrpc: '2.0', ...message })); };
  function request(method, params) {
    if(disposed) return Promise.reject(new Error('LSP disconnected.'));
    return new Promise((resolve, reject) => {
      const id = ++nextId;
      const timer = setTimeout(() => { pending.delete(id); reject(new Error('LSP request timed out.')); }, 20000);
      pending.set(id, { resolve, reject, timer }); send({ id, method, params });
    });
  }
  socket.onmessage = event => {
    const message = JSON.parse(event.data), task = pending.get(message.id);
    if(task) {
      pending.delete(message.id); clearTimeout(task.timer);
      message.error ? task.reject(new Error(message.error.message)) : task.resolve(message.result);
    } else if(message.id != null && message.method) {
      send({ id: message.id, result: message.method === 'workspace/configuration' ? message.params.items.map(() => null) : null });
    }
  };
  const close = () => {
    if(disposed) return;
    disposed = true;
    clearTimeout(openTimer);
    rejectOpen?.(new Error('LSP disconnected.'));
    rejectOpen = undefined;
    socket.close(); view.dispatch({ effects: compartment.reconfigure([]) });
    signal?.removeEventListener('abort', close);
    for(const task of pending.values()) { clearTimeout(task.timer); task.reject(new Error('LSP disconnected.')); }
    pending.clear(); status('off');
  };
  signal?.addEventListener('abort', close, { once: true }); socket.onclose = close;
  try {
    if(signal?.aborted) throw new Error('LSP cancelled.');
    await new Promise((resolve, reject) => {
      rejectOpen = reject;
      openTimer = setTimeout(() => reject(new Error('LSP service unavailable.')), 10000);
      socket.onopen = () => { clearTimeout(openTimer); rejectOpen = undefined; resolve(); };
      socket.onerror = () => { clearTimeout(openTimer); rejectOpen = undefined; reject(new Error('LSP service unavailable.')); };
    });
    await request('initialize', { processId: null, rootUri: 'file:///work', capabilities: { textDocument: { hover: { contentFormat: ['plaintext'] } } } });
    send({ method: 'initialized', params: {} });
    send({ method: 'textDocument/didOpen', params: { textDocument: { uri, languageId: 'rust', version: 1, text: view.state.doc.toString() } } });
    async function gotoDefinition(pos) {
      try {
        const result = await request('textDocument/definition', { textDocument: { uri }, position: position(view.state.doc, pos) });
        const locations = Array.isArray(result) ? result : result ? [result] : [];
        const local = locations.find(location => (location.targetUri || location.uri) === uri);
        if(!local) { status('local-only'); return; }
        const range = local.targetSelectionRange || local.range;
        view.dispatch({ selection: { anchor: offset(view.state.doc, range.start) }, scrollIntoView: true }); view.focus(); status('connected');
      } catch(error) { status(error.message); }
    }
    view.dispatch({ effects: compartment.reconfigure([
      hoverTooltip(async (editor, pos) => {
        try {
          const hover = await request('textDocument/hover', { textDocument: { uri }, position: position(editor.state.doc, pos) });
          if(!hover?.contents) return null;
          const contents = Array.isArray(hover.contents) ? hover.contents : [hover.contents];
          return { pos, create() { const dom = document.createElement('pre'); dom.className = 'cp-lsp-hover'; dom.textContent = contents.map(c => typeof c === 'string' ? c : c.value).join('\n'); return { dom }; } };
        } catch { return null; }
      }),
      keymap.of([{ key: 'F12', run: editor => { gotoDefinition(editor.state.selection.main.head); return true; } }]),
      EditorView.domEventHandlers({ click(event, editor) { if(event.ctrlKey || event.metaKey) { const pos = editor.posAtCoords({ x: event.clientX, y: event.clientY }); if(pos != null) { gotoDefinition(pos); return true; } } return false; } }),
    ]) });
    status('connected'); return { close };
  } catch(error) { close(); throw error; }
}
