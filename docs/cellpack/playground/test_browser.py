"""Exercise fixed examples, highlighting and read-only file-local analysis."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]
OUTPUT = ROOT / 'tmp/api-browser-review'
BASE = os.environ.get('CELLPACK_REVIEW_URL', 'http://127.0.0.1:5191')


def main():
    OUTPUT.mkdir(exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get('CELLPACK_CHROMIUM'), args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.post_data_json) if request.url.endswith('/run') and request.method == 'POST' else None)
        page.goto(BASE + '/api.html', wait_until='networkidle')
        page.get_by_label('Find an interface').fill('ReadSelection')
        assert page.locator('.cp-api-result').count() == 1
        page.get_by_role('button', name='Quick view', exact=True).click()
        page.locator('dialog[open]').wait_for()
        page.locator('dialog details summary').click()
        page.wait_for_function("document.querySelector('dialog details pre').textContent.includes('pub struct ReadSelection')")
        page.keyboard.press('Escape')
        page.goto(BASE + '/generated-store-struct-CpkWriter.html', wait_until='networkidle')
        page.locator('.cp-signature a').filter(has_text='CpkOptions').first.click()
        page.locator('dialog[open]').wait_for()
        assert page.url.endswith('/rust/api/generated-store-struct-CpkWriter.html')
        page.keyboard.press('Escape')
        example = page.locator('.cp-example').first
        assert example.get_by_role('button', name='Edit code').count() == 0
        assert example.locator('.shiki span[style*="color"]').count() > 20
        colors = example.locator('.shiki span[style*="color"]').evaluate_all('(spans)=>[...new Set(spans.map(s=>getComputedStyle(s).color))]')
        assert len(colors) >= 4, colors
        example.get_by_role('button', name='Run example', exact=True).click()
        example.get_by_text('Run completed.', exact=True).wait_for(timeout=15000)
        assert 'shape: 2 × 3' in example.locator('.cp-result-body').inner_text()
        assert requests[-1] == {'example': 'write'}
        example.screenshot(path=str(OUTPUT / 'example-light.png'))
        example.get_by_role('button', name='LSP', exact=True).click()
        example.get_by_role('button', name='LSP on', exact=True).wait_for(timeout=30000)
        content = example.locator('.cm-content')
        assert content.get_attribute('contenteditable') == 'false'
        original = content.inner_text()
        content.click()
        page.keyboard.type('unsafe edit')
        assert content.inner_text() == original
        page.keyboard.press('Control+Home')
        for _ in range(6):
            page.keyboard.press('ArrowRight')
        page.keyboard.press('F12')
        example.get_by_text('Definition is outside this file.', exact=False).wait_for(timeout=20000)
        assert page.locator('dialog[open]').count() == 0
        assert page.url.endswith('/rust/api/generated-store-struct-CpkWriter.html')
        example.get_by_role('button', name='LSP on', exact=True).click()
        example.get_by_role('tab', name='common.rs', exact=True).click()
        assert 'pub const MATRIX' in example.locator('.cp-code-static').inner_text()
        example.get_by_role('tab', name='main.rs', exact=True).click()
        page.locator('button.VPSwitchAppearance').first.click()
        example.screenshot(path=str(OUTPUT / 'example-dark.png'))
        for width in [375, 768, 1440]:
            page.set_viewport_size({'width': width, 'height': 900})
            for document in ['api.html', 'api-examples.html', 'generated-store-struct-CpkWriter.html', 'generated-loader-struct-Run.html']:
                page.goto(BASE + '/' + document, wait_until='networkidle')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, document)
            page.locator('.cp-example').first.screenshot(path=str(OUTPUT / f'example-{width}.png'))
        page.goto(BASE + '/api-examples.html', wait_until='networkidle')
        page.route('**/run', lambda route: route.fulfill(status=503, content_type='application/json', body='{"error":"offline"}'))
        offline = page.locator('.cp-example').first
        offline.get_by_role('button', name='Run example', exact=True).click()
        offline.get_by_text('Run service unavailable.', exact=False).wait_for()
        assert 'shape: 2 × 3' in offline.locator('.cp-result-body').inner_text()
        assert not errors, errors
        browser.close()
    print('PASS: fixed-example request, highlighted light/dark source, read-only LSP, file-local navigation, file tabs, responsive pages, offline fallback')


if __name__ == '__main__':
    main()
