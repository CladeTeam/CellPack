import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createHighlighter } from 'shiki';

const file = fileURLToPath(new URL('../../../public/api/examples.json', import.meta.url));
export default {
  watch: [file],
  async load() {
    const source = JSON.parse(readFileSync(file, 'utf8'));
    const examples = {};
    const files = [];
    const highlighter = await createHighlighter({ themes: ['github-light', 'github-dark'], langs: ['rust'] });
    const interned = new Map();
    const intern = code => {
      if (!interned.has(code)) {
        interned.set(code, files.length);
        files.push({ code, html: highlighter.codeToHtml(code, {
          lang: 'rust', themes: { light: 'github-light', dark: 'github-dark' },
        }) });
      }
      return interned.get(code);
    };
    try {
      for (const [key, { code, common, ...metadata }] of Object.entries(source)) {
        examples[key] = { ...metadata, main: intern(code), common: intern(common) };
      }
      return { examples, files };
    } finally {
      highlighter.dispose();
    }
  },
};
