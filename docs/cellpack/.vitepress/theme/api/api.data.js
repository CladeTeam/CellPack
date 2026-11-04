import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { canonicalLink } from '../../routes.mjs';

const directory = fileURLToPath(new URL('../../../public/api/', import.meta.url));
export default {
  watch: [directory + 'symbols.json'],
  load() {
    const source = JSON.parse(readFileSync(directory + 'symbols.json', 'utf8'));
    const symbols = {};
    for (const symbol of Object.values(source)) {
      const [page, fragment] = canonicalLink(symbol.href).split('#');
      const href = page.replace(/(?:\.html)?$/, '.html') + (fragment ? '#' + fragment : '');
      const { key, module, kind, summary } = symbol;
      symbols[key] = { key, module, kind, summary, href };
    }
    return {
      symbols,
    };
  },
};
