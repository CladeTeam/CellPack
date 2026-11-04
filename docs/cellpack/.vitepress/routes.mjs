// Source files remain stable while each documentation surface owns its URLs.
export const rustGuides = ['overview', 'installation', 'quickstart', 'concepts', 'storage', 'data-sources', 'accesses', 'loading', 'planning', 'process-functions', 'codecs', 'data-contracts', 'abi', 'performance', 'profiling', 'architecture', 'format', 'troubleshooting', 'contributing'];

export function canonicalLink(value) {
  if (!value || /^(?:[a-z][\w+.-]*:|\/\/|#)/i.test(value)) return value;
  const match = value.match(/^\/?([^?#]*)(.*)$/);
  const name = match[1].replace(/\.(md|html)$/, '');
  const suffix = match[1].endsWith('.html') ? '.html' : match[1].endsWith('.md') ? '.md' : '';
  let destination;
  if (name === 'api') destination = suffix ? '/rust/api/index' : '/rust/api/';
  else if (name === 'api-examples') destination = '/rust/api/examples';
  else if (name === 'contracts' || /^(generated-|reference-)/.test(name)) destination = '/rust/api/' + name;
  else if (name === 'overview') destination = suffix ? '/rust/docs/index' : '/rust/docs/';
  else if (rustGuides.includes(name)) destination = '/rust/docs/' + name;
  else return value;
  return destination + suffix + match[2];
}

export function rewriteFor(file) {
  const destination = canonicalLink('/' + file);
  return destination === '/' + file ? null : destination.slice(1);
}

export function surfaceOf(relativePath) {
  return relativePath.startsWith('python/') ? 'python' : relativePath.startsWith('rust/') ? 'rust' : null;
}

export function counterpart(relativePath, target) {
  const source = surfaceOf(relativePath);
  if (!source || source === target) return '/' + target + '/';
  const path = relativePath.replace(/\.md$/, '').replace(/\/index$/, '/');
  const pairs = [
    ['rust/', 'python/'], ['rust/docs/', 'python/docs/'],
    ['rust/docs/installation', 'python/docs/installation'], ['rust/docs/quickstart', 'python/docs/quickstart'],
    ['rust/docs/storage', 'python/docs/storage'], ['rust/docs/loading', 'python/docs/native-loader'],
    ['rust/docs/data-sources', 'python/docs/alignment'], ['rust/docs/accesses', 'python/docs/access-functions'],
    ['rust/docs/process-functions', 'python/docs/process-functions'], ['rust/docs/abi', 'python/docs/process-functions'],
    ['rust/docs/planning', 'python/docs/planning'],
    ['rust/docs/codecs', 'python/docs/codecs'],
    ['rust/api/', 'python/api/'], ['rust/api/reference-store', 'python/api/storage'],
    ['rust/api/reference-loader', 'python/api/loader'], ['rust/api/reference-scans', 'python/api/codecs'],
    ['rust/api/reference-opt', 'python/api/planning'], ['rust/api/reference-prof', 'python/api/statistics'],
  ];
  const pair = pairs.find(pair => pair[source === 'rust' ? 0 : 1] === path);
  return pair ? '/' + pair[target === 'rust' ? 0 : 1] + (pair[target === 'rust' ? 0 : 1].endsWith('/') ? '' : '.html') : '/' + target + '/';
}
