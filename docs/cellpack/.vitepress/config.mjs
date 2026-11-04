import { defineConfig } from "vitepress";
import { readFileSync, readdirSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { canonicalLink, rewriteFor, surfaceOf } from "./routes.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const navigation = JSON.parse(readFileSync(path.join(root, "_catalog/navigation.json"), "utf8"));
const page = (text, name) => ({ text, link: canonicalLink("/" + name) });
const apiGroups = Object.entries(navigation).filter(([key]) => key !== "root").map(([key, value]) => ({
  text: value.title,
  link: canonicalLink("/" + value.reference),
  collapsed: true,
  items: value.items.map(item => ({ ...item, link: canonicalLink(item.link) })),
}));
const apiSidebar = [
  { text: "API reference", items: [page("Find an API", "api"), page("Runnable examples", "api-examples"), page("Data contracts", "contracts"), page("Errors and exports", "reference-errors")] },
  ...apiGroups,
];
const guideSidebar = [
  { text: "Getting started", items: [page("Overview", "overview"), page("Installation", "installation"), page("Quickstart", "quickstart"), page("Data model", "concepts")] },
  { text: "Guides", items: [
    page("Storage and selections", "storage"), page("Data sources and mapping", "data-sources"),
    page("Access sequences", "accesses"), page("Loading and ownership", "loading"),
    page("Cache planning", "planning"), page("Native process functions", "process-functions"),
    page("Codec components", "codecs"),
  ] },
  { text: "Contracts", items: [page("Data contracts", "data-contracts"), page("ABI", "abi")] },
  { text: "Reference", items: [page("API overview", "api"), page("Memory and performance", "performance"), page("Statistics and profiling", "profiling")] },
  { text: "Internals", items: [page("Architecture", "architecture"), page("File format", "format"), page("Troubleshooting", "troubleshooting"), page("Documentation development", "contributing")] },
];
const rtdBase = process.env.READTHEDOCS === "True"
  ? "/" + (process.env.READTHEDOCS_LANGUAGE || "en") + "/" + process.env.READTHEDOCS_VERSION + "/"
  : "/";
const base = process.env.CELLPACK_DOCS_BASE || rtdBase;
const rewrites = Object.fromEntries(readdirSync(root).filter(file => file.endsWith('.md') && rewriteFor(file)).map(file => [file, rewriteFor(file)]));
rewrites['playground/README.md'] = 'rust/docs/example-service.md';
const pythonDocs = [
  { text: 'Getting started', items: [
    { text: 'Overview', link: '/python/docs/' }, { text: 'Installation', link: '/python/docs/installation' }, { text: 'Quickstart', link: '/python/docs/quickstart' },
  ] },
  { text: 'Using PyCellPack', items: [
    { text: 'Matrix access and writes', link: '/python/docs/storage' }, { text: 'Gene alignment', link: '/python/docs/alignment' }, { text: 'Training DataLoader', link: '/python/docs/dataloader' },
    { text: 'Native NumPy loading', link: '/python/docs/native-loader' }, { text: 'Process functions', link: '/python/docs/process-functions' }, { text: 'Planning and diagnostics', link: '/python/docs/planning' }, { text: 'AnnData and H5CP', link: '/python/docs/anndata' }, { text: 'Native access functions', link: '/python/docs/access-functions' },
    { text: 'Shared and distributed loading', link: '/python/docs/distributed' },
    { text: 'Codec components', link: '/python/docs/codecs' },
  ] },
  { text: 'Reference', items: [{ text: 'API reference', link: '/python/api/' }, { text: 'Training', link: '/python/training/' }] },
];
const pythonApi = [
  { text: 'Python API', items: [
    { text: 'Overview and imports', link: '/python/api/' }, { text: 'Storage', link: '/python/api/storage' },
    { text: 'Training DataLoader', link: '/python/api/dataloader' }, { text: 'Native loader', link: '/python/api/loader' },
    { text: 'Access sequences and functions', link: '/python/api/access-functions' }, { text: 'Rings and lifecycle', link: '/python/api/rings' }, { text: 'Planning', link: '/python/api/planning' }, { text: 'Process functions and presets', link: '/python/api/process-functions' }, { text: 'Shared and distributed', link: '/python/api/distributed' }, { text: 'Statistics and costs', link: '/python/api/statistics' }, { text: 'Codecs', link: '/python/api/codecs' }, { text: 'AnnData and H5CP', link: '/python/api/anndata' },
  ] },
  { text: 'Continue', items: [{ text: 'Python documentation', link: '/python/docs/' }, { text: 'Training', link: '/python/training/' }] },
];
const pythonTraining = [{ text: 'Training', items: [
  { text: 'Choose an example', link: '/python/training/' }, { text: 'DataLoader and ordering', link: '/python/training/dataloader' },
  { text: 'Native process functions', link: '/python/training/process-functions' }, { text: 'Full training integration', link: '/python/training/integration' },
] }];

export default defineConfig({
  title: "CellPack",
  description: "Compact matrix storage and ordered data loading for billion-cell training.",
  lang: "en-US",
  useWebFonts: false,
  head: [["link", { rel: "icon", type: "image/svg+xml", href: base + "logo.svg" }]],
  base,
  // Compare cold loads against reuse of cached metadata without changing page content.
  metaChunk: process.env.CELLPACK_DOCS_META_CHUNK !== 'false',
  router: { prefetchLinks: false },
  transformHead({ assets }) {
    // The body and heading faces are needed above the fold on every surface.
    return ['dm-sans', 'space-grotesk'].flatMap(family => {
      const href = assets.find(asset => asset.includes(`${family}-latin-wght-normal.`) && asset.endsWith('.woff2'));
      return href ? [["link", { rel: "preload", href, as: "font", type: "font/woff2", crossorigin: "" }]] : [];
    });
  },
  rewrites,
  async buildEnd(config) {
    for (const [source, destination] of Object.entries(rewrites)) {
      const target = base + destination.replace(/index\.md$/, '').replace(/\.md$/, '.html');
      const file = path.join(config.outDir, source.replace(/\.md$/, '.html'));
      await mkdir(path.dirname(file), { recursive: true });
      await writeFile(file, `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex"><link rel="canonical" href="${target}"><meta http-equiv="refresh" content="0;url=${target}"><title>Documentation moved</title></head><body><a href="${target}">Continue to the documentation</a><script>location.replace(${JSON.stringify(target)}+location.search+location.hash)</script></body></html>`);
    }
  },
  transformPageData(data) {
    // Browser titles: "CellPack" for the homepage, "PyCellPack · Python" and
    // "CellPack · Rust" for the two landing pages, "<page> · <product>" elsewhere.
    const page = data.relativePath;
    const surface = surfaceOf(page);
    if (page.startsWith('python/api/') && page !== 'python/api/index.md') {
      data.frontmatter.outline ??= 2;
    }
    const landing = { "index.md": "CellPack", "python/index.md": "PyCellPack · Python", "rust/index.md": "CellPack · Rust" }[page];
    if (landing) {
      data.title = landing;
      data.titleTemplate = false;
    } else {
      data.titleTemplate = surface === "python" ? ":title · PyCellPack" : surface === "rust" ? ":title · CellPack Rust" : ":title · CellPack";
    }
  },
  outDir: process.env.CELLPACK_DOCS_OUTPUT || path.join(root, "_build/html"),
  cacheDir: path.join(root, "_build/vite-cache"),
  srcExclude: ["_*/**", "public/**", "node_modules/**", "examples/**", "README.md"],
  cleanUrls: false,
  ignoreDeadLinks: false,
  lastUpdated: false,
  sitemap: process.env.READTHEDOCS_CANONICAL_URL
    ? { hostname: new URL(process.env.READTHEDOCS_CANONICAL_URL).origin }
    : undefined,
  themeConfig: {
    playgroundUrl: process.env.CELLPACK_PLAYGROUND_URL || "",
    repositoryUrl: "https://github.com/CladeTeam/CellPack",
    logo: { src: "/logo.svg", alt: "" },
    logoLink: '/',
    surfaces: {
      rust: { siteTitle: 'CellPack', logoLink: '/rust/', subsite: 'rust', nav: [
        { text: 'Docs', link: '/rust/docs/', activeMatch: '^/rust/docs/' },
        { text: 'API', link: '/rust/api/', activeMatch: '^/rust/api/' },
      ] },
      python: { siteTitle: 'PyCellPack', logoLink: '/python/', subsite: 'python', nav: [
        { text: 'Docs', link: '/python/docs/', activeMatch: '^/python/docs/' },
        { text: 'API', link: '/python/api/', activeMatch: '^/python/api/' },
        { text: 'Training', link: '/python/training/', activeMatch: '^/python/training/' },
      ] },
    },
    nav: [
      { text: 'Rust', link: '/rust/' },
      { text: 'Python', link: '/python/' },
      { text: "Technical report", link: "/report" },
    ],
    sidebar: {
      '/rust/api/': apiSidebar,
      '/rust/docs/': guideSidebar,
      '/python/docs/': pythonDocs,
      '/python/api/': pythonApi,
      '/python/training/': pythonTraining,
      '/': [],
    },
    search: { provider: 'local', options: {
      _render(src, env, md) {
        const html = md.render(src, env);
        if (env.frontmatter?.search === false) return '';
        const canonical = rewrites[env.relativePath] || env.relativePath;
        const surface = surfaceOf(canonical);
        return surface ? html.replace(/(<h1\b[^>]*>)/, '$1' + (surface === 'python' ? 'Python · ' : 'Rust · ')) : html;
      },
    } },
    outline: { level: [2, 3], label: "On this page" },
    docFooter: { prev: "Previous page", next: "Next page" },
  },
  vite: {
    optimizeDeps: {
      include: ["@codemirror/state", "@codemirror/view", "@codemirror/lang-rust", "@codemirror/commands", "@codemirror/language"],
    },
    resolve: {
      alias: [{
        find: /^.*\/VPSidebarItem\.vue$/,
        replacement: fileURLToPath(new URL("./theme/components/VPSidebarItem.vue", import.meta.url)),
      }, {
        find: /^.*\/VPNavBarTitle\.vue$/,
        replacement: fileURLToPath(new URL('./theme/site/SubsiteTitle.vue', import.meta.url)),
      }],
    },
    server: {
      proxy: process.env.CELLPACK_PLAYGROUND_PROXY ? {
        [base + "playground"]: {
          target: process.env.CELLPACK_PLAYGROUND_PROXY,
          ws: true,
          rewrite: url => url.slice((base + "playground").length) || "/",
        },
      } : undefined,
    },
  },
  markdown: {
    math: true,
    config(md) {
      const linkOpen = md.renderer.rules.link_open;
      md.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
        const href = tokens[index].attrGet("href") || "";
        if (href && !/^(?:[a-z][\w+.-]*:|\/\/|#)/i.test(href) && /(?:\.md|\.html)?(?:[?#].*)?$/.test(href)) {
          const [, pathname, suffix] = href.match(/^([^?#]*)(.*)$/);
          if (!path.extname(pathname) || /\.(md|html)$/.test(pathname)) {
            const source = env.realPath || env.path || path.join(root, 'index.md');
            const relative = pathname.startsWith('/') ? pathname : '/' + path.relative(root, path.resolve(path.dirname(source), pathname)).replaceAll(path.sep, '/');
            const rewritten = relative === '/playground/README.md' ? '/rust/docs/example-service.md' : canonicalLink(relative);
            tokens[index].attrSet('href', rewritten + suffix);
          }
        }
        const staticRoot = href.startsWith("/downloads/");
        const hostedStatic = href.startsWith(base + "downloads/");
        if (staticRoot) tokens[index].attrSet("href", base + href.slice(1));
        if (staticRoot || hostedStatic) {
          tokens[index].attrSet("target", "_self");
        }
        return linkOpen ? linkOpen(tokens, index, options, env, renderer) : renderer.renderToken(tokens, index, options);
      };
    },
  },
});
