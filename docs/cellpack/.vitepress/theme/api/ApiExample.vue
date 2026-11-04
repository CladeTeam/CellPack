<script setup>
import { computed, nextTick, onBeforeUnmount, ref } from 'vue';
import { useData, withBase } from 'vitepress';
import { data as catalog } from './examples.data.js';
import SelectionDemo from './SelectionDemo.vue';
const props = defineProps({ example: { type: String, required: true } });
const { theme } = useData();
const example = computed(() => catalog.examples[props.example]);
const endpoint = computed(() => theme.value.playgroundUrl || withBase('/playground'));
const file = ref('main.rs'), host = ref(null), insights = ref(false), running = ref(false), result = ref(null);
const expanded = ref(false);
const notice = ref(''), analysis = ref('off'), copied = ref(false), output = ref('expected');
const selectedFile = computed(() => catalog.files[file.value === 'main.rs' ? example.value.main : example.value.common]);
const code = computed(() => selectedFile.value.code);
const html = computed(() => selectedFile.value.html);
const connected = computed(() => ['connected', 'local-only'].includes(analysis.value));
let viewer, connection, controller, analysisController, copyTimer, disposed = false, sequence = 0;
function closeAnalysis() { analysisController?.abort(); connection?.close(); connection = null; viewer?.destroy(); viewer = null; insights.value = false; analysis.value = 'off'; }
function selectFile(name) { closeAnalysis(); file.value = name; expanded.value = false; }
async function copy() {
  try { await navigator.clipboard.writeText(code.value); if (disposed) return; copied.value = true; clearTimeout(copyTimer); copyTimer = setTimeout(() => { copied.value = false; }, 2000); }
  catch { notice.value = 'Select the code to copy it.'; }
}
async function run() {
  if(running.value) return;
  const current = ++sequence;
  running.value = true; notice.value = ''; output.value = 'actual';
  controller = new AbortController();
  const timer = setTimeout(() => controller?.abort(), 60000);
  try {
    const response = await fetch(endpoint.value.replace(/\/$/, '') + '/run', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ example: props.example }), signal: controller.signal });
    if(!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.error || `HTTP ${response.status}`); }
    const body = await response.json();
    if(current !== sequence) return;
    result.value = body; notice.value = body.success ? 'Run completed.' : 'The example did not complete successfully.';
  } catch(error) {
    if(current === sequence) { notice.value = error.name === 'AbortError' ? 'Run stopped.' : 'Run service unavailable. Expected output is shown below.'; output.value = 'expected'; }
  } finally { clearTimeout(timer); if(current === sequence) running.value = false; }
}
async function toggleInsights() {
  if(insights.value) { closeAnalysis(); return; }
  insights.value = true; analysis.value = 'connecting'; notice.value = '';
  const attempt = new AbortController();
  analysisController = attempt;
  await nextTick();
  try {
    const [{ createViewer }, { connectAnalysis }] = await Promise.all([import('./editor.js'), import('./lsp.js')]);
    if(attempt.signal.aborted || disposed) return;
    viewer = createViewer(host.value, code.value);
    const connected = await connectAnalysis(viewer, endpoint.value, value => {
      if (!attempt.signal.aborted && !disposed) analysis.value = value;
    }, attempt.signal, props.example, file.value);
    if (attempt.signal.aborted || disposed) connected.close();
    else connection = connected;
  } catch {
    if (analysisController === attempt && !attempt.signal.aborted && !disposed) {
      closeAnalysis(); notice.value = 'LSP service unavailable. Syntax highlighting and fixed examples remain available.';
    }
  }
}
function download() {
  const url = URL.createObjectURL(new Blob([code.value], { type: 'text/plain' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = file.value; anchor.click(); URL.revokeObjectURL(url);
}
onBeforeUnmount(() => { disposed = true; ++sequence; clearTimeout(copyTimer); controller?.abort(); closeAnalysis(); });
</script>

<template>
  <section class="cp-example" :aria-label="example.title">
    <header class="cp-example-header">
      <div class="cp-example-eyebrow"><span class="cp-example-dot"></span> FIXED EXAMPLE <span>RUST / CPU</span></div>
      <h3>{{ example.title }}</h3>
      <p>{{ example.description }}</p>
    </header>
    <div class="cp-code-toolbar">
      <div class="cp-file-tabs" role="tablist" aria-label="Example files">
        <button v-for="name in ['main.rs', 'common.rs']" :key="name" role="tab" :aria-selected="file === name" @click="selectFile(name)">{{ name }}</button>
      </div>
      <div class="cp-code-actions">
        <button class="cp-lsp-toggle" :class="{ 'is-active': connected }" :disabled="analysis === 'connecting'" :aria-pressed="insights" title="Read-only type information and definitions within this file" @click="toggleInsights"><span class="cp-lsp-dot"></span>{{ analysis === 'connecting' ? 'Connecting' : connected ? 'LSP on' : 'LSP' }}</button>
        <button class="cp-code-icon" :aria-label="copied ? 'Copied' : 'Copy file'" :title="copied ? 'Copied' : 'Copy file'" @click="copy"><svg viewBox="0 0 20 20" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M12 7V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h3"/></svg></button>
        <button class="cp-code-icon" :aria-label="'Download ' + file" title="Download file" @click="download"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v9m-4-4 4 4 4-4M4 13v4h12v-4"/></svg></button>
      </div>
    </div>
    <div v-if="insights" ref="host" class="cp-readonly-viewer"></div>
    <div v-else class="cp-code-static" :class="{ 'is-expanded': expanded }" v-html="html"></div>
    <button v-if="!insights" class="cp-code-expand" :aria-expanded="expanded" @click="expanded = !expanded">{{ expanded ? 'Collapse code' : 'Show full example · ' + code.split('\n').length + ' lines' }}<span aria-hidden="true">{{ expanded ? '↑' : '↓' }}</span></button>
    <div v-if="connected" class="cp-insights-note">Hover for types · F12 or Ctrl/⌘ click for definitions in this file<span v-if="analysis === 'local-only'"> · Definition is outside this file.</span></div>
    <div class="cp-execution-bar">
      <div><span class="cp-output-label">OUTPUT</span><span class="cp-output-mode">{{ output === 'actual' && result ? 'Server result' : 'Expected result' }}</span></div>
      <button class="cp-run-button" :disabled="running" @click="run"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 3 8 5-8 5Z"/></svg>{{ running ? 'Running…' : 'Run example' }}</button>
    </div>
    <div class="cp-result-body">
      <pre v-if="output === 'actual' && result"><code>{{ result.stdout || result.stderr || '(no output)' }}{{ result.stdout && result.stderr ? '\n' + result.stderr : '' }}</code></pre>
      <pre v-else><code>{{ example.expected }}</code></pre>
      <div v-if="result" class="cp-result-meta"><span :class="{ 'is-success': result.success }">{{ result.success ? 'Completed' : 'Failed' }} · {{ result.elapsed_ms }} ms</span><button @click="output = output === 'expected' ? 'actual' : 'expected'">{{ output === 'expected' ? 'Show server result' : 'Show expected result' }}</button></div>
    </div>
    <p v-if="notice" role="status" class="cp-run-notice">{{ notice }}</p>
    <details v-if="props.example === 'selection'" class="cp-selection-details"><summary>Explore how row and column selection works</summary><SelectionDemo /></details>
  </section>
</template>
