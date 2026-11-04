<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { withBase } from 'vitepress';
import { data } from './api.data.js';
import { definition, showDefinition } from './state.js';
const dialog = ref(null);
const query = ref('');
const sourceCode = ref('');
const details = shallowRef(null);
const detailLoading = ref(false);
const detailError = ref('');
let sources;
let detailsRequest;
let detailsController;
let previousFocus;
let disposed = false;
const sourceController = new AbortController();
const symbols = Object.values(data.symbols);
const searchKeys = symbols.map(symbol => symbol.key.toLowerCase());
const symbolsByHref = new Map(symbols.map(symbol => [withBase(symbol.href).replace(/\.html(?=#|$)/, ''), symbol]));
const item = computed(() => {
  const current = definition.value;
  if (!current?.key) return current;
  const index = data.symbols[current.key];
  return index ? { ...details.value?.[current.key], ...index } : current;
});
const matches = computed(() => {
  const term = query.value.trim().toLowerCase();
  if (!term) return [];
  const found = [];
  for (let i = 0; i < symbols.length && found.length < 15; i++) {
    if (searchKeys[i].includes(term)) found.push(symbols[i]);
  }
  return found;
});
function cancelDetails() {
  detailsController?.abort();
  detailsController = undefined;
  detailsRequest = undefined;
  detailLoading.value = false;
  detailError.value = '';
}
function loadDetails() {
  if (details.value || !data.symbols[definition.value?.key]) return;
  if (detailsRequest) return detailsRequest;
  const controller = new AbortController();
  detailsController = controller;
  detailLoading.value = true;
  detailError.value = '';
  detailsRequest = fetch(withBase('/api/symbols.json'), { signal: controller.signal })
    .then(response => {
      if (!response.ok) throw new Error('Definition unavailable.');
      return response.json();
    })
    .then(value => {
      if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Definition unavailable.');
      if (!disposed && !controller.signal.aborted) details.value = value;
    })
    .catch(() => {
      if (!disposed && !controller.signal.aborted) detailError.value = 'Definition unavailable. Open the full documentation or try again.';
    })
    .finally(() => {
      if (detailsController === controller) {
        detailsController = undefined;
        detailsRequest = undefined;
        detailLoading.value = false;
      }
    });
  return detailsRequest;
}
watch(definition, async (value, _, onCleanup) => {
  let cancelled = false;
  onCleanup(() => { cancelled = true; });
  sourceCode.value = '';
  if (!value) cancelDetails();
  await nextTick();
  if (cancelled || disposed || !dialog.value) return;
  if (value) {
    if (!dialog.value.open) { previousFocus = document.activeElement; dialog.value.showModal(); }
    query.value = '';
    void loadDetails();
  } else if (dialog.value.open) { dialog.value.close(); previousFocus?.focus(); }
}, { immediate: true });
async function loadSource(event) {
  if(!event.target.open || !item.value?.source) return;
  const key = item.value.key;
  sourceCode.value = 'Loading source…';
  try {
    sources ||= fetch(withBase('/api/sources.json'), { signal: sourceController.signal }).then(response => { if(!response.ok) throw new Error('Source unavailable'); return response.json(); }).catch(error => { sources = null; throw error; });
    const all = await sources;
    if(!disposed && item.value?.key === key) sourceCode.value = all[key] || 'Source excerpt unavailable.';
  } catch(error) { if(!disposed && item.value?.key === key) sourceCode.value = error.message; }
}
function close() { definition.value = null; }
function intercept(event) {
  const link = event.target.closest('.cp-signature a.cp-type-link');
  if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const url = new URL(link.href);
  const href = url.pathname.replace(/\.html$/, '') + url.hash;
  const symbol = symbolsByHref.get(href);
  if (symbol) { event.preventDefault(); event.stopPropagation(); showDefinition(symbol.key); }
}
onMounted(() => document.addEventListener('click', intercept, true));
onBeforeUnmount(() => { disposed = true; cancelDetails(); sourceController.abort(); document.removeEventListener('click', intercept, true); definition.value = null; });
</script>

<template>
  <dialog ref="dialog" class="cp-definition" aria-labelledby="cp-definition-title" @cancel.prevent="close" @click="event => { if(event.target === dialog) close(); }">
    <div class="cp-definition-header"><strong id="cp-definition-title">Definition</strong><button aria-label="Close definition" @click="close">Close · Esc</button></div>
    <label>Look up another symbol<input v-model="query" type="search" placeholder="e.g. Loader::prepare" /></label>
    <ul v-if="matches.length" class="cp-definition-matches"><li v-for="match in matches" :key="match.key"><button @click="showDefinition(match.key)">{{ match.key }}</button></li></ul>
    <template v-if="item">
      <p class="cp-api-kind">{{ item.kind || 'Rust definition' }}</p>
      <h2>{{ item.key || item.name }}</h2>
      <p>{{ item.summary }}</p>
      <p v-if="detailLoading" role="status">Loading definition…</p>
      <div v-if="detailError"><p role="status">{{ detailError }}</p><button @click="loadDetails">Try again</button></div>
      <pre v-if="item.signature"><code>{{ item.signature }}</code></pre>
      <a v-if="item.href" :href="withBase(item.href)" @click="close">Open full API documentation →</a>
      <details v-if="item.source" :key="item.key" @toggle="loadSource"><summary>Source · {{ item.source.path }}:{{ item.source.line }}</summary><pre><code>{{ sourceCode }}</code></pre><small>Excerpt begins at line {{ item.source.start }}.</small></details>
      <pre v-if="item.code"><code>{{ item.code }}</code></pre>
    </template>
  </dialog>
</template>
