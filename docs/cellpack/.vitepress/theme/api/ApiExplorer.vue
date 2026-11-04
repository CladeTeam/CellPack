<script setup>
import { computed, ref, watch } from 'vue';
import { withBase } from 'vitepress';
import { data } from './api.data.js';
import { showDefinition } from './state.js';

const query = ref('');
const module = ref('all');
const kind = ref('all');
const items = Object.values(data.symbols).filter(item => item.kind !== 'method');
const searchText = new Map(items.map(item => [item.key, `${item.key} ${item.summary}`.toLowerCase()]));
const pageSize = 40;
const limit = ref(pageSize);
const modules = [...new Set(items.map(item => item.module))];
const kinds = [...new Set(items.map(item => item.kind))];
const results = computed(() => {
  const terms = query.value.trim().toLowerCase().split(/\s+/);
  return items.filter(item => (module.value === 'all' || item.module === module.value)
    && (kind.value === 'all' || item.kind === kind.value)
    && terms.every(term => searchText.get(item.key).includes(term)));
});
const visibleResults = computed(() => results.value.slice(0, limit.value));
watch([query, module, kind], () => { limit.value = pageSize; }, { flush: 'sync' });
</script>

<template>
  <div class="cp-api-explorer">
    <div class="cp-api-filters">
      <label class="cp-api-search">Find an interface
        <input v-model="query" type="search" placeholder="Name or purpose, e.g. ReadSelection" />
      </label>
      <label>Module<select v-model="module"><option value="all">All modules</option><option v-for="m in modules" :key="m">{{ m }}</option></select></label>
      <label>Kind<select v-model="kind"><option value="all">All kinds</option><option v-for="k in kinds" :key="k">{{ k }}</option></select></label>
    </div>
    <p role="status" class="cp-api-result-count">{{ results.length }} interfaces</p>
    <div class="cp-api-result-list">
      <article v-for="item in visibleResults" :key="item.key" class="cp-api-result">
        <div><a :href="withBase(item.href)"><code>{{ item.key }}</code></a><span class="cp-api-kind">{{ item.kind }}</span></div>
        <p>{{ item.summary || 'Open the declaration and module contract.' }}</p>
        <button class="cp-api-text-button" @click="showDefinition(item.key)">Quick view</button>
      </article>
      <button v-if="visibleResults.length < results.length" class="cp-api-more" @click="limit += pageSize">Show more interfaces · {{ visibleResults.length }} of {{ results.length }}</button>
      <p v-if="!results.length">No interfaces match. Try a shorter name or reset the filters.</p>
    </div>
  </div>
</template>
