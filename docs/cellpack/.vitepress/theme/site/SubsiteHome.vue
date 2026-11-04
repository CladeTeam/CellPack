<script setup>
import { computed } from 'vue';
import { withBase } from 'vitepress';
const props = defineProps({ area: { type: String, required: true } });
const python = computed(() => props.area === 'python');
const cards = computed(() => python.value ? [
  { title: 'Store and select matrices', text: 'Write compact matrices, compose lazy selections, and read NumPy or CSR outputs.', link: '/python/docs/quickstart.html', label: 'Start with a matrix' },
  { title: 'Prepare training data', text: 'Build a corpus, supply the sample order, and iterate named Tensor batches.', link: '/python/training/', label: 'Explore training examples' },
  { title: 'Connect AnnData', text: 'Use CellPack-backed matrices with AnnData and H5AD / H5CP workflows.', link: '/python/docs/anndata.html', label: 'Read the integration guide' },
] : [
  { title: 'Write and read matrices', text: 'Create a stream, prepare selections, and decode into caller-owned buffers.', link: '/rust/docs/quickstart.html', label: 'Run the quickstart' },
  { title: 'Deliver ordered batches', text: 'Compose sources, schemas, access sequences, and bounded output rings.', link: '/rust/docs/loading.html', label: 'Understand data loading' },
  { title: 'Explore the interfaces', text: 'Look up declarations, contracts, and fixed examples from the current crate.', link: '/rust/api/', label: 'Browse the Rust API' },
]);
</script>

<template>
  <section class="cp-subsite-home">
    <div class="cp-subsite-kicker">{{ python ? 'PYTHON LIBRARY' : 'RUST CORE' }}</div>
    <h1><img class="cp-subsite-logo" :src="withBase('/logo.svg')" width="64" height="64" alt="" />{{ python ? 'PyCellPack' : 'CellPack' }} <span>{{ python ? 'Python' : 'Rust' }}</span></h1>
    <p class="cp-subsite-intro">{{ python ? 'Compact matrix storage and native data loading, through a Python interface.' : 'Compact matrix streams, composable codecs, and ordered data loading for Rust applications.' }}</p>
    <div class="cp-subsite-links"><a :href="withBase('/' + area + '/docs/installation.html')">Installation →</a><a :href="withBase('/' + area + '/docs/')">Documentation →</a><a :href="withBase('/' + area + '/api/')">API reference →</a></div>
    <div class="cp-subsite-cards"><a v-for="card in cards" :key="card.link" :href="withBase(card.link)"><h2>{{ card.title }}</h2><p>{{ card.text }}</p><span>{{ card.label }} →</span></a></div>
    <div class="cp-subsite-footer"><p>{{ python ? 'Need the core crate or native integration details?' : 'Using Python for analysis or training?' }} <a :href="withBase(python ? '/rust/' : '/python/')">{{ python ? 'Open the Rust documentation' : 'Open PyCellPack' }} →</a></p><a :href="withBase('/')">← Back to the CellPack project</a></div>
  </section>
</template>
