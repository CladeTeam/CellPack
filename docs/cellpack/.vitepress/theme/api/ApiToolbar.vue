<script setup>
import { onBeforeUnmount, ref } from 'vue';
import { data } from './api.data.js';
import { showDefinition } from './state.js';
const props = defineProps({ symbol: { type: String, required: true } });
const copied = ref(false);
let copyTimer;
async function copy() {
  try {
    await navigator.clipboard.writeText('cellpack::' + props.symbol);
    copied.value = true;
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { copied.value = false; }, 2000);
  } catch { copied.value = false; }
}
onBeforeUnmount(() => clearTimeout(copyTimer));
</script>

<template>
  <div class="cp-api-toolbar">
    <span class="cp-api-kind">{{ data.symbols[symbol]?.kind }}</span>
    <button @click="copy">{{ copied ? 'Copied' : 'Copy import path' }}</button>
    <button @click="showDefinition(symbol)">Definition &amp; source</button>
  </div>
</template>
