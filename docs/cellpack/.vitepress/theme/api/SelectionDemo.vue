<script setup>
import { computed, ref } from 'vue';
const matrix = [[0, 2, 0, 1], [3, 0, 4, 0], [0, 0, 5, 6], [7, 8, 0, 0]];
const input = ref('3, 0, 3');
const columns = ref([2, 0]);
const rows = computed(() => input.value.split(',').map(n => Number(n.trim())));
const error = computed(() => !input.value.trim() || input.value.split(',').some(n => !/^\s*[0-3]\s*$/.test(n)) || rows.value.length > 12
  ? 'Enter 1–12 row IDs from 0 to 3, separated by commas.' : !columns.value.length ? 'Keep at least one output column.' : '');
const map = computed(() => [0, 1, 2, 3].map(col => columns.value.includes(col) ? columns.value.indexOf(col) : 'UNMAPPED'));
function toggle(col) { columns.value = columns.value.includes(col) ? columns.value.filter(c => c !== col) : [...columns.value, col]; }
</script>

<template>
  <div class="cp-selection-demo">
    <strong>Explore the selection</strong><small>Instant illustration in your browser</small>
    <div class="cp-selection-controls">
      <label>Row IDs in output order<input v-model="input" inputmode="numeric" :aria-invalid="!!error" placeholder="3, 0, 3" /></label>
      <fieldset><legend>Keep source columns (click order = output order)</legend><label v-for="col in [0, 1, 2, 3]" :key="col"><input type="checkbox" :checked="columns.includes(col)" @change="toggle(col)" />Column {{ col }}</label></fieldset>
    </div>
    <p v-if="error" role="status">{{ error }}</p>
    <template v-else>
      <table><caption>Output · {{ rows.length }} rows × {{ columns.length }} columns</caption><thead><tr><th>Source row</th><th v-for="col in columns" :key="col">Source col {{ col }}</th></tr></thead><tbody><tr v-for="(row, pos) in rows" :key="pos"><th>{{ row }}</th><td v-for="col in columns" :key="col">{{ matrix[row][col] }}</td></tr></tbody></table>
      <p><code>source → output: [{{ map.join(', ') }}]</code></p>
      <p class="cp-selection-caption">This illustration runs in the browser. The fixed server example always uses rows [3, 0, 3].</p>
    </template>
  </div>
</template>
