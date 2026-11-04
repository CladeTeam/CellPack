<template>
  <figure class="store-selection" aria-label="Selection example: source columns zero and two become output columns one and zero; requested rows three, zero, three retain their order">
    <figcaption>One map, two independent orders</figcaption>
    <div class="selection-map">
      <div v-for="column in columns" :key="column.source" class="selection-column" :class="{ dropped: column.target === null }">
        <span>Source column {{ column.source }}</span>
        <strong>{{ column.target === null ? 'Dropped' : `Output column ${column.target}` }}</strong>
      </div>
    </div>
    <div class="selection-result">
      <div class="selection-request">
        <span class="selection-label">Row request</span>
        <code>[3, 0, 3]</code>
        <p>Rows follow the request. Columns follow the map. Repeated rows remain repeated.</p>
      </div>
      <table>
        <thead><tr><th scope="col">Source row</th><th scope="col">Output 0<br><small>source 2</small></th><th scope="col">Output 1<br><small>source 0</small></th></tr></thead>
        <tbody><tr v-for="(row, index) in rows" :key="index"><th scope="row">{{ row.source }}</th><td>{{ row.values[0] }}</td><td>{{ row.values[1] }}</td></tr></tbody>
      </table>
    </div>
    <p class="selection-note">Fixture from the runnable storage example. Dense output is row-major: [0, 7, 0, 0, 0, 7].</p>
  </figure>
</template>

<script setup>
const columns = [{ source: 0, target: 1 }, { source: 1, target: null }, { source: 2, target: 0 }, { source: 3, target: null }];
const rows = [{ source: 3, values: [0, 7] }, { source: 0, values: [0, 0] }, { source: 3, values: [0, 7] }];
</script>

<style scoped>
.store-selection { margin: 24px 0; padding: 20px; border: 1px solid var(--vp-c-divider); border-radius: 12px; background: var(--vp-c-bg-alt); color: var(--vp-c-text-1); }
.store-selection figcaption { margin-bottom: 16px; font-family: var(--cp-font-heading); font-weight: 600; }
.selection-map { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.selection-column { min-width: 0; padding: 12px; border: 1px solid var(--vp-c-divider); border-top: 3px solid var(--cp-teal); border-radius: 6px; background: var(--vp-c-bg-elv); }
.selection-column.dropped { border-top-color: var(--vp-c-text-3); }
.selection-column span, .selection-label { display: block; font-size: 12px; color: var(--vp-c-text-2); }
.selection-column strong { display: block; margin-top: 4px; font-size: 13px; font-weight: 600; }
.selection-result { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.5fr); align-items: center; gap: 20px; margin-top: 20px; }
.selection-request code { display: inline-block; margin-top: 8px; font-size: 16px; }
.store-selection p { color: var(--vp-c-text-2); font-size: 13px; line-height: 1.65; }
.store-selection table { display: table; width: 100%; margin: 0; border-collapse: separate; border-spacing: 3px; font-family: var(--vp-font-family-mono); font-size: 13px; }
.store-selection tr { border: 0; background: none; }
.store-selection th, .store-selection td { padding: 8px; border: 0; border-radius: 4px; text-align: center; }
.store-selection th { font-size: 12px; font-weight: 400; color: var(--vp-c-text-2); }
.store-selection td { background: var(--vp-c-bg-elv); }
.store-selection small { font-size: 11px; }
.store-selection .selection-note { margin: 16px 0 0; padding-top: 12px; border-top: 1px solid var(--vp-c-divider); }
@media (max-width: 640px) { .store-selection { padding: 16px; } .selection-map { grid-template-columns: repeat(2, minmax(0, 1fr)); } .selection-result { grid-template-columns: minmax(0, 1fr); gap: 8px; } }
</style>
