<script setup>
import { computed } from 'vue';
import { withBase } from 'vitepress';
import { diagrams } from './doc-diagrams.mjs';

const props = defineProps({ name: { type: String, required: true } });
const diagram = computed(() => {
  const value = diagrams[props.name];
  if (!value) throw new Error(`Unknown documentation diagram: ${props.name}`);
  return value;
});
</script>

<template>
  <figure class="doc-diagram" :class="`doc-diagram-${diagram.kind}`" :aria-label="diagram.title">
    <figcaption>
      <span class="diagram-mark" aria-hidden="true"></span>
      {{ diagram.title }}
    </figcaption>

    <div v-if="diagram.kind === 'matrix'" class="diagram-matrix-wrap">
      <table class="diagram-matrix">
        <thead><tr><th scope="col">Row</th><th v-for="(_, index) in diagram.rows[0]" :key="index" scope="col">Col {{ index }}</th></tr></thead>
        <tbody><tr v-for="(row, index) in diagram.rows" :key="index"><th scope="row">{{ index }}</th><td v-for="(value, column) in row" :key="column" :class="{ 'matrix-nonzero': value !== 0 }">{{ value }}</td></tr></tbody>
      </table>
    </div>

    <div v-else-if="diagram.kind === 'accesses'" class="diagram-accesses">
      <div v-for="position in 3" :key="position" class="diagram-position">
        <div class="diagram-position-title">Position {{ position - 1 }}</div>
        <div v-for="schema in 2" :key="schema" class="diagram-access">
          <span>Schema {{ schema - 1 }}</span>
          <code>k = {{ (position - 1) * 2 + schema - 1 }}</code>
        </div>
      </div>
    </div>

    <div v-else-if="diagram.kind === 'alignment'" class="diagram-alignment">
      <div class="diagram-map-row">
        <div class="diagram-map-label">Vocabulary<span>Shared identity</span></div>
        <div class="diagram-genes">
          <div v-for="(gene, index) in diagram.genes" :key="gene" class="diagram-gene" :class="`gene-${index}`">
            <strong>{{ gene }}</strong><span>id {{ diagram.ids[index] }}</span>
          </div>
        </div>
      </div>
      <div v-for="(source, index) in diagram.sources" :key="index" class="diagram-map-row">
        <div class="diagram-map-label">Matrix {{ String.fromCharCode(65 + index) }}<span>Stored order</span></div>
        <div class="diagram-genes">
          <div v-for="(gene, column) in source" :key="column" class="diagram-gene" :class="`gene-${gene}`">
            <strong>{{ diagram.genes[gene] }}</strong><span>column {{ column }} → id {{ diagram.ids[gene] }}</span>
          </div>
        </div>
      </div>
      <div class="diagram-map-row diagram-map-output">
        <div class="diagram-map-label">Output panel<span>Requested order</span></div>
        <div class="diagram-genes">
          <div v-for="(gene, column) in diagram.output" :key="column" class="diagram-gene" :class="`gene-${gene}`">
            <strong>{{ diagram.genes[gene] }}</strong><span>output {{ column }} · id {{ diagram.ids[gene] }}</span>
          </div>
        </div>
      </div>
    </div>

    <component :is="diagram.kind === 'flow' ? 'ol' : 'div'" v-else class="diagram-blocks">
      <component :is="block.link ? 'a' : diagram.kind === 'flow' ? 'li' : 'div'" v-for="(block, index) in diagram.blocks" :key="block.title" :href="block.link ? withBase(block.link) : undefined" class="diagram-block">
        <span v-if="diagram.kind === 'flow' || diagram.kind === 'fields'" class="diagram-number">{{ String(index + 1).padStart(2, '0') }}</span>
        <div class="diagram-block-body">
          <strong class="diagram-block-title">{{ block.title }}<span v-if="block.link" class="diagram-link-arrow" aria-hidden="true">↗</span></strong>
          <code v-if="block.code">{{ block.code }}</code>
          <p>{{ block.detail }}</p>
        </div>
      </component>
    </component>

    <p v-if="diagram.note" class="diagram-note">{{ diagram.note }}</p>
  </figure>
</template>

<style scoped>
.doc-diagram {
  margin: 24px 0;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
  font-size: 14px;
  line-height: 1.6;
}
.doc-diagram figcaption { display: flex; align-items: baseline; gap: 9px; margin-bottom: 18px; font-weight: 650; }
.diagram-mark { flex: 0 0 8px; width: 8px; height: 8px; border-radius: 2px; background: var(--cp-teal); }
.doc-diagram .diagram-blocks { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0; padding: 0; list-style: none; }
.doc-diagram .diagram-block { display: flex; gap: 12px; min-width: 0; margin: 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg-elv); }
.diagram-block-body { min-width: 0; }
.doc-diagram a.diagram-block { color: var(--vp-c-text-1); text-decoration: none; transition: border-color .15s; }
.doc-diagram a.diagram-block:hover { border-color: var(--cp-teal); }
.doc-diagram a.diagram-block:focus-visible { outline: 2px solid var(--cp-teal); outline-offset: 3px; }
.diagram-link-arrow { margin-left: 8px; color: var(--cp-teal); }
.diagram-block-title { display: block; font-family: var(--cp-font-heading); font-size: 16px; font-weight: 600; line-height: 1.4; }
.doc-diagram p { margin: 8px 0 0; color: var(--vp-c-text-2); font-size: 13px; line-height: 1.65; }
.doc-diagram code { display: inline-block; max-width: 100%; margin-top: 8px; white-space: normal; overflow-wrap: anywhere; font-size: 12px; }
.diagram-number { flex-shrink: 0; color: var(--cp-teal); font-family: var(--vp-font-family-mono); font-size: 12px; line-height: 23px; }
.doc-diagram .diagram-note { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--vp-c-divider); }
.doc-diagram-fields .diagram-blocks { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.doc-diagram-fields .diagram-block { border-top: 3px solid var(--cp-teal); }
.diagram-accesses { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.diagram-position { min-width: 0; padding: 12px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg-elv); }
.diagram-position-title { margin-bottom: 12px; font-weight: 600; }
.diagram-access { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 4px 8px; padding: 8px; border-radius: 4px; background: var(--cp-teal-soft); }
.diagram-access + .diagram-access { margin-top: 8px; background: var(--cp-accent-soft); }
.diagram-access code { margin: 0; color: var(--vp-c-text-1); background: none; }
.diagram-map-row { display: grid; grid-template-columns: 110px minmax(0, 1fr); align-items: center; gap: 12px; padding: 12px 0; }
.diagram-map-row + .diagram-map-row { border-top: 1px solid var(--vp-c-divider); }
.diagram-map-label { font-size: 13px; font-weight: 600; }
.diagram-map-label span { display: block; color: var(--vp-c-text-2); font-size: 12px; font-weight: 400; }
.diagram-genes { display: flex; flex-wrap: wrap; gap: 8px; }
.diagram-gene { flex: 1 1 100px; padding: 8px 10px; border: 1px solid var(--vp-c-border); border-radius: 6px; background: var(--vp-c-bg-elv); }
.diagram-gene strong { display: block; font-family: var(--vp-font-family-mono); font-size: 12px; }
.diagram-gene span { display: block; margin-top: 3px; font-size: 12px; color: var(--vp-c-text-2); }
.gene-0 { border-top: 3px solid var(--cp-teal); }
.gene-1 { border-top: 3px solid var(--cp-accent); }
.gene-2 { border-top: 3px solid var(--vp-c-text-3); }
.diagram-map-output { margin-top: 4px; }
.diagram-matrix-wrap { overflow-x: auto; }
.doc-diagram .diagram-matrix { display: table; width: 100%; margin: 0; border-collapse: separate; border-spacing: 4px; font-family: var(--vp-font-family-mono); font-size: 13px; }
.diagram-matrix tr { background: none; border: 0; }
.diagram-matrix th, .diagram-matrix td { padding: 8px; border: 0; border-radius: 4px; text-align: center; }
.diagram-matrix th { color: var(--vp-c-text-2); font-size: 12px; font-weight: 400; }
.diagram-matrix td { background: var(--vp-c-bg-elv); color: var(--vp-c-text-2); }
.diagram-matrix td.matrix-nonzero { background: var(--cp-teal-soft); color: var(--cp-teal); font-weight: 600; }
@media (prefers-reduced-motion: reduce) { .doc-diagram a.diagram-block { transition: none; } }
@media (max-width: 640px) {
  .doc-diagram { padding: 16px; }
  .doc-diagram .diagram-blocks, .diagram-accesses { grid-template-columns: minmax(0, 1fr); }
  .diagram-map-row { grid-template-columns: minmax(0, 1fr); gap: 8px; }
  .diagram-map-label span { display: inline; margin-left: 8px; }
  .diagram-gene { flex-basis: 75px; }
}
</style>
