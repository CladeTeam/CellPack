<script setup>
import { computed, ref } from 'vue';

const selected = ref(0);
const stages = [
  { label: 'Acquire Batch', batch: true, array: false, held: true, detail: 'The completed slot is leased by the Batch. Workers may fill other slots but cannot overwrite this one.' },
  { label: 'Export view', batch: true, array: true, held: true, detail: 'batch["X"] exports a NumPy view. The Batch and the view now both keep the same slot alive.' },
  { label: 'Release Batch', batch: false, array: true, held: true, detail: 'batch.release() drops the Batch reference. The view is still valid, and workers still cannot reuse its slot.' },
  { label: 'Drop final view', batch: false, array: false, held: false, detail: 'After the last exported view or sharing reader is gone, the slot can be reused. A copied array is independent.' },
];
const stage = computed(() => stages[selected.value]);
</script>

<template>
  <figure class="ring-lease" aria-label="Explore how a Batch and its NumPy view retain one output slot">
    <figcaption>One slot, two ways to retain it</figcaption>
    <div class="lease-controls" role="group" aria-label="Choose an ownership stage">
      <button v-for="(item, index) in stages" :key="item.label" type="button" :aria-pressed="selected === index" @click="selected = index">
        <span aria-hidden="true">{{ index + 1 }}</span>{{ item.label }}
      </button>
    </div>
    <div class="lease-state" aria-live="polite" aria-atomic="true">
      <div class="lease-owners">
        <div class="lease-owner" :class="{ retains: stage.batch }"><span>Batch reference</span><strong>{{ stage.batch ? 'Retains slot' : 'Released' }}</strong></div>
        <div class="lease-owner" :class="{ retains: stage.array }"><span>Exported view</span><strong>{{ stage.array ? 'Retains slot' : 'No view' }}</strong></div>
      </div>
      <div class="lease-result"><code>held = {{ stage.held ? 'True' : 'False' }}</code><p>{{ stage.detail }}</p></div>
    </div>
    <p class="lease-note">Illustrative ownership trace. A LeaseObserver reports the state without retaining the slot.</p>
  </figure>
</template>

<style scoped>
.ring-lease { margin: 24px 0; padding: 20px; border: 1px solid var(--vp-c-divider); border-radius: 12px; background: var(--vp-c-bg-alt); color: var(--vp-c-text-1); }
.ring-lease figcaption { margin-bottom: 16px; font-family: var(--cp-font-heading); font-size: 17px; font-weight: 650; }
.lease-controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.lease-controls button { display: flex; align-items: center; gap: 9px; min-height: 44px; padding: 10px 12px; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg-elv); color: var(--vp-c-text-2); font-size: 13px; text-align: left; cursor: pointer; }
.lease-controls button span { flex: none; font-family: var(--vp-font-family-mono); font-size: 11px; }
.lease-controls button[aria-pressed="true"] { border-color: var(--cp-teal); background: var(--cp-teal-soft); color: var(--vp-c-text-1); font-weight: 650; }
.lease-controls button:hover { border-color: var(--cp-teal); }
.lease-controls button:focus-visible { outline: 2px solid var(--cp-teal); outline-offset: 3px; }
.lease-state { margin-top: 18px; }
.lease-owners { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.lease-owner { min-width: 0; padding: 14px; border: 1px solid var(--vp-c-divider); border-top: 3px solid var(--vp-c-text-3); border-radius: 7px; background: var(--vp-c-bg-elv); }
.lease-owner.retains { border-top-color: var(--cp-teal); }
.lease-owner span { display: block; color: var(--vp-c-text-2); font-size: 12px; }
.lease-owner strong { display: block; margin-top: 6px; font-family: var(--cp-font-heading); font-size: 16px; }
.lease-result { margin-top: 16px; }
.ring-lease .lease-result p { margin: 8px 0 0; font-size: 14px; line-height: 1.65; }
.ring-lease .lease-note { margin: 16px 0 0; padding-top: 12px; border-top: 1px solid var(--vp-c-divider); color: var(--vp-c-text-2); font-size: 12px; line-height: 1.65; }
@media (max-width: 640px) { .ring-lease { padding: 16px; } .lease-controls, .lease-owners { grid-template-columns: minmax(0, 1fr); } }
</style>
