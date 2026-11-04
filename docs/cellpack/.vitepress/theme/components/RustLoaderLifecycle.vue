<script setup>
import { computed, ref } from 'vue';

const selected = ref(0);
const stages = [
  { label: 'Prepare', state: 'Open', actor: 'Application', detail: 'prepare binds the sequence, ring and planning mode. It reserves this run generation; it does not start the producer.', consequence: 'Start production and consumption together with for_each_batch, or provide concurrent threads.' },
  { label: 'Fill', state: 'Writing', actor: 'Workers', detail: 'Workers write each position into the region for its schema. A slot becomes ready only after all required outputs in it are complete.', consequence: 'Workers may finish in a different order from the requested sequence; output positions stay fixed.' },
  { label: 'Read', state: 'Leased', actor: 'Consumer', detail: 'A BatchLease gives one consumer access to the completed slot. The producer cannot overwrite this slot while it is leased.', consequence: 'The producer can use other available slots. Holding every slot eventually blocks production.' },
  { label: 'Release', state: 'Reusable', actor: 'Consumer → workers', detail: 'Dropping the lease opens the slot for the next batch that maps to it. Finish every reader before releasing it.', consequence: 'A device copy remains a reader until completion. Copy output into independent storage when it must outlive the lease.' },
];
const stage = computed(() => stages[selected.value]);
</script>

<template>
  <figure class="loader-lifecycle" aria-label="Explore one output slot's lifetime">
    <figcaption>One slot, four responsibilities</figcaption>
    <div class="lifecycle-controls" role="group" aria-label="Choose a lifecycle stage">
      <button v-for="(item, index) in stages" :key="item.label" type="button" :aria-pressed="selected === index" @click="selected = index">
        <span aria-hidden="true">{{ index + 1 }}</span>{{ item.label }}
      </button>
    </div>
    <div class="lifecycle-body" aria-live="polite" aria-atomic="true">
      <div class="lifecycle-slot"><span>Slot state</span><strong>{{ stage.state }}</strong><span>{{ stage.actor }}</span></div>
      <div class="lifecycle-explanation"><p>{{ stage.detail }}</p><p>{{ stage.consequence }}</p></div>
    </div>
    <p class="lifecycle-formula">Batch <code>b</code> uses slot <code>b % slots</code>. All schemas share that slot's lifetime.</p>
  </figure>
</template>

<style scoped>
.loader-lifecycle { margin: 24px 0; padding: 20px; border: 1px solid var(--vp-c-divider); border-radius: 12px; background: var(--vp-c-bg-alt); }
figcaption { margin-bottom: 16px; font-family: var(--cp-font-heading); font-size: 17px; font-weight: 650; }
.lifecycle-controls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.lifecycle-controls button { display: flex; justify-content: center; align-items: center; gap: 7px; min-height: 44px; padding: 8px; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg-elv); color: var(--vp-c-text-2); font-size: 13px; cursor: pointer; }
.lifecycle-controls button span { font-family: var(--vp-font-family-mono); font-size: 11px; }
.lifecycle-controls button[aria-pressed="true"] { border-color: var(--cp-teal); background: var(--cp-teal-soft); color: var(--vp-c-text-1); font-weight: 650; }
.lifecycle-controls button:hover { border-color: var(--cp-teal); }
.lifecycle-controls button:focus-visible { outline: 2px solid var(--cp-teal); outline-offset: 3px; }
.lifecycle-body { display: grid; grid-template-columns: 130px minmax(0, 1fr); gap: 20px; align-items: start; margin-top: 20px; }
.lifecycle-slot { display: flex; flex-direction: column; gap: 8px; padding: 16px; border: 1px solid var(--vp-c-divider); border-top: 3px solid var(--cp-teal); border-radius: 8px; background: var(--vp-c-bg-elv); }
.lifecycle-slot span { color: var(--vp-c-text-2); font-size: 12px; }
.lifecycle-slot strong { font-family: var(--cp-font-heading); font-size: 19px; }
.loader-lifecycle .lifecycle-explanation p { margin: 0; font-size: 14px; line-height: 1.65; }
.loader-lifecycle .lifecycle-explanation p + p { margin-top: 12px; color: var(--vp-c-text-2); }
.loader-lifecycle .lifecycle-formula { margin: 18px 0 0; padding-top: 14px; border-top: 1px solid var(--vp-c-divider); color: var(--vp-c-text-2); font-size: 12px; line-height: 1.7; }
@media (max-width: 640px) {
  .loader-lifecycle { padding: 16px; }
  .lifecycle-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .lifecycle-body { grid-template-columns: minmax(0, 1fr); gap: 14px; }
  .lifecycle-slot { flex-direction: row; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; }
}
</style>
