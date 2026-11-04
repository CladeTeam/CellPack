<script setup>
import { computed, ref, watch } from "vue";
import { withBase } from "vitepress";
import { useDiagramPlayback } from "./useDiagramPlayback.js";
const props = defineProps({ data: { type: Object, required: true } });
const selected = ref("Lance");
const gpuCount = ref(1);
const duration = 2200;
const { root, elapsed, playing, ended, reduced, state, toggle, restart } =
  useDiagramPlayback(duration, { reducedAtEnd: true, threshold: 0.18 });
const names = [
  {
    title: "Single-cell loaders",
    methods: [
      "AnnLoader",
      "MappedCollection",
      "scDataset*",
      "annbatch*",
      "TileDB*",
      "cellarium*",
      "BioNeMo",
    ],
  },
  {
    title: "General formats",
    methods: ["memmap", "HF Datasets", "Parquet", "Lance", "Zarr"],
  },
];
const methods = computed(() =>
  names.map((group) => ({
    ...group,
    methods: group.methods
      .map((name) => props.data.loaders.find((method) => method.name === name))
      .filter(Boolean),
  })),
);
const loader = computed(() =>
  props.data.loaders.find((method) => method.name === selected.value),
);
const vectorModels = ["scVI", "CPA (Tahoe)", "CPA (Replogle)", "GEARS"];
const groups = computed(() => [
  {
    title: "Other model input representations",
    models: props.data.models
      .filter((model) => !vectorModels.includes(model.name))
      .sort((a, b) => b.rate - a.rate),
  },
  {
    title: "Expression-vector inputs",
    models: props.data.models
      .filter((model) => vectorModels.includes(model.name))
      .sort((a, b) => b.rate - a.rate),
  },
]);
const allModels = computed(() => groups.value.flatMap((group) => group.models));
const index = (model) =>
  allModels.value.findIndex((item) => item.name === model.name);
const position = (rate) =>
  Math.max(0, Math.min(100, (Math.log10(rate) - 2) * 20));
const rate = (value) =>
  value >= 1e6
    ? `${(value / 1e6).toFixed(2)}M`
    : value >= 1e3
      ? `${(value / 1e3).toFixed(1)}K`
      : value.toFixed(0);
const displayName = (method) => method.name.replace(/\*$/, "");
function entry(model) {
  const value = Math.max(
    0,
    Math.min(1, (elapsed.value - index(model) * 45) / 650),
  );
  return 1 - Math.pow(1 - value, 3);
}
function overflow(model) {
  return Math.max(
    0,
    position(model.rate * gpuCount.value) * entry(model) -
      position(loader.value.rate),
  );
}
watch([selected, gpuCount], () => {
  if (reduced.value) {
    elapsed.value = duration;
    playing.value = false;
  } else restart();
});
</script>

<template>
  <div class="story-wrap supply-comparison">
    <div class="section-intro">
      <span class="story-kicker">02 / THE THROUGHPUT GAP</span>
      <h2>Can the loader keep<br />the model fed?</h2>
      <p>
        Choose a loading method. Its grey supply line shows how its delivery
        rate compares with the model’s data demand.
      </p>
    </div>
    <div class="supply-method-groups">
      <div
        v-for="group in methods"
        :key="group.title"
        class="supply-method-group"
      >
        <h3>{{ group.title }}</h3>
        <div role="group" :aria-label="group.title">
          <button
            v-for="method in group.methods"
            :key="method.name"
            type="button"
            :aria-pressed="selected === method.name"
            @click="selected = method.name"
          >
            <span
              >{{ displayName(method) }}<sup v-if="method.block">*</sup></span
            ><small>{{ rate(method.rate) }}/s</small>
          </button>
        </div>
      </div>
    </div>
    <div class="supply-chart-options">
      <span>Model demand · cells/s · log scale</span>
      <div role="group" aria-label="GPU demand">
        <button
          type="button"
          :aria-pressed="gpuCount === 1"
          @click="gpuCount = 1"
        >
          1 H200 · measured</button
        ><button
          type="button"
          :aria-pressed="gpuCount === 8"
          @click="gpuCount = 8"
        >
          8× demand · projection
        </button>
      </div>
    </div>
    <div
      ref="root"
      class="supply-chart"
      role="button"
      tabindex="0"
      :data-state="state"
      :aria-label="`${ended ? 'Replay' : playing ? 'Pause' : 'Continue'} model demand animation`"
      :aria-pressed="playing"
      @click="toggle"
      @keydown.enter.prevent="toggle"
      @keydown.space.prevent="toggle"
    >
      <div class="supply-axis">
        <span />
        <div>
          <span
            v-for="tick in ['100', '1K', '10K', '100K', '1M', '10M']"
            :key="tick"
            >{{ tick }}</span
          >
        </div>
        <span />
      </div>
      <div
        v-for="group in groups"
        :key="group.title"
        class="supply-model-group"
      >
        <h3>{{ group.title }}</h3>
        <div
          v-for="model in group.models"
          :key="model.name"
          class="supply-demand-row"
          :style="{
            opacity: 0.4 + entry(model) * 0.6,
            transform: `translateY(${(1 - entry(model)) * 8}px)`,
          }"
        >
          <span>{{ model.name }}</span>
          <div class="supply-demand-track">
            <i
              class="supply-demand-bar"
              :style="{
                width: position(model.rate * gpuCount) + '%',
                transform: `scaleX(${entry(model)})`,
              }"
            /><i
              class="supply-demand-overflow"
              :style="{
                left: position(loader.rate) + '%',
                width: overflow(model) + '%',
              }"
            />
          </div>
          <span class="supply-model-rate">{{
            rate(model.rate * gpuCount)
          }}</span>
        </div>
      </div>
      <div class="supply-baseline-area" aria-hidden="true">
        <div
          class="supply-method-baseline"
          :style="{ left: position(loader.rate) + '%' }"
        >
          <div class="supply-baseline-name">
            <strong>{{ displayName(loader) }}</strong
            ><span>{{ rate(loader.rate) }} cells/s</span>
          </div>
        </div>
      </div>
    </div>
    <p class="supply-selection-description">
      {{ displayName(loader) }} ·
      {{ loader.block ? "native block/chunk sampling" : "random sampling" }}.
      Click the chart to pause, continue, or replay.
    </p>
    <p class="source-note">
      Models follow Figure 1’s order and grouping. Loading methods follow the
      report’s configuration table. Model demand: GPU-resident synthetic inputs
      on one H200; 8× is a linear projection. Loader supply: single NVMe, 16
      CPUs, 20 GB cache, best variant per loader, mean over seeds. * Native
      block/chunk sampling. This compares data supply and demand, not end-to-end
      training speed.
      <a :href="withBase('/research/home/model_compute.csv')">Model data</a> ·
      <a :href="withBase('/research/home/loader_methods.csv')">Loader data</a>
    </p>
  </div>
</template>

<style scoped>
.supply-comparison {
  --supply-grey: #989f9c;
  --demand-blue: #327bc3;
}
:global(.dark) .supply-comparison {
  --supply-grey: #8f9994;
  --demand-blue: #77b4ef;
}
.supply-method-groups {
  display: grid;
  gap: 18px;
  margin-bottom: 34px;
}
.supply-method-group {
  display: grid;
  grid-template-columns: 145px 1fr;
  align-items: start;
  gap: 14px;
}
.supply-method-group h3 {
  font-size: 14px;
  font-weight: 550;
  padding-top: 10px;
  color: var(--vp-c-text-2);
}
.supply-method-group > div {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.supply-method-group button {
  display: flex;
  align-items: baseline;
  gap: 12px;
  min-height: 42px;
  padding: 8px 12px;
  border: 1px solid var(--vp-c-border);
  border-radius: 5px;
  background: var(--vp-c-bg);
  font-size: 13px;
}
.supply-method-group small {
  color: var(--vp-c-text-2);
  font: 11px var(--vp-font-family-mono);
}
.supply-method-group button[aria-pressed="true"] {
  border-color: var(--supply-grey);
  background: var(--vp-c-bg-soft);
  box-shadow: inset 0 -2px var(--supply-grey);
}
.supply-method-group sup {
  margin-left: 2px;
}
.supply-method-group button:hover {
  border-color: var(--vp-c-text-2);
}
.supply-chart-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 22px;
  color: var(--vp-c-text-2);
  font-size: 13px;
}
.supply-chart-options > div {
  display: flex;
  gap: 8px;
}
.supply-chart-options button {
  padding: 8px 12px;
  min-height: 40px;
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  font-size: 12px;
}
.supply-chart-options button[aria-pressed="true"] {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-text-2);
  color: var(--vp-c-text-1);
}
.supply-chart {
  user-select: none;
  touch-action: manipulation;
  --model-column: 145px;
  --rate-column: 95px;
  --plot-gap: 14px;
  position: relative;
  padding-bottom: 72px;
  cursor: pointer;
  border-radius: 6px;
}
.supply-chart:focus-visible {
  outline: 2px solid var(--cp-teal);
  outline-offset: 6px;
}
.supply-axis,
.supply-demand-row {
  display: grid;
  grid-template-columns: var(--model-column) minmax(0, 1fr) var(--rate-column);
  gap: var(--plot-gap);
  align-items: center;
}
.supply-axis > div {
  display: flex;
  justify-content: space-between;
  font: 12px var(--vp-font-family-mono);
  color: var(--vp-c-text-2);
}
.supply-model-group h3 {
  margin-block: 22px 10px;
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 500;
}
.supply-demand-row {
  min-height: 43px;
  font-size: 14px;
}
.supply-demand-track {
  position: relative;
  height: 43px;
  background: repeating-linear-gradient(
    90deg,
    var(--vp-c-divider) 0 1px,
    transparent 1px 20%
  );
}
.supply-demand-bar {
  display: block;
  position: absolute;
  top: 13px;
  height: 17px;
  background: var(--demand-blue);
  transform-origin: left;
}
.supply-demand-overflow {
  position: absolute;
  top: 13px;
  height: 17px;
  background: var(--story-danger);
  pointer-events: none;
}
.supply-model-rate {
  font: 13px var(--vp-font-family-mono);
}
.supply-baseline-area {
  position: absolute;
  left: calc(var(--model-column) + var(--plot-gap));
  right: calc(var(--rate-column) + var(--plot-gap));
  top: 35px;
  bottom: 67px;
  pointer-events: none;
}
.supply-method-baseline {
  position: absolute;
  top: 0;
  bottom: 0;
  border-left: 2px dashed var(--supply-grey);
  transition: left 300ms ease;
}
.supply-baseline-name {
  position: absolute;
  top: calc(100% + 12px);
  left: -1px;
  transform: translateX(-50%);
  display: grid;
  gap: 5px;
  text-align: center;
  white-space: nowrap;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.supply-baseline-name strong {
  font-weight: 550;
}
.supply-baseline-name span {
  font: 11px var(--vp-font-family-mono);
}
.supply-selection-description {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}
@media (max-width: 640px) {
  .supply-method-group {
    grid-template-columns: 1fr;
    gap: 8px;
  }
  .supply-method-group h3 {
    padding-top: 0;
  }
  .supply-method-group button {
    font-size: 12px;
    padding-inline: 9px;
    gap: 8px;
  }
  .supply-chart {
    --model-column: 99px;
    --rate-column: 51px;
    --plot-gap: 7px;
  }
  .supply-demand-row {
    font-size: 12px;
  }
  .supply-model-rate {
    font-size: 11px;
  }
  .supply-axis > div {
    font-size: 10px;
  }
  .supply-baseline-name {
    font-size: 11px;
  }
  .supply-baseline-name span {
    font-size: 10px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .supply-method-baseline {
    transition: none;
  }
}
</style>
