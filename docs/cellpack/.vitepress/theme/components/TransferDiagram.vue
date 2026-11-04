<script setup>
import { computed, useId } from "vue";
const props = defineProps({ kind: String, stage: Number, progress: Number });
const marker = `transfer-${useId()}`;
const t = computed(() => Math.min(1, props.progress * 1.25));
const lerp = (a, b, p) => a + (b - a) * p;
const request = computed(() =>
  props.stage > 0
    ? { x: 90, y: 66 }
    : t.value < 0.8
      ? { x: lerp(476, 90, t.value / 0.8), y: 25 }
      : { x: 90, y: lerp(25, 66, (t.value - 0.8) / 0.2) },
);
</script>

<template>
  <div class="transfer-diagram" :data-stage="stage">
    <div class="transfer-request-label">
      access(cell 07)<span>{{
        kind === "mmap"
          ? "cold page fault → storage read"
          : "locate → read → decompress"
      }}</span>
    </div>
    <svg
      viewBox="0 0 560 260"
      role="img"
      :aria-label="
        kind === 'mmap'
          ? 'Pages travel from storage to the page cache, then cell 07 reaches the batch'
          : 'A whole chunk travels to the decoder; only cell 07 continues to the batch'
      "
    >
      <defs>
        <marker
          :id="marker"
          viewBox="0 0 8 8"
          refX="6"
          refY="4"
          markerWidth="5"
          markerHeight="5"
          orient="auto"
        >
          <path d="m1 1 5 3-5 3" fill="none" stroke="currentColor" />
        </marker>
      </defs>
      <path
        d="M476 25H90V65"
        class="transfer-request-path"
        :marker-end="`url(#${marker})`"
      />
      <g
        v-if="stage === 0"
        class="transfer-request-packet"
        :transform="`translate(${request.x}, ${request.y})`"
      >
        <circle r="12" />
        <text y="5" text-anchor="middle">07</text>
      </g>
      <rect
        x="16"
        y="67"
        width="148"
        height="139"
        rx="7"
        class="transfer-node"
      />
      <rect
        x="216"
        y="67"
        width="148"
        height="139"
        rx="7"
        class="transfer-node"
      />
      <rect
        x="416"
        y="67"
        width="128"
        height="139"
        rx="7"
        class="transfer-node"
      />
      <path
        d="M166 137H213M366 137H413"
        class="transfer-data-path"
        :marker-end="`url(#${marker})`"
      />
      <g
        v-for="n in 12"
        :key="n"
        :transform="`translate(${29 + ((n - 1) % 4) * 31}, ${95 + Math.floor((n - 1) / 4) * 31})`"
      >
        <rect
          width="24"
          height="23"
          rx="2"
          :class="n === 7 ? 'transfer-target' : 'transfer-source-cell'"
        />
        <text x="12" y="16" text-anchor="middle">
          {{ String(n).padStart(2, "0") }}
        </text>
      </g>
      <g
        v-if="stage === 1"
        class="transfer-read-packet"
        :transform="`translate(${lerp(154, 235, t)}, 114)`"
      >
        <template v-if="kind === 'chunk'">
          <rect
            v-for="n in 12"
            :key="n"
            :x="((n - 1) % 4) * 10"
            :y="Math.floor((n - 1) / 4) * 11"
            width="8"
            height="9"
            :class="n === 7 ? 'transfer-target' : 'transfer-extra-byte'"
          />
        </template>
        <template v-else>
          <rect width="20" height="34" rx="2" class="transfer-page" />
          <rect x="24" width="20" height="34" rx="2" class="transfer-page" />
        </template>
      </g>
      <g v-if="kind === 'chunk' && stage >= 2">
        <g
          v-for="n in 12"
          :key="n"
          :transform="`translate(${229 + ((n - 1) % 4) * 31}, ${95 + Math.floor((n - 1) / 4) * 31 + (stage === 3 && n !== 7 ? t * 35 : 0)})`"
          :opacity="stage === 2 ? t : n === 7 ? 1 : 1 - t"
        >
          <rect
            width="24"
            height="23"
            rx="2"
            :class="n === 7 ? 'transfer-target' : 'transfer-decoded-cell'"
          />
          <text x="12" y="16" text-anchor="middle">
            {{ String(n).padStart(2, "0") }}
          </text>
        </g>
      </g>
      <g
        v-else-if="kind === 'mmap' && stage >= 2"
        :opacity="stage === 2 ? t : 1"
      >
        <rect
          x="235"
          y="102"
          width="44"
          height="58"
          rx="3"
          class="transfer-page"
        />
        <rect
          x="293"
          y="102"
          width="44"
          height="58"
          rx="3"
          class="transfer-page"
        />
        <path
          d="M243 113h28m-28 10h28m-28 10h28m30-20h28m-28 10h28m-28 10h28"
          class="transfer-page-lines"
        />
      </g>
      <g v-if="stage === 2" :transform="`translate(${231 + t * 101}, 190)`">
        <circle r="3" class="transfer-processing-dot" />
      </g>
      <g
        v-if="stage === 3"
        class="transfer-output-packet"
        :transform="`translate(${lerp(290, 468, t)}, 119)`"
      >
        <rect width="26" height="26" rx="3" class="transfer-target" />
        <text x="13" y="18" text-anchor="middle">07</text>
      </g>
      <g :opacity="stage === 3 ? t : 0" class="transfer-batch-grid">
        <rect
          v-for="n in 12"
          :key="n"
          :x="438 + ((n - 1) % 4) * 20"
          :y="157 + Math.floor((n - 1) / 4) * 10"
          width="15"
          height="6"
          rx="1"
        />
      </g>
      <text x="90" y="233" text-anchor="middle">
        {{ kind === "mmap" ? "CSR pages" : "Compressed unit" }}
      </text>
      <text x="290" y="233" text-anchor="middle">
        {{ kind === "mmap" ? "OS page cache" : "Decode all 12" }}
      </text>
      <text x="480" y="233" text-anchor="middle">Model batch</text>
    </svg>
    <div class="transfer-legend">
      <span><i /> requested cell</span
      ><span v-if="kind === 'chunk'"
        ><i class="extra" /> unrelated cells move, too</span
      ><span v-else>Read pages, then gather the row.</span>
    </div>
  </div>
</template>

<style scoped>
.transfer-request-label {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 6px;
  font: 13px var(--vp-font-family-mono);
  color: var(--cp-teal);
}
.transfer-request-label span {
  color: var(--vp-c-text-2);
  font: 12px var(--vp-font-family-base);
}
svg {
  display: block;
  width: 100%;
  height: auto;
  margin-top: 12px;
  overflow: visible;
}
text {
  fill: var(--vp-c-text-2);
  font: 16px var(--vp-font-family-mono);
}
.transfer-node {
  fill: var(--vp-c-bg);
  stroke: var(--vp-c-border);
}
.transfer-source-cell {
  fill: var(--vp-c-bg-soft);
  stroke: var(--vp-c-border);
}
.transfer-target,
.transfer-request-packet circle {
  fill: var(--cp-teal-soft);
  stroke: var(--cp-teal);
  stroke-width: 1.3;
}
.transfer-extra-byte {
  fill: var(--cp-accent);
  opacity: 0.7;
}
.transfer-decoded-cell {
  fill: var(--cp-accent-soft);
  stroke: var(--vp-c-border);
}
.transfer-page {
  fill: var(--cp-teal-soft);
  stroke: var(--cp-teal);
}
.transfer-request-path {
  fill: none;
  stroke: var(--cp-teal);
  stroke-width: 1.5;
  stroke-dasharray: 5 4;
}
.transfer-data-path {
  fill: none;
  stroke: var(--vp-c-text-3);
  stroke-width: 1.5;
}
.transfer-page-lines {
  fill: none;
  stroke: var(--cp-teal);
  stroke-width: 2;
  opacity: 0.7;
}
.transfer-processing-dot,
.transfer-batch-grid rect {
  fill: var(--cp-teal);
}
.transfer-legend {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}
.transfer-legend span {
  display: flex;
  align-items: center;
  gap: 6px;
}
.transfer-legend i {
  width: 8px;
  height: 8px;
  background: var(--cp-teal);
  border-radius: 2px;
}
.transfer-legend i.extra {
  background: var(--cp-accent);
}
</style>
