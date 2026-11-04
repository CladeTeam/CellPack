<script setup>
import { computed } from "vue";
const props = defineProps({
  stage: Number,
  progress: Number,
  ratio: Number,
  codec: String,
});
const nonzeros = [4, 29, 43, 65, 88, 105, 127, 150];
const cells = Array.from({ length: 160 }, (_, i) => ({
  i,
  x: 40 + (i % 16) * 30,
  y: 56 + Math.floor(i / 16) * 15,
  nonzero: nonzeros.includes(i),
  rank: nonzeros.indexOf(i),
}));
const t = computed(() => Math.min(1, props.progress * 1.6));
const lerp = (a, b, amount) => a + (b - a) * amount;
const csrOpacity = computed(() =>
  props.stage === 0 ? 0 : props.stage === 1 ? t.value : 1 - t.value,
);
const chunkOpacity = computed(() => (props.stage === 2 ? t.value : 0));
const amount = computed(() =>
  props.stage === 0
    ? 160
    : props.stage === 1
      ? lerp(160, 16, t.value)
      : lerp(16, 16 / props.ratio, t.value),
);
const title = computed(
  () =>
    [
      "Dense · store every entry",
      "CSR · keep nonzeros and positions",
      `${props.codec} · compressed chunks`,
    ][props.stage],
);
function cellGeometry(cell) {
  if (props.stage === 0)
    return { x: cell.x, y: cell.y, width: 25, height: 11, opacity: 1 };
  if (!cell.nonzero)
    return {
      x: cell.x,
      y: cell.y,
      width: 25 * (1 - t.value),
      height: 11 * (1 - t.value),
      opacity: props.stage === 1 ? 1 - t.value : 0,
    };
  const x = 116 + cell.rank * 43;
  if (props.stage === 1)
    return {
      x: lerp(cell.x, x, t.value),
      y: lerp(cell.y, 97, t.value),
      width: lerp(25, 32, t.value),
      height: lerp(11, 23, t.value),
      opacity: 1,
    };
  return {
    x: lerp(x, 190 + (cell.rank % 3) * 60, t.value),
    y: lerp(97, 130, t.value),
    width: lerp(32, 10, t.value),
    height: lerp(23, 10, t.value),
    opacity: 1 - t.value,
  };
}
</script>

<template>
  <div class="storage-scale-diagram" :data-stage="stage">
    <div class="scale-metric">
      <span>{{ title }}</span
      ><strong
        >{{
          amount < 10
            ? amount.toFixed(2)
            : amount.toFixed(1).replace(/\.0$/, "")
        }}
        <small>TB</small></strong
      >
    </div>
    <svg
      viewBox="0 0 560 238"
      role="img"
      aria-label="Dense entries pack into CSR arrays, then smaller compressed chunks"
    >
      <g v-if="stage === 0" class="scale-annotations">
        <text x="40" y="32">CELLS × FEATURES</text>
        <text x="40" y="225">Zeros take space, too.</text>
      </g>
      <rect
        v-for="cell in cells"
        :key="cell.i"
        v-bind="cellGeometry(cell)"
        :class="cell.nonzero ? 'scale-value' : 'scale-zero'"
        rx="1.5"
      />
      <g :opacity="csrOpacity" class="scale-csr-arrays">
        <text x="32" y="113">data</text>
        <text x="32" y="154">indices</text>
        <text x="32" y="195">indptr</text>
        <g v-for="n in 8" :key="n">
          <rect
            :x="116 + (n - 1) * 43"
            y="138"
            width="32"
            height="23"
            class="scale-index"
            rx="2"
          />
          <text :x="132 + (n - 1) * 43" y="154" text-anchor="middle">
            {{ nonzeros[n - 1] % 16 }}
          </text>
        </g>
        <g v-for="n in 11" :key="n">
          <rect
            :x="116 + (n - 1) * 30"
            y="179"
            width="23"
            height="23"
            class="scale-pointer"
            rx="2"
          />
          <text :x="127.5 + (n - 1) * 30" y="195" text-anchor="middle">
            {{ [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8][n - 1] }}
          </text>
        </g>
      </g>
      <g :opacity="chunkOpacity" class="scale-chunks">
        <g
          v-for="n in 3"
          :key="n"
          :transform="`translate(${170 + (n - 1) * 74}, ${116 + (1 - t) * 35}) scale(${0.6 + t * 0.4})`"
        >
          <path d="M0 0 17-10 65-10 48 0Z" class="scale-chunk-top" />
          <path d="M48 0 65-10 65 42 48 52Z" class="scale-chunk-side" />
          <rect width="48" height="52" rx="2" class="scale-chunk-front" />
          <rect
            v-for="row in 5"
            :key="row"
            x="8"
            :y="7 + (row - 1) * 8"
            :width="19 + ((row + n) % 3) * 6"
            height="4"
            class="scale-chunk-stripe"
          />
        </g>
        <text x="280" y="217" text-anchor="middle">
          Compressed units · fewer bytes to store
        </text>
      </g>
    </svg>
    <div class="storage-footprint">
      <span>Footprint</span>
      <div><i :style="{ transform: `scaleX(${amount / 160})` }" /></div>
      <span>{{ (160 / amount).toFixed(1) }}× smaller</span>
    </div>
  </div>
</template>

<style scoped>
.storage-scale-diagram {
  width: 100%;
}
.scale-metric {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 2px 4px 8px;
}
.scale-metric > span {
  max-width: 60%;
  color: var(--vp-c-text-2);
  font-size: 14px;
  line-height: 1.5;
}
.scale-metric strong {
  font: 500 38px var(--cp-font-heading);
  white-space: nowrap;
  letter-spacing: -1.5px;
}
.scale-metric strong small {
  font-size: 17px;
  letter-spacing: 0;
  color: var(--vp-c-text-2);
}
svg {
  display: block;
  width: 100%;
  height: auto;
}
text {
  fill: var(--vp-c-text-2);
  font: 16px var(--vp-font-family-mono);
}
.scale-annotations text {
  font-size: 16px;
}
.scale-value {
  fill: var(--cp-teal);
}
.scale-zero {
  fill: var(--vp-c-divider);
}
.scale-index {
  fill: var(--cp-accent-soft);
  stroke: var(--cp-accent);
  stroke-width: 0.7;
}
.scale-pointer {
  fill: var(--vp-c-bg-soft);
  stroke: var(--vp-c-border);
  stroke-width: 0.7;
}
.scale-chunk-front {
  fill: var(--cp-teal);
}
.scale-chunk-side {
  fill: color-mix(in srgb, var(--cp-teal) 75%, var(--vp-c-text-1));
}
.scale-chunk-top {
  fill: var(--cp-teal-soft);
  stroke: var(--cp-teal);
  stroke-width: 0.6;
}
.scale-chunk-stripe {
  fill: var(--vp-c-bg);
  opacity: 0.65;
}
.storage-footprint {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  color: var(--vp-c-text-2);
  font-size: 12px;
  margin-top: 10px;
}
.storage-footprint > div {
  height: 10px;
  background: var(--vp-c-bg-soft);
}
.storage-footprint i {
  display: block;
  width: 100%;
  height: 100%;
  background: var(--cp-teal);
  transform-origin: left;
}
@media (max-width: 540px) {
  .scale-metric strong {
    font-size: 31px;
  }
  .scale-metric > span {
    font-size: 13px;
  }
  .storage-footprint {
    font-size: 11px;
    gap: 7px;
  }
}
</style>
