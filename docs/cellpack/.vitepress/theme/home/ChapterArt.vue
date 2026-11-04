<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import { alpha, hash, mix } from "./engine.js";

// Ambient art for a chapter card: a slice of an expression matrix whose
// nonzero cells pulse, while records stream out to the right.
const props = defineProps({ seed: { type: Number, default: 0 } });
const canvas = ref(null);
let ctx, frame, observer, media, themer, resizer, colors;
let dimensions;
let disposed = false;
let visible = false;
let start;

function readColors() {
  const css = getComputedStyle(canvas.value);
  const v = (name) => css.getPropertyValue(name).trim() || "#888";
  colors = { bg: v("--vp-c-bg"), divider: v("--vp-c-divider"), teal: v("--cp-teal"), accent: v("--cp-accent") };
}

function draw(now) {
  const el = canvas.value;
  if (!el) return;
  const { W, H, dpr } = dimensions;
  const t = media.matches ? 3 : (now - start) / 1000;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const rows = 14;
  const s = Math.min(H / rows, 26);
  const cols = Math.floor((W * 0.62) / s);
  const y0 = (H - rows * s) / 2;
  const zero = mix(colors.bg, colors.divider, 0.6);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const on = hash(r + props.seed * 31, c) < 0.11;
      // A diagonal wave sweeps across the matrix.
      const wave = Math.max(0, Math.sin(t * 1.4 - (c + r * 0.6) * 0.32));
      ctx.fillStyle = on ? alpha(colors.teal, 0.35 + 0.65 * wave) : alpha(zero, 0.9);
      const pad = s * 0.12;
      ctx.fillRect(c * s + pad, y0 + r * s + pad, s - pad * 2, s - pad * 2);
    }
    // Packed records leave each row toward the right edge.
    const x0 = cols * s + 24;
    const span = W - x0;
    for (let k = 0; k < 3; k++) {
      const speed = 0.08 + hash(r, k + props.seed) * 0.12;
      const u = (t * speed + hash(k, r * 3 + props.seed)) % 1;
      const w = s * (0.8 + hash(r, k) * 1.4);
      ctx.fillStyle = alpha(k === 0 && r % 4 === 1 ? colors.accent : colors.teal, 0.85 * Math.min(1, u * 6, (1 - u) * 6));
      ctx.fillRect(x0 + u * (span - w), y0 + r * s + s * 0.3, w, s * 0.4);
    }
  }
  frame = visible && !media.matches && !document.hidden ? requestAnimationFrame(draw) : undefined;
}

function measure() {
  if (!visible || disposed || document.hidden) return;
  const el = canvas.value;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  dimensions = { W: el.clientWidth, H: el.clientHeight, dpr };
  const width = Math.round(dimensions.W * dpr);
  const height = Math.round(dimensions.H * dpr);
  if (el.width !== width) el.width = width;
  if (el.height !== height) el.height = height;
  readColors();
  sync();
}

function sync() {
  if (frame !== undefined) cancelAnimationFrame(frame);
  frame = undefined;
  if (disposed || !visible || document.hidden) {
    if (canvas.value) canvas.value.width = canvas.value.height = 1;
    dimensions = undefined;
    return;
  }
  if (!dimensions) { measure(); return; }
  frame = requestAnimationFrame(draw);
}

onMounted(() => {
  ctx = canvas.value.getContext("2d");
  media = matchMedia("(prefers-reduced-motion: reduce)");
  media.addEventListener("change", sync);
  document.addEventListener("visibilitychange", sync);
  window.addEventListener("resize", measure);
  start = performance.now() - 3000;
  themer = new MutationObserver(() => {
    if (visible && !document.hidden) {
      readColors();
      sync();
    }
  });
  themer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  observer = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    sync();
  });
  observer.observe(canvas.value);
  resizer = new ResizeObserver(measure);
  resizer.observe(canvas.value);
  sync();
});
onBeforeUnmount(() => {
  disposed = true;
  if (canvas.value) canvas.value.width = canvas.value.height = 1;
  if (frame !== undefined) cancelAnimationFrame(frame);
  observer?.disconnect();
  themer?.disconnect();
  resizer?.disconnect();
  media?.removeEventListener("change", sync);
  document.removeEventListener("visibilitychange", sync);
  window.removeEventListener("resize", measure);
});
</script>

<template>
  <canvas ref="canvas" width="1" height="1" class="chapter-art" aria-hidden="true" />
</template>
