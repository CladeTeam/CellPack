<script setup>
import { computed, inject, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { storyNavigationKey } from "./useStoryNavigation.js";

// A pinned stage whose canvas animation is driven by scroll position.
// `scene.draw(ctx, env)` renders one frame for progress `env.p`; text beats
// cross-fade as the progress passes their `at` value.
const props = defineProps({
  id: { type: String, required: true },
  scene: { type: Object, required: true },
  beats: { type: Array, required: true },
  length: { type: Number, default: 400 },
  source: String,
  // "side": text beside the canvas; "top": text above a wide canvas.
  layout: { type: String, default: "side" },
});

const section = ref(null);
const stage = ref(null);
const canvas = ref(null);
const copy = ref(null);
const rail = ref(null);
const sourceEl = ref(null);
const beat = ref(0);
const reduced = ref(false);
const short = ref(false);
const flow = computed(() => reduced.value || short.value);
const env = shallowRef(null);
const navigation = inject(storyNavigationKey, ref("idle"));
const style = computed(() => ({ "--scene-length": `${props.length}vh` }));

const TOKENS = {
  bg: "--vp-c-bg",
  bgAlt: "--vp-c-bg-alt",
  surface: "--cp-surface",
  text: "--vp-c-text-1",
  text2: "--vp-c-text-2",
  text3: "--vp-c-text-3",
  divider: "--vp-c-divider",
  border: "--vp-c-border",
  teal: "--cp-teal",
  tealSoft: "--cp-teal-soft",
  accent: "--cp-accent",
  accentSoft: "--cp-accent-soft",
  danger: "--story-danger",
};

let ctx, frame, last, observer, resizer, themer, media, viewport, measurement;
let disposed = false;
let visible = false;
let target = 0;
let progress = 0;
let clock = 0;
let state = null;
let geometry;

function colors() {
  const css = getComputedStyle(section.value);
  return Object.fromEntries(Object.entries(TOKENS).map(([k, v]) => [k, css.getPropertyValue(v).trim() || "#888"]));
}

function measure() {
  const el = flow.value ? canvas.value : stage.value;
  if (disposed || !visible || document.hidden || !el || !canvas.value) return;
  const W = el.clientWidth;
  const H = el.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  let box;
  let wide;
  if (flow.value) {
    // Static layout: the canvas sits in its own grid cell beside the text.
    box = { x: 12, y: 12, w: W - 24, h: H - 24 };
  } else {
    const s = el.getBoundingClientRect();
    const c = copy.value.getBoundingClientRect();
    // Match the copy's CSS breakpoint even with a reserved navigation gutter.
    const beside = matchMedia("(min-width: 960px)").matches;
    const side = beside ? Math.max(40, (W - 1240) / 2) : 16;
    const bottom = sourceEl.value ? sourceEl.value.getBoundingClientRect().top - s.top - 20 : H - 24;
    if (beside && props.layout === "side") {
      box = { x: c.right - s.left + 40, y: 32, w: 0, h: bottom - 32 };
      box.w = W - side - box.x;
      // Large screens: cap the drawing area and centre it vertically, so a
      // scene never hugs the top with empty space below.
      const maxH = Math.min(box.w * 0.95, 820);
      if (box.h > maxH) {
        box.y += (box.h - maxH) / 2;
        box.h = maxH;
      }
    } else {
      box = { x: side, y: c.bottom - s.top + 20, w: W - side * 2, h: 0 };
      box.h = bottom - box.y;
      if (beside && box.h > 720) {
        box.y += (box.h - 720) / 2;
        box.h = 720;
      }
      const maxW = box.h * 2.4;
      if (beside && box.w > maxW) {
        box.x += (box.w - maxW) / 2;
        box.w = maxW;
      }
    }
  }
  const rect = section.value.getBoundingClientRect();
  geometry = {
    start: rect.top + window.scrollY,
    span: rect.height - stage.value.clientHeight,
    offset: parseFloat(getComputedStyle(stage.value).top) || 0,
  };
  // Keep only nearby scenes' backing stores resident.
  ctx ??= canvas.value.getContext("2d");
  const width = Math.round(W * dpr);
  const height = Math.round(H * dpr);
  if (canvas.value.width !== width) canvas.value.width = width;
  if (canvas.value.height !== height) canvas.value.height = height;
  wide = box.w >= 680;
  env.value = { W, H, dpr, box, wide, c: colors() };
  state = props.scene.setup ? props.scene.setup(env.value) : null;
  readTarget();
  progress = target;
  draw(0);
}

function readTarget() {
  if (flow.value) { target = 1; return; }
  if (!geometry) return;
  const { start, span, offset } = geometry;
  target = span > 0 ? Math.min(1, Math.max(0, (window.scrollY + offset - start) / span)) : 1;
}

function draw(dt) {
  if (!env.value || !ctx) return;
  const e = env.value;
  ctx.setTransform(e.dpr, 0, 0, e.dpr, 0, 0);
  ctx.clearRect(0, 0, e.W, e.H);
  props.scene.draw(ctx, { ...e, p: progress, t: clock, dt, state });
  let index = 0;
  props.beats.forEach((b, i) => {
    if (progress >= b.at) index = i;
  });
  if (index !== beat.value) beat.value = index;
  if (rail.value) rail.value.style.transform = `scaleY(${progress})`;
}

function tick(now) {
  const dt = last === undefined ? 0 : Math.min(0.05, (now - last) / 1000);
  last = now;
  clock += dt;
  progress += (target - progress) * (1 - Math.exp(-dt * 7));
  if (Math.abs(target - progress) < 0.0004) progress = target;
  draw(dt);
  // Scroll-only illustrations need no frames once their easing settles.
  frame = props.scene.continuous === false && progress === target
    ? undefined
    : requestAnimationFrame(tick);
}

function sync() {
  const active = !disposed && visible && !document.hidden;
  if (active && !env.value) measure();
  const run = active && !flow.value && ["idle", "arriving"].includes(navigation.value);
  if (run && frame === undefined) {
    last = undefined;
    readTarget();
    progress = target;
    draw(0);
    frame = requestAnimationFrame(tick);
  } else if (!run && frame !== undefined) {
    cancelAnimationFrame(frame);
    frame = undefined;
  }
}

watch(navigation, (value) => {
  if (!visible || !env.value) return;
  if (value === "positioning" || value === "arriving" || value === "idle") {
    readTarget();
    progress = target;
    draw(0);
  }
  sync();
}, { flush: "post" });

function preference() {
  reduced.value = media.matches;
  short.value = viewport.matches;
  scheduleMeasure();
  sync();
}

function scheduleMeasure() {
  if (disposed || !visible || document.hidden || measurement !== undefined) return;
  measurement = requestAnimationFrame(() => {
    measurement = undefined;
    measure();
    sync();
  });
}

function onScroll() {
  readTarget();
  if (frame === undefined && !flow.value && visible && !document.hidden && navigation.value === "idle" && progress !== target) {
    last = undefined;
    frame = requestAnimationFrame(tick);
  }
}

function release() {
  if (frame !== undefined) cancelAnimationFrame(frame);
  frame = undefined;
  if (measurement !== undefined) cancelAnimationFrame(measurement);
  measurement = undefined;
  canvas.value.width = canvas.value.height = 1;
  env.value = null;
  state = null;
  geometry = undefined;
}

function visibility() {
  if (document.hidden) release();
  else sync();
}

onMounted(() => {
  media = matchMedia("(prefers-reduced-motion: reduce)");
  viewport = matchMedia("(max-height: 699px), (max-width: 540px) and (max-height: 799px)");
  media.addEventListener("change", preference);
  viewport.addEventListener("change", preference);
  reduced.value = media.matches;
  short.value = viewport.matches;
  document.fonts?.ready.then(() => {
    if (!disposed) scheduleMeasure();
  });
  resizer = new ResizeObserver(scheduleMeasure);
  resizer.observe(stage.value);
  resizer.observe(copy.value);
  if (sourceEl.value) resizer.observe(sourceEl.value);
  window.addEventListener("resize", scheduleMeasure);
  themer = new MutationObserver(() => {
    if (env.value) {
      env.value = { ...env.value, c: colors() };
      state = props.scene.setup ? props.scene.setup(env.value) : null;
      draw(0);
    }
  });
  themer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  observer = new IntersectionObserver(
    ([entry]) => {
      visible = entry.isIntersecting;
      if (visible) {
        window.addEventListener("scroll", onScroll, { passive: true });
        sync();
      } else {
        window.removeEventListener("scroll", onScroll);
        release();
      }
    },
    { rootMargin: "20% 0px" },
  );
  observer.observe(section.value);
  document.addEventListener("visibilitychange", visibility);
});

onBeforeUnmount(() => {
  disposed = true;
  if (frame !== undefined) cancelAnimationFrame(frame);
  if (measurement !== undefined) cancelAnimationFrame(measurement);
  observer?.disconnect();
  resizer?.disconnect();
  themer?.disconnect();
  media?.removeEventListener("change", preference);
  viewport?.removeEventListener("change", preference);
  window.removeEventListener("resize", scheduleMeasure);
  document.removeEventListener("visibilitychange", visibility);
  window.removeEventListener("scroll", onScroll);
  state = null;
  env.value = null;
  if (canvas.value) canvas.value.width = canvas.value.height = 1;
});
</script>

<template>
  <section
    :id="id"
    ref="section"
    class="scene"
    :class="{ 'is-static': flow, 'layout-top': layout === 'top' }"
    :style="style"
  >
    <div ref="stage" class="scene-stage">
      <canvas ref="canvas" width="1" height="1" class="scene-canvas" aria-hidden="true" />
      <div ref="copy" class="scene-copy">
        <article
          v-for="(b, i) in beats"
          :key="i"
          class="scene-beat"
          :class="{ active: i === beat, past: i < beat }"
          :aria-hidden="!flow && i !== beat"
        >
          <p v-if="b.kicker" class="scene-kicker">{{ b.kicker }}</p>
          <component :is="i === 0 ? 'h2' : 'h3'" class="scene-title" v-html="b.title" />
          <p class="scene-body" v-html="b.body" />
        </article>
      </div>
      <p v-if="source" ref="sourceEl" class="scene-source" v-html="source" />
      <div class="scene-rail" aria-hidden="true"><i ref="rail" /></div>
    </div>
  </section>
</template>
