<script setup>
import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { withBase } from "vitepress";

const props = defineProps({
  paused: Boolean,
  autoplayDelay: { type: Number, default: 1500 },
});
const stage = ref(null);
const host = ref(null);
const frameSource = ref("");
const phase = ref("waiting");
const reduced = ref(true);
const ready = ref(false);
const manualMotion = ref(false);
const showAnimation = computed(
  () => ready.value && (!reduced.value || manualMotion.value),
);
const label = computed(() =>
  phase.value === "playing"
    ? "Pause logo animation"
    : phase.value === "paused"
      ? "Resume logo animation"
      : phase.value === "ended"
        ? "Replay logo animation"
        : "Play logo animation",
);
const poster = withBase("/cellpack-logo-poster.svg");
const animation = withBase("/cellpack-logo-animation.svg");
let svg;
let observer;
let preference;
let loadTimer;
let visible = false;
let disposed = false;
let wantsPlayback = false;
let autoplayPending = true;
let autoplayTimer;
let autoplayStartedAt = 0;
let autoplayRemaining = props.autoplayDelay;
let finishTimer;
let duration = 0;

function clearAutoplay() {
  if (autoplayTimer !== undefined) {
    window.clearTimeout(autoplayTimer);
    autoplayRemaining = Math.max(
      0,
      autoplayRemaining - (performance.now() - autoplayStartedAt),
    );
    autoplayTimer = undefined;
  }
}

function pause() {
  window.clearTimeout(finishTimer);
  clearAutoplay();
  svg?.pauseAnimations();
  if (phase.value === "playing") phase.value = "paused";
}

function play() {
  if (!svg || phase.value === "playing") return;
  window.clearTimeout(finishTimer);
  svg.unpauseAnimations();
  phase.value = "playing";
  finishTimer = window.setTimeout(
    () => {
      if (disposed || !svg) return;
      svg.setCurrentTime(duration);
      svg.pauseAnimations();
      wantsPlayback = false;
      phase.value = "ended";
    },
    Math.max(0, duration - svg.getCurrentTime()) * 1000 + 40,
  );
}

function failed() {
  if (disposed) return;
  window.clearTimeout(loadTimer);
  window.clearTimeout(finishTimer);
  clearAutoplay();
  svg?.pauseAnimations();
  svg = undefined;
  ready.value = false;
  frameSource.value = "";
  phase.value = "error";
}

function loaded() {
  if (disposed || !frameSource.value) return;
  window.clearTimeout(loadTimer);
  try {
    // The same-origin, scriptless SVG document keeps animation nodes outside
    // the page's style and layout tree while exposing the native timeline API.
    const root = host.value?.contentDocument?.documentElement;
    if (root?.namespaceURI !== "http://www.w3.org/2000/svg" || typeof root.pauseAnimations !== "function") {
      throw new Error("Logo animation unavailable");
    }
    root.pauseAnimations();
    root.setCurrentTime(0);
    duration = Number(root.getAttribute("data-duration"));
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("Logo animation unavailable");
    svg = root;
    ready.value = true;
    phase.value = "ready";
    sync();
  } catch {
    failed();
  }
}

function load() {
  if (disposed || phase.value === "loading" || svg) return;
  phase.value = "loading";
  frameSource.value = animation;
  // CSP or a failed navigation can leave an embedded document without a load
  // event; retain the poster and let an explicit click retry in that case.
  loadTimer = window.setTimeout(failed, 15000);
}

function sync() {
  if (disposed) return;
  if (
    (reduced.value && !manualMotion.value) ||
    props.paused ||
    !visible ||
    document.hidden
  ) {
    pause();
    return;
  }
  if (!svg) {
    if (phase.value !== "error") void load();
    return;
  }
  if (autoplayPending) {
    if (autoplayTimer === undefined) {
      autoplayStartedAt = performance.now();
      autoplayTimer = window.setTimeout(() => {
        autoplayTimer = undefined;
        autoplayPending = false;
        wantsPlayback = true;
        play();
      }, autoplayRemaining);
    }
  } else if (wantsPlayback && phase.value !== "ended") {
    play();
  }
}

function updatePreference() {
  reduced.value = preference.matches;
  manualMotion.value = false;
  sync();
}

function toggle() {
  if (props.paused) return;
  clearAutoplay();
  autoplayPending = false;
  if (phase.value === "playing") {
    wantsPlayback = false;
    pause();
    return;
  }
  manualMotion.value = true;
  wantsPlayback = true;
  if (phase.value === "ended" && svg) {
    svg.setCurrentTime(0);
    phase.value = "ready";
  }
  if (phase.value === "error") phase.value = "waiting";
  sync();
}

watch(() => props.paused, sync);
onMounted(() => {
  preference = window.matchMedia("(prefers-reduced-motion: reduce)");
  reduced.value = preference.matches;
  preference.addEventListener("change", updatePreference);
  document.addEventListener("visibilitychange", sync);
  observer = new IntersectionObserver(
    ([entry]) => {
      visible = entry.isIntersecting;
      sync();
    },
    { threshold: 0.15 },
  );
  observer.observe(stage.value);
});

onBeforeUnmount(() => {
  disposed = true;
  pause();
  window.clearTimeout(loadTimer);
  observer?.disconnect();
  preference?.removeEventListener("change", updatePreference);
  document.removeEventListener("visibilitychange", sync);
  svg = undefined;
});
</script>

<template>
  <div ref="stage" class="cellpack-logo-stage">
    <img
      v-show="!showAnimation"
      class="logo-poster"
      :src="poster"
      width="512"
      height="600"
      alt=""
      aria-hidden="true"
    />
    <iframe
      v-if="frameSource"
      ref="host"
      v-show="showAnimation"
      class="logo-animation"
      :src="frameSource"
      sandbox="allow-same-origin"
      tabindex="-1"
      aria-hidden="true"
      title="CellPack logo animation"
      scrolling="no"
      @load="loaded"
      @error="failed"
    />
    <button
      type="button"
      class="cellpack-logo-player"
      :data-phase="phase"
      :aria-label="label"
      :aria-pressed="phase === 'playing'"
      :title="label"
      :disabled="props.paused"
      @click="toggle"
    />
  </div>
</template>

<style scoped>
.cellpack-logo-stage {
  position: relative;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: 12px;
}
.cellpack-logo-player {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  border-radius: 12px;
  background: transparent;
}
.cellpack-logo-player:disabled {
  cursor: default;
}
.cellpack-logo-player:focus-visible {
  outline: 2px solid var(--cp-teal);
  outline-offset: -3px;
}
.logo-poster,
.logo-animation {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border: 0;
  pointer-events: none;
}
.logo-poster {
  object-fit: contain;
}
.logo-animation {
  /* Match the standalone SVG document's normal scheme so the browser keeps
     its canvas transparent when the surrounding page uses a dark scheme. */
  color-scheme: normal;
  background: transparent;
}
</style>
