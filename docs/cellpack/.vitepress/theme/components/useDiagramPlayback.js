import { computed, onBeforeUnmount, onMounted, ref, toValue, watch } from "vue";

export function useDiagramPlayback(
  duration,
  { reducedAtEnd = false, threshold = 0.25 } = {},
) {
  const root = ref(null);
  const elapsed = ref(0);
  const playing = ref(false);
  const visible = ref(false);
  const documentVisible = ref(true);
  const reduced = ref(false);
  const total = computed(() => Math.max(1, toValue(duration)));
  const ended = computed(() => elapsed.value >= total.value);
  const state = computed(() =>
    ended.value
      ? "ended"
      : playing.value
        ? "playing"
        : elapsed.value
          ? "paused"
          : "ready",
  );
  let observer, media, frame, previous;

  function stopFrame() {
    if (frame !== undefined) cancelAnimationFrame(frame);
    frame = undefined;
    previous = undefined;
  }
  function tick(time) {
    if (previous !== undefined)
      elapsed.value = Math.min(total.value, elapsed.value + time - previous);
    previous = time;
    if (ended.value) {
      playing.value = false;
      frame = undefined;
    } else frame = requestAnimationFrame(tick);
  }
  function sync() {
    stopFrame();
    if (playing.value && visible.value && documentVisible.value && !ended.value)
      frame = requestAnimationFrame(tick);
  }
  function restart() {
    elapsed.value = 0;
    playing.value = true;
    sync();
  }
  function toggle() {
    if (ended.value) restart();
    else playing.value = !playing.value;
  }
  function visibility() {
    documentVisible.value = !document.hidden;
  }
  function preference() {
    const wasReduced = reduced.value;
    reduced.value = media.matches;
    if (wasReduced && !media.matches) elapsed.value = 0;
    playing.value = !media.matches;
    if (media.matches && reducedAtEnd) elapsed.value = total.value;
  }

  watch([playing, visible, documentVisible], sync);
  onMounted(() => {
    media = matchMedia("(prefers-reduced-motion: reduce)");
    preference();
    visibility();
    media.addEventListener("change", preference);
    document.addEventListener("visibilitychange", visibility);
    observer = new IntersectionObserver(
      ([entry]) => {
        visible.value = entry.isIntersecting;
      },
      { threshold },
    );
    observer.observe(root.value);
  });
  onBeforeUnmount(() => {
    stopFrame();
    observer?.disconnect();
    media?.removeEventListener("change", preference);
    document.removeEventListener("visibilitychange", visibility);
  });
  return {
    root,
    elapsed,
    playing,
    visible,
    reduced,
    ended,
    state,
    toggle,
    restart,
  };
}
