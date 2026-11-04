import { nextTick, onBeforeUnmount, onMounted, provide, ref } from "vue";

export const storyNavigationKey = Symbol("story-navigation");
const clamp = (value) => Math.min(1, Math.max(0, value));

export function useStoryNavigation(root, chapters) {
  const phase = ref("idle");
  const visible = ref(false);
  const activeChapter = ref(chapters[0].id);
  const activeScene = ref("");
  const progress = ref(chapters.map(() => 0));
  const pending = ref("");
  const announcement = ref("");
  provide(storyNavigationKey, phase);

  let media, observer, frame, timer, arrivalFrame, revealFrame;
  let initialFrame, initialSettleFrame, measurementFrame;
  let sections = [], start = 0, end = 0, offset = 64;
  let disposed = false;
  let interacted = false;

  function measure() {
    if (!root.value) return;
    offset = parseFloat(getComputedStyle(root.value).getPropertyValue("--vp-nav-height")) || 64;
    const top = (el) => el.getBoundingClientRect().top + window.scrollY - offset;
    start = top(root.value);
    end = start + root.value.offsetHeight;
    sections = chapters.map((chapter) => ({
      id: chapter.id,
      top: top(document.getElementById(chapter.id)),
      scenes: chapter.scenes.map((scene) => ({ id: scene.id, top: top(document.getElementById(scene.id)) })),
    }));
    if (frame !== undefined) cancelAnimationFrame(frame);
    update();
  }

  function update() {
    frame = undefined;
    const y = window.scrollY;
    // Include the reading space that native hash links leave below the header.
    const readingY = y + Math.min(96, window.innerHeight * 0.25);
    visible.value = y >= start - 2 && y < end - 2;
    let current = sections[0];
    for (const section of sections) if (readingY + 2 >= section.top) current = section;
    if (!current) return;
    activeChapter.value = current.id;
    activeScene.value = current.scenes.filter((scene) => readingY + 2 >= scene.top).at(-1)?.id || "";
    const next = sections.map((section, i) => clamp((y - section.top) / ((sections[i + 1]?.top ?? end) - section.top)));
    if (next.some((value, i) => value !== progress.value[i])) progress.value = next;
  }

  function scheduleMeasure() {
    if (disposed || measurementFrame !== undefined) return;
    measurementFrame = requestAnimationFrame(() => {
      measurementFrame = undefined;
      measure();
    });
  }

  function schedule() {
    if (frame === undefined) frame = requestAnimationFrame(update);
  }

  function onScroll() {
    if (phase.value === "leaving") cancel();
    else schedule();
  }

  function clearJump() {
    clearTimeout(timer);
    cancelAnimationFrame(arrivalFrame);
    cancelAnimationFrame(revealFrame);
  }

  function cancel() {
    clearJump();
    pending.value = "";
    phase.value = "idle";
    schedule();
  }

  function jump(id) {
    interacted = true;
    const target = document.getElementById(id);
    const home = id === "home" && target?.classList.contains("home-hero");
    if (!target || (!home && !root.value?.contains(target))) return;
    clearJump();
    pending.value = id;
    const land = () => {
      if (disposed) return;
      phase.value = media.matches ? "arriving" : "positioning";
      // Resolve geometry at landing time, after any responsive layout change.
      const y = home ? 0 : target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top: y, behavior: "instant" });
      target.querySelector(".chapter-text")?.classList.add("in-view");
      // Match VitePress's hash updates without adding one Back step per scene.
      history.replaceState(history.state, "", home ? location.pathname + location.search : `#${id}`);
      if (home) target.querySelector("h1")?.focus({ preventScroll: true });
      measure();
      announcement.value = home ? "Home page: CellPack overview" : `Now reading: ${target.querySelector("h2")?.textContent?.replace(/\s+/g, " ") || id}`;
      arrivalFrame = requestAnimationFrame(() => {
        revealFrame = requestAnimationFrame(() => {
          phase.value = media.matches ? "idle" : "arriving";
          pending.value = "";
          timer = setTimeout(() => { phase.value = "idle"; }, 240);
        });
      });
    };
    if (media.matches) land();
    else {
      phase.value = "leaving";
      timer = setTimeout(land, 140);
    }
  }

  function interrupt(event) {
    interacted = true;
    if (phase.value === "idle") return;
    if (event.type === "keydown" && !["Escape", "ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) return;
    if (event.type === "keydown" && event.key !== "Escape" && event.target.closest("a, button, input, textarea, select")) return;
    cancel();
  }

  function markInteraction() { interacted = true; }

  onMounted(() => {
    media = matchMedia("(prefers-reduced-motion: reduce)");
    observer = new ResizeObserver(scheduleMeasure);
    observer.observe(root.value);
    observer.observe(root.value.parentElement);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", scheduleMeasure);
    window.addEventListener("wheel", interrupt, { passive: true });
    window.addEventListener("touchstart", interrupt, { passive: true });
    window.addEventListener("keydown", interrupt);
    window.addEventListener("pointerdown", markInteraction, { passive: true });
    media.addEventListener("change", cancel);
    measure();
    const hash = location.hash;
    if (!hash || history.state?.scrollPosition > 0) return;
    // Static scene layouts and fonts settle after the router's first anchor pass.
    document.fonts.ready.then(async () => {
      await nextTick();
      if (disposed || interacted || !hash || hash !== location.hash) return;
      initialFrame = requestAnimationFrame(() => {
        initialSettleFrame = requestAnimationFrame(() => {
          if (disposed || interacted || hash !== location.hash) return;
          let id;
          try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
          const target = document.getElementById(id);
          if (!target || !root.value?.contains(target)) return;
          measure();
          window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - offset, behavior: "instant" });
          update();
        });
      });
    });
  });

  onBeforeUnmount(() => {
    disposed = true;
    clearJump();
    cancelAnimationFrame(frame);
    cancelAnimationFrame(measurementFrame);
    cancelAnimationFrame(initialFrame);
    cancelAnimationFrame(initialSettleFrame);
    observer?.disconnect();
    window.removeEventListener("scroll", onScroll);
    window.removeEventListener("resize", scheduleMeasure);
    window.removeEventListener("wheel", interrupt);
    window.removeEventListener("touchstart", interrupt);
    window.removeEventListener("keydown", interrupt);
    window.removeEventListener("pointerdown", markInteraction);
    media?.removeEventListener("change", cancel);
  });

  return { phase, visible, activeChapter, activeScene, progress, pending, announcement, jump };
}
