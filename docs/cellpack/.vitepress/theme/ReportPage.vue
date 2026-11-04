<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { withBase } from "vitepress";
import SiteFooter from "./components/SiteFooter.vue";
import "./report.css";

const container = ref(null);
const pages = ref(null);
const loading = ref(true);
const error = ref(false);
const page = ref(1);
const pageCount = ref(0);
const query = ref("");
const matchStatus = ref("");
const zoom = ref("page-width");
const scale = ref(1);
const minScale = 0.25;
const maxScale = 4;
const percent = computed(() => Math.round(scale.value * 100));
const numericZooms = ['0.25', '0.5', '0.75', '1', '1.25', '1.5', '2', '3', '4'];
const customZoom = computed(() => Number(zoom.value) > 0 && !numericZooms.includes(zoom.value));
const ready = computed(() => !loading.value && !error.value);
const pdfUrl = withBase("/reports/cellpack-technical-report.pdf");
let viewer;
let eventBus;
let linkService;
let loadingTask;
let lifetime;
let resizeObserver;
let disposed = false;
let searchTimer;
let viewerStyles;
let resizeFrame;
let readerWidth = 0;

function setPage(value) {
  if (!ready.value) return;
  const target = Math.max(1, Math.min(pageCount.value, Number(value) || 1));
  viewer.currentPageNumber = target;
  page.value = target;
}

function setZoom() {
  if (ready.value) viewer.currentScaleValue = zoom.value;
}

function zoomTo(value, origin) {
  if (!ready.value) return;
  const target = Math.max(minScale, Math.min(maxScale, Math.round(value * 100) / 100));
  viewer.updateScale({ scaleFactor: target / viewer.currentScale, origin, drawingDelay: 120 });
}

function zoomBy(direction) {
  if (!ready.value) return;
  zoomTo(viewer.currentScale * (direction > 0 ? 1.2 : 1 / 1.2));
}

function readerKeydown(event) {
  if (event.target.closest('input, select, button') || event.ctrlKey || event.metaKey || event.altKey || !ready.value) return;
  if (['+', '=', '-', '0'].includes(event.key)) {
    event.preventDefault();
    if (event.key === '0') zoomTo(1);
    else zoomBy(event.key === '-' ? -1 : 1);
  }
}

function find(again = false, previous = false) {
  if (!ready.value) return;
  eventBus.dispatch("find", {
    source: container.value,
    type: again ? "again" : "",
    query: query.value,
    caseSensitive: false,
    entireWord: false,
    highlightAll: true,
    findPrevious: previous,
    matchDiacritics: false,
  });
}

function scheduleFind() {
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => find(), 200);
}

onMounted(async () => {
  lifetime = new AbortController();
  try {
    // The viewer reads pdfjsLib when its module is evaluated.
    const [pdfjs, styles] = await Promise.all([
      import("pdfjs-dist/build/pdf.mjs"),
      import("pdfjs-dist/web/pdf_viewer.css?inline"),
    ]);
    if (disposed) return;
    // Keep the PDF viewer's substantial stylesheet out of documentation loads.
    viewerStyles = document.createElement('style');
    viewerStyles.dataset.cellpackPdfViewer = '';
    viewerStyles.textContent = styles.default;
    document.head.append(viewerStyles);
    const [{ PDFViewer, PDFLinkService, PDFFindController, EventBus }, worker] =
      await Promise.all([
        import("pdfjs-dist/web/pdf_viewer.mjs"),
        import("pdfjs-dist/build/pdf.worker.min.mjs?url"),
      ]);
    if (disposed) return;
    pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
    eventBus = new EventBus();
    linkService = new PDFLinkService({
      eventBus,
      externalLinkTarget: 2,
      externalLinkRel: "noopener noreferrer",
    });
    const findController = new PDFFindController({ eventBus, linkService });
    viewer = new PDFViewer({
      container: container.value,
      viewer: pages.value,
      eventBus,
      linkService,
      findController,
      abortSignal: lifetime.signal,
      annotationEditorMode: -1,
      enableAutoLinking: false,
      maxCanvasPixels: 8_000_000,
      l10n: {
        getLanguage: () => "en-US",
        getDirection: () => "ltr",
        get: async (_ids, _args, fallback) => fallback || "",
        translate: async () => {},
        pause: () => {},
        resume: () => {},
        destroy: async () => {},
      },
    });
    linkService.setViewer(viewer);
    const listenerOptions = { signal: lifetime.signal };
    eventBus.on('scalechanging', event => {
      scale.value = event.scale;
      zoom.value = event.presetValue || String(event.scale);
    }, listenerOptions);
    eventBus.on(
      "pagesinit",
      () => {
        viewer.currentScaleValue = zoom.value;
        loading.value = false;
      },
      listenerOptions,
    );
    eventBus.on(
      "pagechanging",
      (event) => {
        page.value = event.pageNumber;
      },
      listenerOptions,
    );
    eventBus.on(
      "updatefindmatchescount",
      ({ matchesCount }) => {
        matchStatus.value = `${matchesCount.current} / ${matchesCount.total}`;
      },
      listenerOptions,
    );
    eventBus.on(
      "updatefindcontrolstate",
      ({ state }) => {
        if (!query.value) matchStatus.value = "";
        else if (state === 1) matchStatus.value = "No matches";
        else if (state === 3) matchStatus.value = "Searching…";
      },
      listenerOptions,
    );
    eventBus.on(
      "pagerendered",
      ({ error: renderError }) => {
        if (renderError && !disposed) error.value = true;
      },
      listenerOptions,
    );
    loadingTask = pdfjs.getDocument({
      url: pdfUrl,
      cMapUrl: withBase("/pdfjs/cmaps/"),
      cMapPacked: true,
      standardFontDataUrl: withBase("/pdfjs/standard_fonts/"),
      wasmUrl: withBase("/pdfjs/wasm/"),
      isEvalSupported: false,
    });
    const pdfDocument = await loadingTask.promise;
    if (disposed) return;
    pageCount.value = pdfDocument.numPages;
    viewer.setDocument(pdfDocument);
    linkService.setDocument(pdfDocument);
    container.value.addEventListener('wheel', event => {
      if (!(event.ctrlKey || event.metaKey) || !ready.value) return;
      event.preventDefault();
      zoomTo(viewer.currentScale * Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.002), [event.clientX, event.clientY]);
    }, { passive: false, signal: lifetime.signal });
    resizeObserver = new ResizeObserver(([entry]) => {
      const width = Math.round(entry.contentRect.width);
      if (width === readerWidth) return;
      readerWidth = width;
      if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = undefined;
        if (ready.value && zoom.value === "page-width") viewer.currentScaleValue = "page-width";
      });
    });
    resizeObserver.observe(container.value);
  } catch (cause) {
    if (disposed) return;
    console.error("Unable to load the technical report", cause);
    error.value = true;
    loading.value = false;
  }
});

onBeforeUnmount(() => {
  disposed = true;
  window.clearTimeout(searchTimer);
  resizeObserver?.disconnect();
  if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame);
  viewerStyles?.remove();
  viewer?.setDocument(null);
  linkService?.setDocument(null);
  lifetime?.abort();
  void loadingTask?.destroy();
});
</script>

<template>
  <div class="cp-report">
    <header class="cp-report-heading">
      <div>
        <p class="cp-eyebrow">THE RESEARCH BEHIND CELLPACK</p>
        <h1>Technical report<span>.</span></h1>
        <p>CellPack: Efficient Data Loading for Billion-Cell Training</p>
      </div>
      <div class="cp-report-links">
        <a :href="pdfUrl" target="_blank" rel="noopener"
          >Open PDF <span aria-hidden="true">↗</span></a
        ><a :href="pdfUrl" download="cellpack-technical-report.pdf"
          >Download <span aria-hidden="true">↓</span></a
        >
      </div>
    </header>
    <section class="cp-reader" aria-label="Technical report PDF reader" @keydown="readerKeydown">
      <div class="cp-reader-toolbar">
        <div class="cp-page-controls">
          <button
            type="button"
            aria-label="Previous page"
            :disabled="!ready || page <= 1"
            @click="setPage(page - 1)"
          >
            ←
          </button>
          <label
            ><span class="cp-visually-hidden">Page number</span
            ><input
              aria-label="Page number"
              type="number"
              min="1"
              :max="pageCount || 1"
              :value="page"
              :disabled="!ready"
              @change="setPage($event.target.value)"
          /></label>
          <span class="cp-page-total">/ {{ pageCount || "—" }}</span>
          <button
            type="button"
            aria-label="Next page"
            :disabled="!ready || page >= pageCount"
            @click="setPage(page + 1)"
          >
            →
          </button>
        </div>
        <div class="cp-search-controls">
          <label
            ><span class="cp-visually-hidden">Search report</span
            ><svg viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="8.5" cy="8.5" r="5.5" />
              <path d="m13 13 4 4" /></svg
            ><input
              v-model="query"
              type="search"
              placeholder="Find in report"
              aria-label="Search report"
              :disabled="!ready"
              @input="scheduleFind"
              @keydown.enter.prevent="find(true, $event.shiftKey)" /></label
          ><button
            type="button"
            aria-label="Previous search match"
            :disabled="!ready || !query"
            @click="find(true, true)"
          >
            ↑</button
          ><button
            type="button"
            aria-label="Next search match"
            :disabled="!ready || !query"
            @click="find(true)"
          >
            ↓</button
          ><span class="cp-match-status" role="status">{{ matchStatus }}</span>
        </div>
        <div class="cp-zoom-controls" role="group" aria-label="Report zoom">
          <button type="button" aria-label="Zoom out" title="Zoom out (−)" :disabled="!ready || scale <= minScale" @click="zoomBy(-1)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12" /></svg></button>
          <button type="button" class="cp-zoom-percent" aria-label="Reset zoom to 100 percent" title="Reset to 100% (0)" :disabled="!ready" @click="zoomTo(1)">{{ ready ? percent + '%' : '—' }}</button>
          <button type="button" aria-label="Zoom in" title="Zoom in (+)" :disabled="!ready || scale >= maxScale" @click="zoomBy(1)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12M10 4v12" /></svg></button>
        <label class="cp-zoom-control"
          ><span class="cp-visually-hidden">Zoom</span
          ><select
            v-model="zoom"
            aria-label="Zoom"
            :disabled="!ready"
            @change="setZoom"
          >
            <option value="page-width">Fit width</option>
            <option value="page-fit">Fit page</option>
            <option v-if="customZoom" :value="zoom">Custom</option>
            <option v-for="value in numericZooms" :key="value" :value="value">{{ Math.round(Number(value) * 100) }}%</option>
          </select></label
        >
        </div>
      </div>
      <div class="cp-reader-stage">
        <div v-if="loading && !error" class="cp-reader-message" role="status">
          <span class="cp-reader-spinner" aria-hidden="true" />
          <p>Opening the technical report…</p>
        </div>
        <div v-if="error" class="cp-reader-message" role="alert">
          <p>The embedded reader is unavailable.</p>
          <a :href="pdfUrl" target="_blank" rel="noopener"
            >Open the PDF directly <span aria-hidden="true">↗</span></a
          >
        </div>
        <div
          ref="container"
          class="cp-pdf-container"
          :class="{ 'reader-hidden': loading || error }"
          tabindex="0"
          aria-label="Report pages"
        >
          <div ref="pages" class="pdfViewer" />
        </div>
      </div>
    </section>
    <div class="cp-report-caption">
      <p>Original typeset PDF · October 2026</p>
      <a :href="withBase('/overview.html')"
        >Looking for usage examples? Explore the docs
        <span aria-hidden="true">→</span></a
      >
    </div>
    <SiteFooter />
  </div>
</template>
