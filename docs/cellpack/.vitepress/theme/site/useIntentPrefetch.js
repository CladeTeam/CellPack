import { onBeforeUnmount, onMounted, watch } from 'vue';
import { useData, useRoute } from 'vitepress';
import { pathToFile } from 'vitepress/dist/client/app/utils.js';

// Fetch the next page when a visitor shows intent, without downloading every sidebar link.
export function useIntentPrefetch() {
  if (!import.meta.env.PROD) return;

  const { site } = useData();
  const route = useRoute();
  const fetched = new Set();
  const requests = new Map();
  const timers = new Map();
  let mounted = false;
  let supported = false;

  function allowsPrefetch() {
    const connection = navigator.connection;
    return supported && !document.hidden && !connection?.saveData && !/2g/.test(connection?.effectiveType || '');
  }

  function anchor(event) {
    return event.target instanceof Element ? event.target.closest('#app a[href]') : null;
  }

  function pageFile(link) {
    if (!link?.isConnected || link.hasAttribute('download') || (link.target && link.target !== '_self')) return null;
    try {
      const url = new URL(link.href, location.href);
      if (url.origin !== location.origin || !url.pathname.startsWith(site.value.base)) return null;
      const extension = url.pathname.match(/\.\w+$/)?.[0];
      if (extension && extension !== '.html') return null;
      return pathToFile(url.pathname);
    } catch {
      return null;
    }
  }

  function markCurrent() {
    try {
      const file = pathToFile(location.pathname);
      if (file) fetched.add(file);
    } catch {}
  }

  function prefetch(link) {
    if (!allowsPrefetch()) return;
    const file = pageFile(link);
    if (!file || fetched.has(file) || requests.has(file)) return;
    const hint = document.createElement('link');
    hint.rel = 'prefetch';
    hint.href = file;
    hint.as = 'script';
    requests.set(file, hint);
    const finish = success => {
      if (success) fetched.add(file);
      requests.delete(file);
      hint.onload = hint.onerror = null;
      hint.remove();
    };
    hint.onload = () => finish(true);
    hint.onerror = () => finish(false);
    document.head.append(hint);
  }

  function cancel(link) {
    clearTimeout(timers.get(link));
    timers.delete(link);
  }

  function hover(event) {
    if (!['mouse', 'pen'].includes(event.pointerType) || !allowsPrefetch()) return;
    const link = anchor(event);
    if (!link || (event.relatedTarget instanceof Node && link.contains(event.relatedTarget)) || timers.has(link)) return;
    timers.set(link, setTimeout(() => {
      timers.delete(link);
      prefetch(link);
    }, 100));
  }

  function leave(event) {
    const link = anchor(event);
    if (link && !(event.relatedTarget instanceof Node && link.contains(event.relatedTarget))) cancel(link);
  }

  function focus(event) {
    const link = anchor(event);
    cancel(link);
    if (link) prefetch(link);
  }

  function touch(event) {
    if (event.pointerType === 'touch' && event.isPrimary && event.button === 0) {
      const link = anchor(event);
      cancel(link);
      if (link) prefetch(link);
    }
  }

  function clearTimers() {
    for (const timer of timers.values()) clearTimeout(timer);
    timers.clear();
  }

  onMounted(() => {
    mounted = true;
    supported = document.createElement('link').relList?.supports('prefetch') === true;
    markCurrent();
    document.addEventListener('pointerover', hover, { passive: true });
    document.addEventListener('pointerout', leave, { passive: true });
    document.addEventListener('focusin', focus);
    document.addEventListener('pointerdown', touch, { passive: true });
  });

  watch(() => route.path, () => {
    clearTimers();
    if (mounted) markCurrent();
  });

  onBeforeUnmount(() => {
    mounted = false;
    clearTimers();
    document.removeEventListener('pointerover', hover);
    document.removeEventListener('pointerout', leave);
    document.removeEventListener('focusin', focus);
    document.removeEventListener('pointerdown', touch);
    for (const hint of requests.values()) {
      hint.onload = hint.onerror = null;
      hint.remove();
    }
    requests.clear();
  });
}
