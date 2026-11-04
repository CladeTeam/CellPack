import { nextTick, onMounted, onUnmounted, watch } from 'vue';
import { useRoute } from 'vitepress';

export function useDisclosureAnchors() {
  const route = useRoute();
  function reveal() {
    if (typeof window === 'undefined') return;
    const hash = location.hash;
    if (!hash) return;
    let target;
    try { target = document.getElementById(decodeURIComponent(hash.slice(1))); }
    catch { return; }
    if (!target?.closest('.vp-doc')) return;
    let opened = false;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS' && !parent.open) {
        parent.open = true;
        opened = true;
      }
    }
    if (opened) requestAnimationFrame(() => {
      if (target.isConnected && location.hash === hash) target.scrollIntoView({ block: 'start', behavior: 'instant' });
    });
  }
  function followCurrentAnchor(event) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.('a[href]');
    if (!link) return;
    const url = new URL(link.href, location.href);
    if (url.href === location.href) reveal();
  }
  onMounted(() => {
    reveal();
    window.addEventListener('hashchange', reveal);
    document.addEventListener('click', followCurrentAnchor);
  });
  watch(() => route.path, async () => { await nextTick(); reveal(); });
  onUnmounted(() => {
    window.removeEventListener('hashchange', reveal);
    document.removeEventListener('click', followCurrentAnchor);
  });
}
