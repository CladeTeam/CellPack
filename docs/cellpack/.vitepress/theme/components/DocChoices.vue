<script setup>
import { withBase } from 'vitepress';

defineProps({
  title: { type: String, required: true },
  items: { type: Array, required: true },
});
</script>

<template>
  <section class="doc-choices" :aria-label="title">
    <p class="choices-title">{{ title }}</p>
    <ul class="choices-grid">
      <li v-for="item in items" :key="item.label">
        <component :is="item.href ? 'a' : 'div'" class="choice" :href="item.href ? (item.href.startsWith('#') ? item.href : withBase(item.href)) : undefined">
          <span class="choice-when">{{ item.when }}</span>
          <strong class="choice-label">{{ item.label }}<span v-if="item.href" class="choice-arrow" aria-hidden="true">→</span></strong>
          <span v-if="item.detail" class="choice-detail">{{ item.detail }}</span>
        </component>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.doc-choices {
  margin: 24px 0;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
}
.doc-choices .choices-title {
  margin: 0 0 14px;
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 650;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.doc-choices .choices-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.choices-grid > li { min-width: 0; margin: 0; }
.doc-choices .choice {
  display: flex;
  flex-direction: column;
  gap: 7px;
  height: 100%;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text-1);
  text-decoration: none;
  overflow-wrap: anywhere;
}
a.choice { transition: border-color .15s ease, background-color .15s ease; }
a.choice:hover { border-color: var(--cp-teal); background: var(--vp-c-bg); }
a.choice:focus-visible { outline: 2px solid var(--cp-teal); outline-offset: 3px; }
.choice-when { color: var(--vp-c-text-2); font-size: 13px; line-height: 1.55; }
.choice-label { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; font-family: var(--cp-font-heading); font-size: 17px; line-height: 1.4; }
.choice-arrow { flex: none; color: var(--cp-teal); }
.choice-detail { color: var(--vp-c-text-2); font-size: 13px; line-height: 1.65; }
@media (max-width: 640px) {
  .doc-choices { padding: 16px; }
  .doc-choices .choices-grid { grid-template-columns: minmax(0, 1fr); }
}
@media (prefers-reduced-motion: reduce) { a.choice { transition: none; } }
</style>
