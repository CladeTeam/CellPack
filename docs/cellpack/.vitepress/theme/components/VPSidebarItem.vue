<script setup>
import { computed, ref, useId, watch } from "vue";
import { useData, withBase } from "vitepress";

const props = defineProps({
  item: { type: Object, required: true },
  depth: { type: Number, default: 0 },
});
const { page, site } = useData();
const childrenId = `cp-sidebar-${useId()}`;
const hasChildren = computed(() => Boolean(props.item.items?.length));
const headingTag = computed(() => `h${Math.min(props.depth + 2, 6)}`);

function pagePath(link = "") {
  const path = link.split(/[?#]/)[0].replace(/^\//, "");
  return path
    .replace(/\.(md|html)$/, "")
    .replace(/(^|\/)index$/, "$1")
    .replace(/\/$/, "");
}

function isCurrent(link) {
  return (
    Boolean(link) &&
    !/^(?:[a-z][\w+.-]*:|\/\/|#)/i.test(link) &&
    pagePath(link) === pagePath(page.value.relativePath)
  );
}

function containsCurrent(item) {
  return isCurrent(item.link) || Boolean(item.items?.some(containsCurrent));
}

function href(link) {
  if (!link || /^(?:[a-z][\w+.-]*:|\/\/|#)/i.test(link)) return link;
  const [, path, suffix] = link.match(/^([^?#]*)(.*)$/);
  const normalized = /\/$|\.\w+$/.test(path)
    ? path.replace(/\.md$/, site.value.cleanUrls ? "" : ".html")
    : path + (site.value.cleanUrls ? "" : ".html");
  return withBase(normalized + suffix);
}

const active = computed(() => isCurrent(props.item.link));
const activeGroup = computed(() => containsCurrent(props.item));
const expanded = ref(props.item.collapsed !== true || activeGroup.value);
// Open the destination's ancestors on navigation; explicit toggles own the current page's state.
watch(
  () => page.value.relativePath,
  () => {
    if (activeGroup.value) expanded.value = true;
  },
);
</script>

<template>
  <component
    :is="hasChildren ? 'section' : 'div'"
    class="VPSidebarItem cp-sidebar-item"
    :class="[
      `level-${depth}`,
      {
        collapsed: hasChildren && !expanded,
        collapsible: hasChildren,
        'is-active': active,
        'has-active': activeGroup,
      },
    ]"
  >
    <div class="item cp-sidebar-row">
      <component v-if="hasChildren" :is="headingTag" class="cp-sidebar-heading">
        <button
          type="button"
          class="cp-sidebar-toggle"
          :aria-expanded="expanded"
          :aria-controls="childrenId"
          @click="expanded = !expanded"
        >
          <span class="text" v-html="item.text" />
          <svg
            class="cp-sidebar-chevron"
            :class="{ expanded }"
            viewBox="0 0 16 16"
            aria-hidden="true"
          >
            <path d="m6 3 5 5-5 5" />
          </svg>
        </button>
      </component>
      <a
        v-else-if="item.link"
        class="link cp-sidebar-link"
        :href="href(item.link)"
        :target="item.target"
        :rel="item.rel"
        :aria-current="active ? 'page' : undefined"
        ><span class="text" v-html="item.text"
      /></a>
      <span v-else class="text" v-html="item.text" />
    </div>
    <div v-if="hasChildren" v-show="expanded" :id="childrenId" class="items">
      <template v-if="expanded">
        <a
          v-if="item.link"
          class="link cp-sidebar-link cp-sidebar-overview"
          :href="href(item.link)"
          :target="item.target"
          :rel="item.rel"
          :aria-current="active ? 'page' : undefined"
          :aria-label="`${item.text} overview`"
          ><span class="text">Overview</span></a
        >
        <VPSidebarItem
          v-for="child in item.items"
          :key="child.link || child.text"
          :item="child"
          :depth="depth + 1"
        />
      </template>
    </div>
  </component>
</template>

<style scoped>
.cp-sidebar-item {
  user-select: none;
}
.cp-sidebar-item.level-0 {
  padding-bottom: 20px;
}
.cp-sidebar-item.level-0.collapsed {
  padding-bottom: 8px;
}
.cp-sidebar-row {
  width: 100%;
}
.cp-sidebar-heading {
  margin: 0;
  width: 100%;
}
.cp-sidebar-toggle,
.cp-sidebar-link {
  display: flex;
  align-items: center;
  width: 100%;
  border-radius: 4px;
  color: var(--vp-c-text-2);
  text-align: start;
}
.cp-sidebar-toggle {
  justify-content: space-between;
  gap: 10px;
  padding: 3px 6px;
  margin-inline: -6px;
  width: calc(100% + 12px);
  touch-action: manipulation;
}
.text {
  display: block;
  min-width: 0;
  padding: 5px 0;
  font-size: 14px;
  line-height: 22px;
  font-weight: 450;
  overflow-wrap: anywhere;
}
.level-0 > .cp-sidebar-row .text {
  font-weight: 650;
  color: var(--vp-c-text-1);
}
.cp-sidebar-toggle:hover {
  background: var(--vp-c-default-soft);
}
.cp-sidebar-link:hover .text,
.has-active > .cp-sidebar-row .text,
.cp-sidebar-link[aria-current="page"] .text {
  color: var(--vp-c-brand-1);
}
.cp-sidebar-link {
  padding-inline: 6px;
  margin-inline: -6px;
  width: calc(100% + 12px);
}
.cp-sidebar-link[aria-current="page"] {
  background: var(--vp-c-brand-soft);
}
.cp-sidebar-item .cp-sidebar-toggle:focus-visible,
.cp-sidebar-item .cp-sidebar-link:focus-visible {
  outline: 2px solid var(--cp-teal);
  outline-offset: -2px;
}
.cp-sidebar-item .cp-sidebar-toggle:focus:not(:focus-visible),
.cp-sidebar-item .cp-sidebar-link:focus:not(:focus-visible) {
  outline: none;
}
.cp-sidebar-chevron {
  width: 14px;
  height: 14px;
  flex: none;
  fill: none;
  stroke: var(--vp-c-text-3);
  stroke-width: 1.6;
  transition: transform 120ms ease;
}
.cp-sidebar-chevron.expanded {
  transform: rotate(90deg);
}
.items {
  padding-top: 3px;
}
.level-1 > .items,
.level-2 > .items,
.level-3 > .items,
.level-4 > .items {
  border-left: 1px solid var(--vp-c-divider);
  padding-left: 14px;
}
@media (pointer: coarse) {
  .cp-sidebar-toggle,
  .cp-sidebar-link {
    min-height: 40px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .cp-sidebar-chevron {
    transition: none;
  }
}
</style>
