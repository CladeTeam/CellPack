<script setup>
import { computed, defineAsyncComponent, provide } from 'vue';
import { dataSymbol, useData, withBase } from 'vitepress';
import DefaultTheme from 'vitepress/theme-without-fonts';
import SiteActions from '../components/SiteActions.vue';
import Breadcrumbs from './Breadcrumbs.vue';
import SectionIdentity from './SectionIdentity.vue';
import { surfaceOf } from '../../routes.mjs';
import { useDisclosureAnchors } from './useDisclosureAnchors.js';
import { useIntentPrefetch } from './useIntentPrefetch.js';

useDisclosureAnchors();
useIntentPrefetch();

const data = useData();
const area = computed(() => surfaceOf(data.page.value.relativePath));
const section = computed(() => data.page.value.relativePath.split('/')[1]);
const layoutClasses = computed(() => area.value ? ['cp-subsite-' + area.value, 'cp-section-' + section.value] : []);
const theme = computed(() => {
  const original = data.theme.value;
  if (!area.value) return original;
  const prefix = data.site.value.base + area.value + '/';
  return {
    ...original, ...original.surfaces[area.value],
    search: { ...original.search, options: { ...original.search.options,
      miniSearch: { ...original.search.options?.miniSearch, searchOptions: {
        ...original.search.options?.miniSearch?.searchOptions,
        filter: result => result.id.startsWith(prefix),
      } },
    } },
  };
});
const site = computed(() => ({ ...data.site.value, title: theme.value.siteTitle || 'CellPack', themeConfig: theme.value }));
provide(dataSymbol, { ...data, theme, site });
const ApiDefinitions = defineAsyncComponent(() => import('../api/ApiDefinitions.vue'));
const isRustApi = computed(() => data.page.value.relativePath.startsWith('rust/api/'));
</script>

<template>
  <DefaultTheme.Layout :class="layoutClasses">
    <template #nav-bar-content-after><SiteActions v-if="!area" /></template>
    <template #nav-screen-content-after><SiteActions v-if="!area" mobile /></template>
    <template #sidebar-nav-before><SectionIdentity v-if="area" :area="area" :section="section" /></template>
    <template #doc-before>
      <div v-if="section === 'api' && area" class="cp-reference-banner"><span><svg viewBox="0 0 20 20" aria-hidden="true"><path d="m7 5-4 5 4 5m6-10 4 5-4 5M11 3 9 17" /></svg>API REFERENCE</span><a :href="withBase('/' + area + '/docs/')">Usage guides →</a></div>
      <Breadcrumbs v-if="area" />
    </template>
    <template #doc-after><ApiDefinitions v-if="isRustApi" /></template>
  </DefaultTheme.Layout>
</template>
