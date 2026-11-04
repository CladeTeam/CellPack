<script setup>
import { computed } from 'vue';
import { useData, withBase } from 'vitepress';
import { surfaceOf } from '../../routes.mjs';
const { page } = useData();
const sections = { docs: 'Docs', api: 'API', training: 'Training' };
const crumbs = computed(() => {
  const path = page.value.relativePath, area = surfaceOf(path);
  const items = [{ text: area === 'python' ? 'PyCellPack' : 'CellPack', link: '/' + area + '/' }];
  const section = path.split('/')[1];
  if (sections[section]) items.push({ text: sections[section], link: '/' + area + '/' + section + '/' });
  return items;
});
</script>
<template>
  <nav class="cp-breadcrumbs" aria-label="Breadcrumb"><template v-for="(crumb, index) in crumbs" :key="crumb.link"><span v-if="index" aria-hidden="true">/</span><a :href="withBase(crumb.link)">{{ crumb.text }}</a></template></nav>
</template>
