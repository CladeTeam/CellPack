<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useData, withBase } from 'vitepress';
import { counterpart } from '../../routes.mjs';
const { page, theme } = useData();
const picker = ref(null);
const area = computed(() => theme.value.subsite);
const close = () => { if (picker.value) picker.value.open = false; };
function outside(event) { if (picker.value && !picker.value.contains(event.target)) close(); }
function escape(event) { if (event.key === 'Escape' && picker.value?.open) { close(); picker.value.querySelector('summary').focus(); } }
watch(() => page.value.relativePath, close);
onMounted(() => { document.addEventListener('click', outside); document.addEventListener('keydown', escape); });
onBeforeUnmount(() => { document.removeEventListener('click', outside); document.removeEventListener('keydown', escape); });
</script>

<template>
  <div class="VPNavBarTitle cp-site-title" :data-subsite="area || 'project'">
    <a v-if="area" class="cp-project-home" :href="withBase('/')" aria-label="Project home" title="Project home"><svg viewBox="0 0 32 32" aria-hidden="true"><ellipse cx="16" cy="28.4" rx="12" ry="1.7" fill="#0B2E27" fill-opacity=".16"/><rect x="20.6" y="5.6" width="3" height="7" fill="#1E7A68"/><rect x="20.2" y="4.8" width="3.8" height="1.4" rx=".4" fill="#52BBA2"/><polygon points="7,13.9 16,6.6 25,13.9 25,28 7,28" fill="#2BA58A"/><polygon points="16,6.6 25,13.9 25,28 16,28" fill="#000" fill-opacity=".12"/><polygon points="7,13.9 16,6.6 25,13.9 25,15.6 16,8.3 7,15.6" fill="#000" fill-opacity=".14"/><rect x="8.9" y="18.4" width="3" height="3" rx=".5" fill="#E9FBF5" fill-opacity=".85"/><rect x="20.1" y="18.4" width="3" height="3" rx=".5" fill="#E9FBF5" fill-opacity=".3"/><path d="M13.3 28V21.7a2.7 2.7 0 0 1 5.4 0V28Z" fill="#E9FBF5" fill-opacity=".9"/><path d="M16 19V28h2.7V21.7A2.7 2.7 0 0 0 16 19Z" fill="#000" fill-opacity=".07"/><circle cx="17.5" cy="24.4" r=".45" fill="#1E7A68"/><polygon points="2.6,15.2 16,4.3 16,7.5 4.3,17.1" fill="#5CC7AE"/><polygon points="16,4.3 29.4,15.2 27.7,17.1 16,7.5" fill="#52BBA2"/><polygon points="2.6,15.2 16,4.3 16,5.2 3.3,15.6" fill="#E9FBF5" fill-opacity=".45"/><circle cx="16" cy="13.4" r="2.7" fill="#FFFFFF"/><circle cx="16" cy="13.4" r="2.1" fill="#EEE8FF" stroke="#7B5CD6" stroke-width=".55"/><circle cx="15.3" cy="13.7" r=".85" fill="#7B5CD6"/><circle cx="16.9" cy="12.9" r=".28" fill="#B9A6EE"/></svg></a>
    <a class="title" :href="withBase(theme.logoLink || '/')" :aria-label="(theme.siteTitle || 'CellPack') + ' home'">
      <img class="logo" :src="withBase('/logo.svg')" width="28" height="28" alt="" />
      <span>{{ theme.siteTitle || 'CellPack' }}</span>
    </a>
    <details v-if="area" ref="picker" class="cp-surface-picker">
      <summary aria-label="Switch documentation site"><img :src="withBase('/brands/' + area + '.svg')" class="cp-language-logo" :class="{ 'cp-rust-mark': area === 'rust' }" width="16" height="16" alt="" />{{ area === 'python' ? 'Python' : 'Rust' }}<svg viewBox="0 0 12 12" aria-hidden="true"><path d="m3 4.5 3 3 3-3" /></svg></summary>
      <div class="cp-surface-menu">
        <span>DOCUMENTATION</span>
        <a :href="withBase(counterpart(page.relativePath, 'rust'))" :aria-current="area === 'rust' ? 'true' : undefined"><span><img :src="withBase('/brands/rust.svg')" class="cp-language-logo cp-rust-mark" width="16" height="16" alt="" />CellPack</span><small>Rust</small></a>
        <a :href="withBase(counterpart(page.relativePath, 'python'))" :aria-current="area === 'python' ? 'true' : undefined"><span><img :src="withBase('/brands/python.svg')" class="cp-language-logo" width="16" height="16" alt="" />PyCellPack</span><small>Python</small></a>
      </div>
    </details>
  </div>
</template>
