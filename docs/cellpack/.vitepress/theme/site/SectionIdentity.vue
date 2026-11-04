<script setup>
import { computed } from 'vue';
import { withBase } from 'vitepress';
const props = defineProps({ area: String, section: String });
const label = computed(() => props.section === 'api' ? 'API' : props.section === 'training' ? 'TRAINING' : 'DOCS');
const description = computed(() => (props.area === 'python' ? 'Python' : 'Rust') + (props.section === 'api' ? ' interfaces & contracts' : props.section === 'training' ? ' examples & workflows' : ' guides & concepts'));
</script>
<template>
  <a class="cp-sidebar-section" :href="withBase('/' + area + '/' + section + '/')">
    <div><svg v-if="section === 'api'" viewBox="0 0 20 20" aria-hidden="true"><path d="m7 5-4 5 4 5m6-10 4 5-4 5M11 3 9 17" /></svg><svg v-else-if="section === 'training'" class="cp-training-icon" viewBox="0 0 20 20" aria-hidden="true"><path d="M3 6L10 3M3 6L10 10M3 6L10 17M3 14L10 3M3 14L10 10M3 14L10 17M10 3L17 6M10 3L17 14M10 10L17 6M10 10L17 14M10 17L17 6M10 17L17 14" /><circle cx="3" cy="6" r="1.7" /><circle cx="3" cy="14" r="1.7" /><circle cx="10" cy="3" r="1.7" /><circle cx="10" cy="10" r="1.7" /><circle cx="10" cy="17" r="1.7" /><circle cx="17" cy="6" r="1.7" /><circle cx="17" cy="14" r="1.7" /></svg><svg v-else viewBox="0 0 20 20" aria-hidden="true"><path d="M3 4h5a3 3 0 0 1 2 1 3 3 0 0 1 2-1h5v12h-5a3 3 0 0 0-2 1 3 3 0 0 0-2-1H3ZM10 5v12" /></svg><strong>{{ label }}</strong></div>
    <span>{{ description }}</span>
  </a>
</template>
