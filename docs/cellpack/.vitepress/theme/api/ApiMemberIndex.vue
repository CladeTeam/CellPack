<script setup>
import { computed, ref } from 'vue';
const props = defineProps({ members: { type: Array, required: true } });
const query = ref('');
const filtered = computed(() => props.members.filter(m => `${m.name} ${m.summary}`.toLowerCase().includes(query.value.toLowerCase())));
</script>

<template>
  <div class="cp-member-index">
    <label>Find a member<input v-model="query" type="search" placeholder="Filter methods by name or purpose" /></label>
    <p role="status" class="cp-api-result-count">{{ filtered.length }} of {{ members.length }} members</p>
    <ul><li v-for="member in filtered" :key="member.href"><a :href="member.href"><code>{{ member.name }}</code></a><span>{{ member.summary }}</span></li></ul>
    <p v-if="!filtered.length">No matching members.</p>
  </div>
</template>
