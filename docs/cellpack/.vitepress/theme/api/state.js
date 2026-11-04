import { ref } from 'vue';
export const definition = ref(null);
export function showDefinition(key) { definition.value = { key }; }
