<script setup>
defineProps({ name: String });
</script>

<template>
  <svg class="story-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <g v-if="name === 'home'">
      <path d="m3 10 9-7 9 7M5 9v12h5v-7h4v7h5V9"/>
    </g>
    <g v-else-if="name === 'demand'">
      <rect x="9" y="6" width="10" height="12" rx="2"/>
      <path d="M12 3v3M16 3v3M12 18v3M16 18v3M19 9h3M19 15h3M2 12h11m-3-3 3 3-3 3"/>
    </g>
    <g v-else-if="name === 'storage'">
      <rect x="2" y="3" width="14" height="18" rx="2"/>
      <path d="M7 3v18M2 9h14M2 15h14"/>
      <rect x="7" y="9" width="9" height="6" fill="currentColor" fill-opacity="0.16" stroke="none"/>
      <path d="M22 12H12m3-3-3 3 3 3"/>
    </g>
    <g v-else-if="name === 'format'">
      <path d="M14 2H5a1 1 0 0 0-1 1v18a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1V8l-6-6ZM14 2v6h6M7 11h10M7 15h6M7 19h9"/>
    </g>
    <g v-else-if="name === 'loader'">
      <circle cx="3" cy="5" r="1.5"/>
      <circle cx="3" cy="12" r="1.5"/>
      <circle cx="3" cy="19" r="1.5"/>
      <path d="M5 5h3c6 0 2 7 8 7M5 12h11M5 19h3c6 0 2-7 8-7"/>
      <rect x="16" y="6" width="6" height="12" rx="1.5"/>
      <path d="M18.5 9h1M18.5 12h1M18.5 15h1"/>
    </g>
    <g v-else-if="name === 'practice'">
      <path d="M4 20a9 9 0 1 1 16 0M5 15h2M17 15h2M7 8l1.5 1.5M17 8l-1.5 1.5M12 5v2M13 14l4-4M7 21h10"/>
      <circle cx="12" cy="15" r="1.5"/>
    </g>
  </svg>
</template>
