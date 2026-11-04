<script setup>
import { useData, withBase } from "vitepress";
defineProps({ mobile: Boolean });
const { theme } = useData();
</script>

<template>
  <div class="cp-nav-actions" :class="{ 'is-mobile': mobile }">
    <a
      v-if="theme.repositoryUrl"
      :href="theme.repositoryUrl"
      class="cp-repository"
      aria-label="CellPack on GitHub"
      target="_blank"
      rel="noopener noreferrer"
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M9 19c-4.3 1.3-4.3-2.2-6-2.7m12 5v-3.4a3 3 0 0 0-.8-2.3c2.7-.3 5.6-1.3 5.6-6A4.7 4.7 0 0 0 18.5 6a4.4 4.4 0 0 0-.1-3.5s-1-.3-3.6 1.3a12.4 12.4 0 0 0-6.6 0C5.6 2.2 4.6 2.5 4.6 2.5A4.4 4.4 0 0 0 4.5 6a4.7 4.7 0 0 0-1.3 3.6c0 4.7 2.9 5.7 5.6 6A3 3 0 0 0 8 18v3.3"
        />
      </svg>
      <span>GitHub</span>
    </a>
    <span
      v-else
      class="cp-repository is-placeholder"
      role="link"
      aria-disabled="true"
      title="Repository link coming soon"
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M9 19c-4.3 1.3-4.3-2.2-6-2.7m12 5v-3.4a3 3 0 0 0-.8-2.3c2.7-.3 5.6-1.3 5.6-6A4.7 4.7 0 0 0 18.5 6a4.4 4.4 0 0 0-.1-3.5s-1-.3-3.6 1.3a12.4 12.4 0 0 0-6.6 0C5.6 2.2 4.6 2.5 4.6 2.5A4.4 4.4 0 0 0 4.5 6a4.7 4.7 0 0 0-1.3 3.6c0 4.7 2.9 5.7 5.6 6A3 3 0 0 0 8 18v3.3"
        />
      </svg>
      <span>GitHub <small>Soon</small></span>
    </span>
    <a class="cp-nav-start" :href="withBase('/installation.html')"
      >Get started <span aria-hidden="true">↗</span></a
    >
  </div>
</template>
