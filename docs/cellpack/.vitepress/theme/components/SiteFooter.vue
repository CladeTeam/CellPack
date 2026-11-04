<script setup>
import { withBase } from "vitepress";
</script>

<template>
  <footer class="cp-footer">
    <a :href="withBase('/')" class="cp-footer-brand"
      ><img
        :src="withBase('/logo.svg')"
        width="28"
        height="28"
        alt=""
      />CellPack</a
    >
    <p>Less storage. Faster loading.</p>
    <div>
      <a :href="withBase('/overview.html')">Documentation</a
      ><a :href="withBase('/report.html')"
        >Technical report <span aria-hidden="true">↗</span></a
      >
    </div>
  </footer>
</template>
