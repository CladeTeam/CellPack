<script setup>
const checked = [
  'Buffer and workspace sizes',
  'Tables match metadata',
  'Element and index types',
  'Chunk metadata and cell tables',
  'Row selections and cell ids',
  'Element positions passed back in',
  'Column maps and schemas',
  'Plan entries against their cell',
]
const trusted = [
  'Record bytes of a chunk',
  'Ids and ranks decoded from a record',
  'Callback reads and writes',
  'External ring memory',
]
</script>

<template>
  <figure class="contract-boundary" aria-label="Data flows from the caller through entry point checks into trusted hot loops">
    <div class="boundary-source">
      <span class="boundary-source-label">Caller</span>
      <svg viewBox="0 0 48 16" aria-hidden="true"><path d="M1 8h44m-6-6 6 6-6 6" /></svg>
      <span>Arguments enter the API</span>
    </div>

    <div class="boundary-stages">
      <section class="boundary-stage boundary-checked" aria-label="Entry point checks">
        <header class="boundary-stage-header">
          <span class="boundary-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="m12 3 8 3v6c0 4-4 7-8 9-4-2-8-5-8-9V6l8-3Z" /><path d="m8 12 3 3 5-6" /></svg>
          </span>
          <span class="boundary-step">01 / VALIDATE</span>
          <span class="boundary-badge">Always checked</span>
        </header>
        <h3>Entry point checks</h3>
        <ul>
          <li v-for="item in checked" :key="item">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3 8 3 3 7-7" /></svg>
            <span>{{ item }}</span>
          </li>
        </ul>
        <footer>Invalid arguments <span aria-hidden="true">→</span> <code>Err</code></footer>
      </section>

      <div class="boundary-connector" aria-hidden="true">
        <svg viewBox="0 0 24 24"><path d="M4 12h16m-6-6 6 6-6 6" /></svg>
      </div>

      <section class="boundary-stage boundary-trusted" aria-label="Trusted hot loops">
        <header class="boundary-stage-header">
          <span class="boundary-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="m13 2-9 12h7l-1 8 10-13h-8l1-7Z" /></svg>
          </span>
          <span class="boundary-step">02 / EXECUTE</span>
          <span class="boundary-badge">Trusted data</span>
        </header>
        <h3>Hot loops</h3>
        <ul>
          <li v-for="item in trusted" :key="item">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v6m0 3v1" /></svg>
            <span>{{ item }}</span>
          </li>
        </ul>
        <div class="boundary-debug"><code>debug_assert!</code> checks in debug builds only.</div>
        <footer>Broken trusted data <span aria-hidden="true">→</span> <strong>Undefined behavior</strong></footer>
      </section>
    </div>
  </figure>
</template>

<style scoped>
.contract-boundary {
  margin: 24px 0;
  color: var(--vp-c-text-1);
  font-size: 14px;
  line-height: 1.6;
}
.boundary-source {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 18px;
  color: var(--vp-c-text-2);
  font-size: 13px;
}
.boundary-source-label {
  padding: 5px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-weight: 600;
}
.boundary-source svg { width: 36px; height: 16px; flex-shrink: 0; }
.contract-boundary svg {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.boundary-stages {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 28px minmax(0, 1fr);
  align-items: stretch;
}
.boundary-stage {
  --stage-color: var(--cp-teal);
  --stage-soft: var(--cp-teal-soft);
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-top: 3px solid var(--stage-color);
  border-radius: 10px;
  background: var(--vp-c-bg-elv);
}
.boundary-trusted {
  --stage-color: var(--cp-accent);
  --stage-soft: var(--cp-accent-soft);
}
.boundary-stage-header { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
.boundary-icon {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  color: var(--stage-color);
  background: var(--stage-soft);
}
.boundary-icon svg { width: 20px; height: 20px; }
.boundary-step {
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  letter-spacing: .04em;
}
.boundary-badge {
  padding: 2px 8px;
  border-radius: 4px;
  color: var(--stage-color);
  background: var(--stage-soft);
  font-size: 12px;
  font-weight: 600;
}
.contract-boundary h3 {
  margin: 16px 0 14px;
  border: 0;
  padding: 0;
  font-family: var(--cp-font-heading);
  font-size: 20px;
  line-height: 1.3;
  letter-spacing: -.4px;
}
.contract-boundary ul { margin: 0 0 20px; padding: 0; list-style: none; }
.contract-boundary li {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 0;
  padding: 5px 0;
  color: var(--vp-c-text-2);
  overflow-wrap: anywhere;
}
.contract-boundary li svg {
  flex-shrink: 0;
  width: 16px;
  height: 16px;
  margin-top: 3px;
  color: var(--stage-color);
}
.boundary-connector { display: grid; place-items: center; color: var(--vp-c-text-3); }
.boundary-connector svg { width: 20px; height: 20px; }
.boundary-debug {
  padding: 12px;
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 13px;
  margin-bottom: 20px;
}
.boundary-stage footer {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 4px 7px;
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 12px;
}
.boundary-stage footer strong { color: var(--stage-color); font-weight: 600; }
@media (max-width: 640px) {
  .boundary-stages { grid-template-columns: minmax(0, 1fr); }
  .boundary-connector { height: 32px; }
  .boundary-connector svg { transform: rotate(90deg); }
  .boundary-stage { padding: 18px; }
  .boundary-source { gap: 8px; font-size: 12px; }
}
</style>
