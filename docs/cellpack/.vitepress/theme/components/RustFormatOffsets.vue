<template>
  <figure class="format-offsets">
    <figcaption>One record, three relative offsets</figcaption>
    <p class="format-offsets-intro">Illustrative packed stream embedded at outer-file byte 8192. Every number below is in bytes.</p>
    <ol>
      <li>
        <span class="offset-level">Stream → chunk</span>
        <strong>4096</strong>
        <code>entry.offset</code>
        <p>Measured from the CPK stream start.</p>
      </li>
      <li>
        <span class="offset-level">Chunk → records</span>
        <strong>4096</strong>
        <code>meta.records_start()</code>
        <p>Includes the metadata and alignment padding.</p>
      </li>
      <li>
        <span class="offset-level">Records → row</span>
        <strong>128..192</strong>
        <code>meta.record_range(row)?</code>
        <p>A half-open range in the records region.</p>
      </li>
    </ol>
    <div class="offset-result">
      <span>Outer-file record range</span>
      <code>8192 + 4096 + 4096 + (128..192) = 16512..16576</code>
    </div>
  </figure>
</template>

<style scoped>
.format-offsets {
  margin: 24px 0;
  padding: 22px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
}
.format-offsets figcaption {
  font-family: var(--cp-font-heading);
  font-size: 18px;
  font-weight: 600;
}
.format-offsets .format-offsets-intro {
  margin: 8px 0 18px;
  color: var(--vp-c-text-2);
  font-size: 14px;
  line-height: 1.65;
}
.format-offsets ol {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.format-offsets li {
  min-width: 0;
  margin: 0;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-top: 3px solid var(--cp-teal);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
}
.offset-level { display: block; font-size: 13px; color: var(--vp-c-text-2); }
.format-offsets strong {
  display: block;
  margin: 8px 0;
  font-family: var(--vp-font-family-mono);
  font-size: 20px;
  font-weight: 500;
}
.format-offsets code {
  font-size: 12px;
  overflow-wrap: anywhere;
  white-space: normal;
}
.format-offsets li p {
  margin: 10px 0 0;
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.65;
}
.offset-result {
  display: grid;
  gap: 8px;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--vp-c-divider);
}
.offset-result span { font-size: 13px; font-weight: 600; }
.offset-result code { color: var(--vp-c-text-1); }
@media (max-width: 640px) {
  .format-offsets { padding: 16px; }
  .format-offsets ol { grid-template-columns: minmax(0, 1fr); }
}
</style>
