<script setup>
defineProps({ kind: { type: String, required: true } });
const cells = ["A", "B", "C", "D", "E", "F"];
const sequence = ["A", "B", "C", "A", "D", "B", "A", "C"];
</script>

<template>
  <div class="cp-diagram" :class="`cp-diagram-${kind}`">
    <template v-if="kind === 'records'">
      <div class="cp-diagram-head">
        <span class="cp-micro">ONE CHUNK. MANY INDEPENDENT RECORDS.</span
        ><span class="cp-diagram-index">01</span>
      </div>
      <div class="cp-shared-model">
        <svg
          class="cp-model-mark"
          width="30"
          height="30"
          viewBox="0 0 30 30"
          aria-hidden="true"
        >
          <path
            d="M15 2v26M2 15h26M6 6l18 18M6 24 24 6"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          />
        </svg>
        <div>
          <strong>Shared coding models</strong
          ><small>Statistical context for the whole chunk</small>
        </div>
        <span class="cp-model-status">SHARED</span>
      </div>
      <div class="cp-model-branches" aria-hidden="true">
        <i v-for="cell in cells" :key="cell" />
      </div>
      <div class="cp-records">
        <div
          v-for="(cell, i) in cells"
          :key="cell"
          class="cp-record"
          :class="{ selected: i === 2 }"
        >
          <span>{{ cell }}</span>
          <div>
            <i
              v-for="n in 4"
              :key="n"
              :style="{ width: `${25 + ((i * 17 + n * 23) % 55)}%` }"
            />
          </div>
        </div>
      </div>
      <div class="cp-record-read">
        <span class="cp-micro">REQUEST C</span>
        <div class="cp-read-track"><i /></div>
        <span class="cp-decoded-record"
          >C <span aria-hidden="true">→</span> decoded row</span
        >
      </div>
      <p class="cp-diagram-caption">
        Read the record you need.<br /><strong
          >Leave the neighbours where they are.</strong
        >
      </p>
    </template>

    <template v-else-if="kind === 'pipeline'">
      <div class="cp-diagram-head">
        <span class="cp-micro">CONCURRENT WORK. A SINGLE ORDER.</span
        ><span class="cp-diagram-index">02</span>
      </div>
      <div class="cp-access-sequence">
        <span class="cp-micro">REQUESTS</span>
        <div>
          <span v-for="n in 6" :key="n">{{ String(n).padStart(2, "0") }}</span>
        </div>
      </div>
      <div class="cp-workers">
        <div v-for="worker in 3" :key="worker" class="cp-worker">
          <span class="cp-micro">W{{ worker - 1 }}</span>
          <div class="cp-worker-track">
            <i
              class="cp-worker-read"
              :style="{ '--delay': `${worker * -1.3}s` }"
              >READ</i
            ><i
              class="cp-worker-decode"
              :style="{ '--delay': `${worker * -1.3}s` }"
              >DECODE</i
            ><i
              class="cp-worker-build"
              :style="{ '--delay': `${worker * -1.3}s` }"
              >BUILD</i
            >
          </div>
        </div>
      </div>
      <div class="cp-ring-label">
        <span class="cp-micro">BOUNDED OUTPUT RING</span
        ><span>in requested order <span aria-hidden="true">↓</span></span>
      </div>
      <div class="cp-output-ring">
        <div v-for="n in 6" :key="n" :style="{ '--delay': `${n * -0.65}s` }">
          <span>{{ String(n).padStart(2, "0") }}</span
          ><i /><i /><i />
        </div>
      </div>
      <p class="cp-diagram-caption">
        Workers finish at different times.<br /><strong
          >Your batches keep their order.</strong
        >
      </p>
    </template>

    <template v-else>
      <div class="cp-diagram-head">
        <span class="cp-micro">LOOK AHEAD. KEEP WHAT MATTERS.</span
        ><span class="cp-diagram-index">03</span>
      </div>
      <div class="cp-reuse-sequence">
        <span
          v-for="(cell, i) in sequence"
          :key="i"
          :class="{ 'reuse-a': cell === 'A', 'reuse-b': cell === 'B' }"
          >{{ cell }}<small>{{ i + 1 }}</small></span
        >
      </div>
      <svg class="cp-reuse-arcs" viewBox="0 0 480 90" aria-hidden="true">
        <path
          class="cp-arc-page"
          d="M30 4 Q120 130 210 4M210 4 Q300 130 390 4"
        />
        <path class="cp-arc-row" d="M90 4 Q210 105 330 4" />
        <circle class="cp-reuse-dot" cx="30" cy="4" r="4" />
      </svg>
      <div class="cp-cache-choice">
        <div>
          <span class="cp-cache-symbol page" aria-hidden="true">▤</span
          ><strong>Compressed page</strong>
          <p>Save a storage read</p>
          <span class="cp-cache-bytes"><i /><i /><i /></span>
        </div>
        <div>
          <span class="cp-cache-symbol row" aria-hidden="true">▦</span
          ><strong>Decoded row</strong>
          <p>Save a read + decoding</p>
          <span class="cp-cache-bytes"><i /><i /><i /><i /><i /><i /></span>
        </div>
      </div>
      <div class="cp-budget">
        <span class="cp-micro">ONE CACHE BUDGET</span>
        <div><i /><i /></div>
      </div>
      <p class="cp-diagram-caption">
        Different costs. Different sizes.<br /><strong
          >A plan that accounts for both.</strong
        >
      </p>
    </template>
    <span class="cp-diagram-footnote"
      >CONCEPTUAL ILLUSTRATION · BASED ON THE TECHNICAL REPORT</span
    >
  </div>
</template>
