<script setup>
defineProps({ kind: String, stage: Number });
const indices = ["4", "19", "41", "88", "103", "119"];
const gaps = ["0", "0", "1", "0", "2", "0"];
const values = ["1.5", "1.5", "3.2", "1.5", "3.2", "7.0"];
</script>
<template>
  <div class="format-diagram" :class="[`diagram-${kind}`, `phase-${stage}`]">
    <template v-if="kind === 'scale'">
      <div class="matrix" :class="{ sparse: stage > 0 }">
        <span
          v-for="i in 144"
          :key="i"
          :class="{ nonzero: [8, 31, 56, 90, 115, 139].includes(i) }"
          >{{ [8, 31, 56, 90, 115, 139].includes(i) ? "3" : "0" }}</span
        >
      </div>
      <div class="scale-result">
        <span>{{
          stage === 0
            ? "DENSE · float32"
            : stage === 1
              ? "CSR · values + indices"
              : "COMPRESSED CSR"
        }}</span
        ><strong>{{
          stage === 0 ? "160 TB" : stage === 1 ? "16 TB" : "16 TB ÷ R"
        }}</strong
        ><small>{{
          stage === 0
            ? "Store every entry, including zeros."
            : stage === 1
              ? "Store nonzeros and their positions."
              : "R is the compression ratio. Access cost is a separate question."
        }}</small>
      </div>
    </template>
    <template v-else-if="kind === 'mmap' || kind === 'chunk'">
      <div class="flow-label">
        <b>Loader</b><span class="access-token">access(cell 07)</span>
      </div>
      <div class="flow-arrow" :class="{ active: stage >= 1 }">
        ↓
        <span>{{
          kind === "mmap"
            ? stage === 0
              ? "virtual address"
              : "page fault · cold cache"
            : "locate compressed chunk"
        }}</span>
      </div>
      <div class="storage-box" :class="{ active: stage >= 1 }">
        <small>{{
          kind === "mmap"
            ? "STORAGE · CSR arrays"
            : "STORAGE · compressed chunk"
        }}</small>
        <div class="record-strip">
          <span
            v-for="i in 12"
            :key="i"
            :class="{
              target: i === 7,
              fetched: stage >= 2 && (kind === 'chunk' || i === 7),
              discarded: kind === 'chunk' && stage >= 3 && i !== 7,
            }"
            >{{ String(i).padStart(2, "0") }}</span
          >
        </div>
        <small>{{
          kind === "mmap"
            ? "indptr → indices + data · page-granular reads"
            : "One compression unit contains many cells"
        }}</small>
      </div>
      <div class="flow-arrow" :class="{ active: stage >= 2 }">
        ↓
        <span>{{
          kind === "mmap"
            ? "storage I/O → OS page cache"
            : "read chunk → decompress ALL cells"
        }}</span>
      </div>
      <div class="output-cell" :class="{ active: stage >= 3 }">
        cell 07 → batch
        <span v-if="kind === 'chunk'"
          >11 / 12 decoded cells unused in this example</span
        ><span v-else>Wait for pages, then gather the requested row</span>
      </div>
    </template>
    <template v-else-if="kind === 'native'">
      <small>ONE CELL · native CSR</small>
      <div class="array-row">
        <label>indices</label><span v-for="v in indices" :key="v">{{ v }}</span>
      </div>
      <div class="array-row">
        <label>values</label
        ><span v-for="(v, i) in values" :key="i">{{ v }}</span>
      </div>
      <div class="flow-arrow active">
        ↓
        <span>{{
          stage === 0
            ? "Keep the row boundary"
            : "Bring the two slices into one record"
        }}</span>
      </div>
      <div class="record-box" :class="{ active: stage > 0 }">
        <b>cell record</b><span>nnz</span><span>indices</span
        ><span>values</span>
      </div>
      <p class="diagram-conclusion">
        {{
          stage === 2
            ? "A directory maps cell id → record offset + length."
            : "One cell becomes one independently addressable record."
        }}
      </p>
    </template>
    <template v-else-if="kind === 'indices'">
      <small>{{
        stage === 0
          ? "ORIGINAL GENE IDS"
          : "FREQUENT GENES RECEIVE SMALLER CODED IDS"
      }}</small>
      <div class="array-row">
        <span v-for="(v, i) in indices" :key="i">{{
          stage === 0 ? v : [0, 1, 3, 4, 7, 8][i]
        }}</span>
      </div>
      <div class="flow-arrow active">
        ↓
        <span>{{
          stage < 2
            ? "relabel + sort within the cell"
            : "first index + skipped positions between indices"
        }}</span>
      </div>
      <div class="array-row" :class="{ encoded: stage === 2 }">
        <span v-for="(v, i) in gaps" :key="i">{{
          stage < 2 ? [0, 1, 3, 4, 7, 8][i] : v
        }}</span>
      </div>
      <p class="diagram-conclusion">
        {{
          stage === 2
            ? "Small, repeated gaps are easier to entropy-code."
            : "The shared inverse permutation restores original gene identities."
        }}
      </p>
    </template>
    <template v-else-if="kind === 'values'">
      <small>REPEATED VALUES WITHIN A CELL</small>
      <div class="array-row">
        <span
          v-for="(v, i) in values"
          :key="i"
          :class="{ target: v === '1.5' }"
          >{{ stage === 0 ? v : [0, 0, 1, 0, 1, 2][i] }}</span
        >
      </div>
      <div class="flow-arrow active">
        ↓
        <span>{{
          stage === 0
            ? "Preserve exact floating-point bit patterns"
            : "Dictionary + rank stream"
        }}</span>
      </div>
      <div class="dictionary">
        <span>0 → 1.5</span><span>1 → 3.2</span><span>2 → 7.0</span>
      </div>
      <p class="diagram-conclusion">
        {{
          stage === 2
            ? "Sort dictionary bit patterns and encode their differences. No quantization."
            : "Store each distinct value once, then encode its rank."
        }}
      </p>
    </template>
    <template v-else-if="kind === 'modes'">
      <small>SHARED PER-GENE MODES</small>
      <div class="array-row">
        <label>gene</label><span>A</span><span>B</span><span>C</span>
      </div>
      <div class="array-row">
        <label>mode</label><span>1.5</span><span>3.2</span><span>1.5</span>
      </div>
      <div class="array-row">
        <label>cell</label><span>{{ stage ? "rank 0" : "1.5" }}</span
        ><span>{{ stage ? "rank 1" : "7.0" }}</span
        ><span>{{ stage ? "rank 0" : "1.5" }}</span>
      </div>
      <div class="flow-arrow active">↓</div>
      <div class="dictionary">
        <span>{{
          stage === 2
            ? "Local exception dictionary: 1 → 7.0"
            : "Matches use the shared mode; only exceptions need local values."
        }}</span>
      </div>
    </template>
    <template v-else-if="kind === 'models'">
      <div class="shared-model">
        <small>CHUNK METADATA · shared rANS probability tables</small>
        <div><span>gaps</span><span>dictionary</span><span>ranks</span></div>
      </div>
      <div class="model-branches">↓ <span>↓</span> ↓</div>
      <div class="independent-records">
        <div
          v-for="i in 3"
          :key="i"
          :class="{
            target: stage > 0 && i === 2,
            muted: stage === 2 && i !== 2,
          }"
        >
          <b>cell {{ i }}</b
          ><span>local dictionary</span><span>coding states</span
          ><span>coded streams</span>
        </div>
      </div>
      <p class="diagram-conclusion">
        {{
          stage === 0
            ? "Estimate statistics across cells to amortize model cost."
            : stage === 1
              ? "Start a separate coding state for every record."
              : "Decode cell 2 with metadata. Cells 1 and 3 stay compressed."
        }}
      </p>
    </template>
    <template v-else-if="kind === 'pages'">
      <small>PHYSICAL PAGES · schematic sub-page records</small>
      <div class="page-demo" :class="{ packed: stage > 0 }">
        <div class="page-grid">
          <span>page 0</span><span>page 1</span><span>page 2</span>
        </div>
        <div class="packed-record r-a">A</div>
        <div class="packed-record r-b">B</div>
        <div class="packed-record r-c">C</div>
        <div class="packed-record r-d">D</div>
      </div>
      <div class="directory">
        <span>A → page 0</span
        ><span>B → {{ stage ? "page 1" : "pages 0 + 1" }}</span
        ><span>C → {{ stage ? "page 0" : "page 1" }}</span>
      </div>
      <p class="diagram-conclusion">
        {{
          stage === 0
            ? "Tight concatenation can split a small record across two pages."
            : stage === 1
              ? "Pack record remainders into pages; padding prevents avoidable crossings."
              : "The directory preserves cell ids. A record of s bytes touches ceil(s / P) pages."
        }}
      </p>
    </template>
    <template v-else-if="kind === 'access'">
      <div class="flow-label">
        <b>access(cell B)</b
        ><span class="access-token">directory → offset + length</span>
      </div>
      <div class="flow-arrow active">↓</div>
      <div class="record-strip access-records">
        <span
          v-for="v in ['A', 'B', 'C', 'D', 'E']"
          :key="v"
          :class="{ target: v === 'B', muted: stage > 0 && v !== 'B' }"
          >{{ v }}</span
        >
      </div>
      <div class="flow-arrow" :class="{ active: stage > 0 }">
        ↓ <span>fetch the page(s) containing B</span>
      </div>
      <div class="record-box" :class="{ active: stage === 2 }">
        <b>B + shared metadata</b><span>decode B only</span>
      </div>
      <p class="diagram-conclusion">
        No neighbouring records need decompression. Physical I/O remains
        page-granular.
      </p>
    </template>
  </div>
</template>
