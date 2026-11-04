// The format chapter: from native CSR to a CellPack file, one design step
// per scene, with byte sizes measured on a sample of CELLxGENE chunks.
import { withBase } from "vitepress";
import frecord from "./scenes/frecord.js";
import findices from "./scenes/findices.js";
import fvalues from "./scenes/fvalues.js";
import fshared from "./scenes/fshared.js";
import fpages from "./scenes/fpages.js";
import ffile from "./scenes/ffile.js";
import { B, CSR, FINAL, SAMPLE, STEPS } from "./scenes/fmt-common.js";

const kb = (v) => `${(v / 1000).toFixed(v >= 10000 ? 1 : 2)} KB`;
const pct = (v) => `${Math.round(v * 100)}%`;
const K = STEPS.contrast;
const measured = `Measured on ${SAMPLE} (raw counts and normalized, 4,096-cell chunks) by re-encoding with the current writer. <a href="${withBase("/research/home/format_steps.json")}">Data</a>`;

export const formatChapter = {
  id: "format",
  label: "Format",
  summary: "Records, models & pages",
  icon: "format",
  title: "Compact records. Independent decoding.",
  lede: `Start from native CSR and change one thing at a time, each step fitted to how single-cell data looks. Measured end to end: ${kb(CSR)} per cell becomes ${kb(FINAL)} — ${(CSR / FINAL).toFixed(2)}× smaller — and every cell still decodes on its own.`,
  scenes: [
    {
      id: "format-record",
      label: "Cell records",
      scene: frecord,
      length: 460,
      beats: [
        { at: 0, kicker: "FORMAT · STEP 1", title: "Start from <br>native CSR.", body: `indptr locates a cell's slice in two separate arrays. With 64-bit offsets, 32-bit gene ids and 32-bit values, a cell in our sample takes ${kb(CSR)}.` },
        { at: 0.08, kicker: "FORMAT · STEP 1", title: "Each cell is <br>two slices.", body: "indptr cuts the indices array and the values array into one slice per cell." },
        { at: 0.28, kicker: "FORMAT · STEP 1", title: "Pack each cell <br>into one record.", body: "A cell's ids and values move together behind a 3-byte header (flags + nnz). One cell, one contiguous record." },
          { at: 0.52, kicker: "FORMAT · STEP 1", title: "Narrow the types.", body: `Up to 65,536 genes, a 2-byte id is enough, and raw counts keep their 16-bit type: ${kb(B["2_records_raw"])} per cell.` },
          { at: 0.68, kicker: "FORMAT · STEP 1", title: "Records make <br>a chunk.", body: "Records sit end to end in a chunk; an offset and length per record replace indptr. Every later step compresses records, but never makes one depend on another." },
          { at: 0.85, kicker: "FORMAT · STEP 1", title: "Read just <br>one record.", body: "Look up a cell's offset and length, then read its contiguous record. Neighbouring cells do not have to be decoded." },
      ],
      source: measured,
    },
    {
      id: "format-indices",
      label: "Gene index coding",
      scene: findices,
      length: 440,
      beats: [
        { at: 0, kicker: "FORMAT · STEP 2 · GENE IDS", title: "Store gaps, <br>not gene ids.", body: `Within a record, ids are sorted; store the difference to the previous one. A shared rANS model codes the gaps in ${STEPS.index_bits.source.toFixed(1)} bits instead of 16.` },
        { at: 0.32, kicker: "FORMAT · STEP 2 · GENE IDS", title: "Renumber genes <br>by frequency.", body: "Each chunk ranks genes by how many of its cells express them. A cell's genes are mostly common ones, so their new ids sit close together." },
        { at: 0.62, kicker: "FORMAT · STEP 2 · GENE IDS", title: `${pct(STEPS.zero_gaps.renumbered)} of gaps <br>become zero.`, body: `Up from ${pct(STEPS.zero_gaps.source_order)} in source order. Gene ids drop to ${STEPS.index_bits.renumbered.toFixed(2)} bits each; one permutation per chunk restores the original ids.` },
      ],
      source: measured,
    },
    {
      id: "format-values",
      label: "Value coding",
      scene: fvalues,
      length: 460,
      beats: [
        { at: 0, kicker: "FORMAT · STEP 3 · VALUES", title: "Values repeat.", body: "Expression comes from small integer counts, so even normalized values take only a handful of distinct bit patterns per cell." },
        { at: 0.1, kicker: "FORMAT · STEP 3 · VALUES", title: "Store each value once.", body: "The record keeps a small dictionary of its distinct bit patterns, exactly — nothing is rounded — and replaces every value with a rank." },
        { at: 0.38, kicker: "FORMAT · STEP 3 · VALUES", title: "Entropy-code <br>the ranks.", body: "Ranks are skewed: the common ones cost a fraction of a bit, rare ones a few bits. A shared rANS model codes them close to that ideal length." },
        { at: 0.66, kicker: "FORMAT · STEP 3 · VALUES", title: "The largest saving <br>in the format.", body: `${STEPS.value_bits.raw_u16.toFixed(1)}–${STEPS.value_bits.normed_f32.toFixed(1)} bits per value instead of 16 or 32: a cell goes from ${kb(B["4_gap_renumbered"])} to ${kb(B["5_values_dictionary"])}. The writer can also code values against a per-gene mode, where a chunk benefits.` },
      ],
      source: measured,
    },
    {
      id: "format-shared",
      label: "Shared probability models",
      scene: fshared,
      length: 420,
      beats: [
        { at: 0, kicker: "FORMAT · STEP 4 · SHARED MODELS", title: "Statistics are shared. <br>Decoding is not.", body: "The chunk stores probability tables once. Each record keeps its own coding state and value dictionary: shared tables decode the symbols, then its own dictionary maps ranks to values. No neighbours are decoded." },
        { at: 0.36, kicker: "FORMAT · STEP 4 · SHARED MODELS", title: "Why not compress <br>each cell alone?", body: `Fitting a model to each cell loses context: ${Math.round((K.per_cell_rans / K.shared - 1) * 100)}% larger even if its tables were free, ${(K.per_cell_rans_with_tables / K.shared).toFixed(1)}× with them. zstd on each cell is ${(K.zstd3_per_cell / K.shared).toFixed(1)}× larger.` },
        { at: 0.72, kicker: "FORMAT · STEP 4 · SHARED MODELS", title: "Metadata is <br>a rounding error.", body: `Tables, gene permutation, offsets and nnz totals add ${STEPS.metadata_per_cell.toFixed(0)} bytes per cell — under 2% of the chunk.` },
      ],
      source: measured,
    },
    {
      id: "format-pages",
      label: "Page packing",
      scene: fpages,
      length: 440,
      beats: [
        { at: 0, kicker: "FORMAT · STEP 5 · PAGES", title: "Storage reads pages, <br>not bytes.", body: "Laid end to end, a small record can still cross a 4 KiB page boundary — and cost two reads." },
        { at: 0.24, kicker: "FORMAT · STEP 5 · PAGES", title: "Pack records <br>into pages.", body: `Records are packed so each fits inside one page; the gaps are padding. Measured cost: ${Math.round(B["8_packed_4k_pages"] - B["7_plus_chunk_metadata"])} bytes per cell.` },
        { at: 0.5, kicker: "FORMAT · STEP 5 · PAGES", title: "A directory keeps <br>rows in order.", body: "Records may move physically; the offset table maps each row to its page and position." },
        { at: 0.7, kicker: "FORMAT · STEP 5 · PAGES", title: "One page, <br>one cell.", body: `A random read fetches one page and decodes one record. On disk, ${kb(FINAL)} per cell: ${(CSR / FINAL).toFixed(2)}× smaller than CSR.` },
      ],
      source: `${measured}. Record sizes in the animation are illustrative.`,
    },
    {
      id: "format-file",
      label: "File layout & AnnData",
      scene: ffile,
      length: 440,
      beats: [
        { at: 0, kicker: "FORMAT · THE FILE", title: "The .cpk file.", body: "One matrix as one stream: a 4 KiB superblock, page-aligned chunks of 4,096 cells, and a directory at the end." },
        { at: 0.14, kicker: "FORMAT · THE FILE", title: "Inside a chunk.", body: "A 64-byte header, the shared models, the gene permutation, the offset tables — then records packed into 4 KiB pages." },
        { at: 0.42, kicker: "FORMAT · THE FILE", title: "AnnData, as you know it.", body: "The matrix X sits in the middle; obs, var, obsm and varm annotate its rows and columns. Usually X is stored as CSR." },
        { at: 0.6, kicker: "FORMAT · THE FILE", title: "Swap only X.", body: "CellPack is a general two-dimensional matrix format. An H5CP file is an H5AD file whose X is an embedded .cpk stream; everything else keeps its AnnData encoding." },
      ],
      source: "Layout from the format specification. pycellpack writes H5CP (write_h5cp) and reads it lazily (read_h5).",
    },
  ],
};
