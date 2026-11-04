// The homepage story: chapters of scroll-driven scenes and their copy.
import { withBase } from "vitepress";
import growth from "./scenes/growth.js";
import supply from "./scenes/supply.js";
import scale from "./scenes/scale.js";
import reads from "./scenes/reads.js";
import tradeoff from "./scenes/tradeoff.js";
import pipeline from "./scenes/pipeline.js";
import worker from "./scenes/worker.js";
import planContract from "./scenes/plan-contract.js";
import planCompile from "./scenes/plan-compile.js";
import planScale from "./scenes/plan-scale.js";
import planOnline from "./scenes/plan-online.js";
import race from "./scenes/race.js";
import billion from "./scenes/billion.js";
import { formatChapter } from "./format-story.js";

const data = (name) => withBase(`/research/home/${name}`);
const link = (name, label) => `<a href="${data(name)}">${label}</a>`;

export const chapters = [
  {
    id: "why",
    label: "Training demand",
    summary: "Scale & GPU demand",
    icon: "demand",
    title: "Billion-cell training needs a faster loader.",
    lede: "Single-cell models are moving to billion-cell corpora. Past the size of host memory, every cell must be streamed from storage — and the loader decides how fast training can go.",
    scenes: [
      {
        id: "growth",
        label: "Corpus scale",
        scene: growth,
        length: 380,
        beats: [
          { at: 0, kicker: "DEMAND · WHY NOW", title: "Single-cell pretraining <br>is heading to a billion cells.", body: "scGPT trained on 33M cells; Tahoe-x1 on 266M. scBaseCount holds 502M cells, and Arc Institute plans over a billion cell experiments for virtual-cell modelling." },
          { at: 0.24, kicker: "DEMAND · WHY NOW", title: "Below the memory line, <br>nothing looks slow.", body: "While the corpus fits in host memory, any loader can keep it in RAM. Every inefficiency stays hidden." },
          { at: 0.42, kicker: "DEMAND · WHY NOW", title: "Then it doesn't fit.", body: "At 16 KB per cell in CSR, 266M cells already need 4.26 TB — more than the 2 TB of host memory in a DGX H200. A billion cells need 16 TB." },
          { at: 0.78, kicker: "DEMAND · WHY NOW", title: "Training must stream <br>from storage.", body: "Every epoch reads the corpus in a fresh random order. The loader is now on the critical path of every training step." },
        ],
        source: "Corpus sizes as cited in the technical report (§1). CSR size at 2,000 nonzeros per cell with 32-bit values and indices.",
      },
      {
        id: "supply",
        label: "GPU demand & data supply",
        scene: supply,
        length: 460,
        beats: [
          { at: 0, kicker: "DEMAND · THE SUPPLY GAP", title: "How fast do models <br>consume cells?", body: "Measured training demand of eleven single-cell models on one H200 GPU. Each packet is a batch of cells on its way to the GPU." },
          { at: 0.25, kicker: "DEMAND · THE SUPPLY GAP", title: "Training stalls <br>on the supply side.", body: "The fastest existing random-access loader, Lance, delivers 79K cells/s. Past that line, packets stop arriving: seven of eleven models wait for data, CellFlow for 81% of the time." },
          { at: 0.44, kicker: "DEMAND · CELLPACK", title: "Close the <br>supply gap.", body: "Watch CellPack's supply line pass the models' demand. Each model stops waiting once its batches can arrive fast enough." },
          { at: 0.5, kicker: "DEMAND · CELLPACK", title: "CellPack: <br>1.17M cells/s.", body: "A storage format and loader built for this problem. On the same single NVMe SSD and 16 CPU cores, it feeds every model at its full single-GPU rate." },
          { at: 0.7, kicker: "DEMAND · CELLPACK", title: "Add storage <br>bandwidth.", body: "Scale from one NVMe SSD and 16 workers to two SSDs in RAID-0 and 32 workers." },
          { at: 0.74, kicker: "DEMAND · CELLPACK", title: "2.52M cells/s <br>with more hardware.", body: "Two NVMe SSDs in RAID-0 and 32 workers." },
          { at: 0.86, kicker: "DEMAND · CELLPACK", title: "Billion-cell training, <br>fed continuously.", body: "Persistent, efficient loading at billion-cell scale — so data supply is no longer the bottleneck for foundation models, perturbation models and virtual-cell research." },
        ],
        source: `Model demand: synthetic GPU-resident inputs, one H200. Loaders: best variant each, one NVMe SSD, 16 workers, 20 GB cache, random order (report Figs. 1, 5). GPU idle = 1 − supply / demand. ${link("model_compute.csv", "Models")} · ${link("loader_methods.csv", "Loaders")} · ${link("cellpack_paper_values.json", "CellPack")}`,
      },
    ],
  },
  {
    id: "hard",
    label: "Storage trade-offs",
    summary: "Footprint & random reads",
    icon: "storage",
    title: "Why storage becomes the bottleneck.",
    lede: "A billion cells must be small enough to store and fast enough to read one at a time, in any order. Existing formats give up one to get the other.",
    scenes: [
      {
        id: "scale",
        label: "Dense, CSR & compressed storage",
        scene: scale,
        length: 780,
        beats: [
          { at: 0, kicker: "STORAGE · HOW BIG IS IT", title: "One cell is a row <br>of 40,000 genes.", body: "A cell-by-gene matrix stores one expression value per gene. As dense float32, one cell is 160 KB." },
          { at: 0.09, kicker: "STORAGE · HOW BIG IS IT", title: "A dataset: <br>100,000 cells.", body: "Stack the cells and a typical dataset is 16 GB of dense matrix." },
          { at: 0.23, kicker: "STORAGE · HOW BIG IS IT", title: "A corpus: <br>a billion cells.", body: "A billion-cell training corpus is 160 TB as a dense matrix." },
          { at: 0.41, kicker: "STORAGE · HOW BIG IS IT", title: "Most of it is zero.", body: "Keep only nonzero values and their gene ids. At ~2,000 nonzeros per cell: 16 KB per cell, 16 TB in total." },
          { at: 0.51, kicker: "STORAGE · HOW BIG IS IT", title: "Pack each row.", body: "Each cell's nonzeros move to the start of its row as (value, gene id) pairs. Rows still have different lengths." },
          { at: 0.62, kicker: "STORAGE · HOW BIG IS IT", title: "Concatenate: <br>CSR's three arrays.", body: "All values go into one array, all gene ids into another, and indptr records where each cell starts. A cell is now a slice of two arrays." },
          { at: 0.75, kicker: "STORAGE · HOW BIG IS IT", title: "Formats chunk and <br>compress the arrays.", body: "HDF5 and Zarr split each array into chunks and compress every chunk on its own. With Zarr's PCodec, 16 TB becomes about 3.3 TB." },
          { at: 0.86, kicker: "STORAGE · HOW BIG IS IT", title: "Now read <br>one random cell.", body: "Its slices sit inside compressed chunks of both arrays. To get one cell, the loader fetches and decompresses every chunk it touches." },
        ],
        source: "Illustration: 10⁹ cells × 40,000 genes × 4 bytes; 2,000 nonzeros per cell; Zarr (PCodec-8) at its pooled 4.79× ratio from the report's compression table. Chunk sizes are schematic.",
      },
      {
        id: "reads",
        label: "Read amplification",
        scene: reads,
        length: 400,
        beats: [
          { at: 0, kicker: "STORAGE · EXISTING LOADERS", title: "Each format stores cells <br>its own way.", body: "Top: bytes each loader stores per cell. Bottom: bytes it reads from storage to deliver one random cell. One square is a 4 KiB page." },
          { at: 0.36, kicker: "STORAGE · EXISTING LOADERS", title: "Compact formats <br>read the most.", body: "Zarr and Parquet store a cell in 5–7 KB, yet read 450–1,150 KB per random cell: whole compressed chunks and row groups, mostly other cells." },
          { at: 0.6, kicker: "STORAGE · EXISTING LOADERS", title: "Uncompressed formats <br>move more bytes.", body: "memmap and H5AD skip decoding but store ~20 KB per cell, and page reads still pull in neighbours. Lance comes closest: 12 KB stored, 29 KB read." },
          { at: 0.78, kicker: "STORAGE · EXISTING LOADERS", title: "CellPack: <br>2 KB stored, 4.4 KB read.", body: "Smaller than every format here, and about one page of device traffic per random cell. The next chapter shows how." },
        ],
        source: `Device reads per delivered cell at each loader's smallest runnable cache budget, mean of three seeds; stored bytes = corpus size / 10M cells (report Fig. 2a). ${link("read_amplification.csv", "Data")}`,
      },
      {
        id: "tradeoff",
        label: "Compression vs. random access",
        scene: tradeoff,
        length: 420,
        beats: [
          { at: 0, kicker: "STORAGE · THE TRADE-OFF", title: "Compressors learn <br>from context.", body: "A codec finds redundancy across the bytes it sees together. Put more cells in one compressed unit and the ratio rises; at one cell per unit, general-purpose codecs barely compress." },
          { at: 0.4, kicker: "STORAGE · THE TRADE-OFF", title: "But a random read <br>pays for the whole unit.", body: "Reading one cell decodes its unit: about k / R(k) cells' worth of bytes. At 2,048 cells per unit, XZ-9 reads over 400 cells' worth to return one." },
          { at: 0.7, kicker: "STORAGE · THE TRADE-OFF", title: "CellPack: 5.89× <br>at one cell per decode.", body: "Separate the unit of statistics from the unit of decoding: share models across a 4,096-cell chunk, keep every cell independently decodable." },
        ],
        source: `Pooled mean ratio of raw and normalized CELLxGENE matrices by cells per independently compressed unit (report Fig. 2b); read cost = k / R(k), before page and metadata overhead (report §2.2). ${link("compression_granularity.csv", "Data")}`,
      },
    ],
  },
  formatChapter,
  {
    id: "loader",
    label: "Loader",
    summary: "Planning & async I/O",
    icon: "loader",
    title: "From cell IDs to training batches.",
    lede: "A format is half the answer. The loader turns an access sequence into model-ready batches: it plans the cache ahead, reads asynchronously, decodes in parallel and delivers in order.",
    scenes: [
      {
        id: "pipeline",
        label: "Loading pipeline",
        scene: pipeline,
        length: 440,
        beats: [
          { at: 0, kicker: "LOADER · THE LOADER", title: "Follow one access.", body: "Training already knows its order: shuffled permutations, epoch after epoch. The loader receives that sequence up front. Watch one cell travel through it." },
          { at: 0.18, kicker: "LOADER · PLAN", title: "A planner looks ahead.", body: "An online compiler scans upcoming accesses, finds pages and decoded rows that will be reused within the cache budget, assigns them frames and publishes plan entries ahead of the workers." },
          { at: 0.36, kicker: "LOADER · WORKERS", title: "Workers claim positions.", body: "Each worker claims the next position and follows its plan entry: use a cached frame, admit a page for later reuse, or read a miss." },
          { at: 0.52, kicker: "LOADER · I/O", title: "Reads never block.", body: "Page reads go through io_uring. A worker submits, moves on to other work, and harvests completions later." },
          { at: 0.66, kicker: "LOADER · DECODE", title: "Decode, map, <br>convert, process.", body: "A record decodes on its own; genes are mapped to the model's vocabulary; values are converted; an optional native function builds model inputs." },
          { at: 0.8, kicker: "LOADER · OUTPUT", title: "Delivered in order.", body: "Workers finish out of order, but the output ring publishes slots in the requested order. Training leases a slot without copying and releases it when done." },
        ],
        source: "Conceptual diagram of the loader's modules; not to scale.",
      },
      {
        id: "plan-contract",
        label: "Cache plan & worker contract",
        scene: planContract,
        length: 560,
        beats: [
          { at: 0, kicker: "LOADER · CACHE POLICY", title: "A runtime cache <br>serialises the workers.", body: "A shared LRU cache takes a lock on every access: look the page up, move it to the front, evict the tail. While one worker holds the lock, every other worker waits in line." },
          { at: 0.18, kicker: "LOADER · CACHE POLICY", title: "Guessing costs twice.", body: "Shuffled order makes history a poor guide: at a 10% budget LRU removes only 9.1% of page misses. Meanwhile the lock idles the SSD. The loader drops to 0.80M cells/s and 3.2 GB/s of reads, against 1.45M cells/s and 6.4 GB/s with no cache at all." },
          { at: 0.4, kicker: "LOADER · THE PLAN", title: "Decide ahead, <br>off the hot path.", body: "Training already knows its order. A planner on its own cores reads it ahead and turns it into one entry per position. Workers follow their entries; they never consult or update a shared cache." },
          { at: 0.6, kicker: "LOADER · THE PLAN", title: "A plan is a contract.", body: "Each entry fixes the frame, whether to admit or use, and the generation to expect. A use may run before the admitting read lands, so it waits for that generation. A frame takes a new page only after D quiet positions, when no reader of the old one can still be running." },
          { at: 0.8, kicker: "LOADER · THE PLAN", title: "One read, <br>every later cell.", body: "The admitting access reads page 7 from the SSD once. The later cells on that page find it in f7 and skip the device. No worker asks whether a page is cached: its entry already says so." },
        ],
        source: `Simulated schedule of four workers and one frame (D = 2). LRU and no-cache figures: report Fig. 6 at a 10% budget, 32 workers, one NVMe SSD, three-run means. ${link("cache_policy_panel2048_runs.csv", "Measurements")}`,
      },
      {
        id: "plan-compile",
        label: "Offline cache planning",
        scene: planCompile,
        length: 820,
        beats: [
          { at: 0, kicker: "LOADER · COMPILING A PLAN", title: "Every reuse <br>is an interval.", body: "Two consecutive uses of a page, at positions u and v, form a reuse interval. Keeping the page in a frame from u to v saves the read at v." },
          { at: 0.12, kicker: "LOADER · COMPILING A PLAN", title: "Belady: <br>the optimal baseline.", body: "Sweep the sequence and decide each interval at its second use v: keep it if a frame is free over the whole span, otherwise read the page again. For equal pages served one at a time, this is Belady's optimum with bypass: 9 reads here." },
          { at: 0.33, kicker: "LOADER · WHY NOT BELADY", title: "Workers finish <br>out of order.", body: "Belady assumes one access after another. CellPack's workers overlap and complete out of order: position 7 may admit F into f1 while position 6 is still decoding C from it. The new page would be swapped in before the old one is swapped out." },
          { at: 0.49, kicker: "LOADER · WHY NOT BELADY", title: "So execution <br>stays in a window.", body: "A worker starts position u only after every position at or before u − D has finished, so in-flight positions never drift more than D apart. A frame last used by then is guaranteed free: eviction precedes admission. The bound turns a run-time race into a rule the compiler checks ahead, with no reader counts or locks while loading." },
          { at: 0.61, kicker: "LOADER · WHY NOT BELADY", title: "An approximate <br>Belady.", body: "The compiler keeps Belady's rule but reserves each fresh admission's frame from u − D. Here F no longer fits: 10 reads against Belady's 9. Those reserved frames are the price of running out of order safely." },
          { at: 0.72, kicker: "LOADER · OFFLINE COMPILATION", title: "Compiling offline.", body: "With the whole sequence stored, the compiler collects every reuse interval and sweeps them by right end. A range-max/range-add tree over positions answers whether frames are free over [u − D, v]. Page groups compile in parallel shards, and the result does not depend on scheduling." },
          { at: 0.82, kicker: "LOADER · OFFLINE COMPILATION", title: "Then frames <br>get names.", body: "Residencies take frames in admission order, and each new tenant raises its frame's generation. Every position receives its entry: admit, use or read. The plan is bound to that exact sequence and replayed once to verify it." },
          { at: 0.9, kicker: "LOADER · CACHE POLICY", title: "0.3 points <br>from the optimum.", body: "At a 10% budget the plan removes 38.4% of page misses, against 38.7% for simulated Belady and 23.5% for the best classic policy. Throughput rises to 2.29M cells/s, from 1.45M without a cache." },
        ],
        source: `Toy sequence: 16 accesses, 2 frames, window D = 2, decided with the compiler's page rule; the worker timing is illustrative. Results: report Fig. 6, three-run means; Belady from the report's same-protocol simulation. ${link("cache_policy_panel2048_runs.csv", "Measurements")} · ${link("cache_policy_sim.csv", "Simulation")}`,
      },
      {
        id: "plan-scale",
        label: "Planning at corpus scale",
        scene: planScale,
        length: 500,
        beats: [
          { at: 0, kicker: "LOADER · WHY ONLINE", title: "A sequence the size <br>of training.", body: "An offline plan needs the whole access sequence in hand. For one billion cells and five epochs that is five billion cell ids: 20 GB before compiling anything." },
          { at: 0.22, kicker: "LOADER · WHY ONLINE", title: "Sets multiply it.", body: "Set-based models such as STATE consume cells in sets of 512. When every cell starts a set each epoch, five epochs need 5 × 512 × 1B = 2.56T accesses: 10.2 TB of ids, 5.5× the compressed corpus and 5× host memory, before a single plan entry." },
          { at: 0.54, kicker: "LOADER · WHY ONLINE", title: "Offline: store it all, <br>then sweep.", body: "An offline plan is compiled before loading starts, so the whole sequence and every plan entry sit in memory while the workers sweep across them, and stay there after the sweep has passed. At billion-cell scale with sets of 512, that is 10.2 TB of ids alone." },
          { at: 0.76, kicker: "LOADER · WHY ONLINE", title: "Online: a planner <br>runs ahead.", body: "Online, a planner works one horizon ahead of the workers. It generates each id from the seed only when it reaches it, compiles the entries, and the ids are dropped as soon as the workers have passed. Memory follows H, not the length of training: the billion-cell run looked one billion accesses ahead on eight planner cores and delivered its first batch 106 s after start." },
        ],
        source: "Sequence sizes at 4 bytes per cell id. Corpus size, horizon, planner cores and first-batch time: report, billion-cell run. Host memory: DGX H200, as in chapter I.",
      },
      {
        id: "plan-online",
        label: "Online cache planning",
        scene: planOnline,
        length: 600,
        beats: [
          { at: 0, kicker: "LOADER · ONLINE PLANNING", title: "Choose a horizon.", body: "The online compiler decides reuse only within H accesses of its frontier. A longer horizon keeps more reuse, but holds more of the sequence in memory and delays the first batch until H accesses are compiled. The billion-cell run used one epoch, so pages read in one epoch can be kept for the next." },
          { at: 0.25, kicker: "LOADER · ONLINE PLANNING", title: "Decide within <br>the horizon.", body: "At its frontier X, the compiler decides each interval that has just closed, if it is at most H long. It tracks only how many frames are in use; which frame comes later." },
          { at: 0.42, kicker: "LOADER · ONLINE PLANNING", title: "Freeze, <br>then publish.", body: "At Z = X − H, frames get names and generations. Entries are published at P = Z − D: a later admission reserves its frame from u − D ≥ P, so no published entry ever changes. Workers consume entries at F." },
          { at: 0.59, kicker: "LOADER · ONLINE PLANNING", title: "Reuse beyond <br>the horizon is lost.", body: "An interval longer than H closes after its start has frozen, so its page is read again. In the report's runs this costs nothing measurable: at a 10% budget the online plan removes 38.5% of page misses, the offline plan 38.4%." },
          { at: 0.78, kicker: "LOADER · ONLINE PLANNING", title: "Workers never wait <br>for the plan.", body: "When workers catch up with the published frontier, the compiler freezes early, trading look-ahead for latency. It never freezes past X − D − C, its chunk size C behind, so every published decision stays valid." },
        ],
        source: `Synthetic stream, 3 frames, D = 3, H = 12, decided with the compiler's page rule. Online and offline figures: report Fig. 6, 10% budget. ${link("cache_policy_panel2048_runs.csv", "Measurements")}`,
      },
      {
        id: "worker",
        label: "Asynchronous workers",
        scene: worker,
        length: 520,
        beats: [
          { at: 0, kicker: "LOADER · ONE WORKER", title: "One coroutine <br>per position.", body: "A worker runs many lightweight coroutines, one per claimed position. It never blocks on the disk." },
          { at: 0.12, kicker: "LOADER · ONE WORKER", title: "Cached? <br>Run it through.", body: "Claim, look up the plan entry, decode the record, map genes, run the process function, write the slot." },
          { at: 0.3, kicker: "LOADER · ONE WORKER", title: "Not cached? <br>Submit, park, move on.", body: "The coroutine puts its read on io_uring and parks. The worker claims the next position right away." },
          { at: 0.52, kicker: "LOADER · ONE WORKER", title: "Poll at <br>fixed check points.", body: "After each claim and between stages, the worker checks its completion queue. Landed reads resume their coroutines." },
          { at: 0.78, kicker: "LOADER · ONE WORKER", title: "Never idle.", body: "Reads stay in flight while the CPU decodes other cells. The timeline below fills up with work." },
        ],
        source: "Illustrative schedule with typical latencies (≈180 µs per read, ≈15 µs per decode, 65% cache hits); not a measured trace.",
      },
    ],
  },
  {
    id: "practice",
    label: "In practice",
    summary: "Benchmarks at scale",
    icon: "practice",
    title: "Performance in practice.",
    lede: "Measured on real single-cell data — then on a billion cells, for five full epochs.",
    scenes: [
      {
        id: "race",
        label: "Loader throughput",
        scene: race,
        length: 340,
        beats: [
          { at: 0, kicker: "BENCHMARKS · THROUGHPUT", title: "Loaders on <br>the same machine.", body: "Cells per second delivered from CELLxGENE data in random order, on a linear scale." },
          { at: 0.5, kicker: "BENCHMARKS · THROUGHPUT", title: "1.17M cells/s from <br>one NVMe SSD.", body: "With 16 workers and no data cache. 2.52M cells/s with two SSDs in RAID-0 and 32 workers." },
          { at: 0.72, kicker: "BENCHMARKS · THROUGHPUT", title: "14.7× the best <br>existing loader.", body: "Same SSD, same 16 CPU cores." },
        ],
        source: `Baselines: best variant per loader, one NVMe SSD, 16 workers, 20 GB cache, random order, mean over seeds (report Fig. 5). ${link("loader_methods.csv", "Loader data")} · ${link("cellpack_paper_values.json", "CellPack values")}`,
      },
      {
        id: "billion",
        label: "Billion-cell training",
        scene: billion,
        length: 480,
        layout: "top",
        beats: [
          { at: 0, kicker: "BENCHMARKS · ONE BILLION CELLS", title: "Five epochs over <br>a billion cells.", body: "Two NVMe SSDs in RAID-0, a 1 TB cache and a plan computed online across the full billion-cell horizon. Replayed from the run's per-second log." },
          { at: 0.3, kicker: "BENCHMARKS · ONE BILLION CELLS", title: "Epoch one <br>fills the cache.", body: "Pages the plan admits are kept for the reuse it can see coming in later epochs. Teal packets come from cache, orange from SSD." },
          { at: 0.55, kicker: "BENCHMARKS · ONE BILLION CELLS", title: "Then it holds <br>3.6M cells/s.", body: "Epochs two to five each take about four and a half minutes, at roughly 81% page hits." },
          { at: 0.9, kicker: "BENCHMARKS · ONE BILLION CELLS", title: "Less waiting. <br>More discovery.", body: "Five billion cells delivered in under 26 minutes." },
        ],
        source: `Per-second curves smoothed over 5 s; throughput normalized to its 5 s peak (4.0M cells/s), reads to the array's page-read ceiling, memory to the 1.89 TB limit (report Fig. 7). ${link("billion_cell_online.csv", "Data")}`,
      },
    ],
  },
];
