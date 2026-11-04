//! Chunk models: the scan pass (validation, record shapes, gene presence,
//! column-mode votes) and the prepare pass (each row's record candidates,
//! whose symbols are counted for the models), and the column-mode decision.
//!
//! Symbols are counted for the layout and value coding each row is expected
//! to use. The expected coding depends on whether the chunk keeps column
//! modes, which only the whole chunk decides; the counts of rows preferring
//! modes are therefore kept for both codings until the decision.

use super::input::{Index, any_zero, load, store, with_indices, with_width, zeros};
use super::prep::{Arena, CodingPrep, Dense, Hist, SparsePrep, ValuesPrep, ValuesWs, order, prepare_sparse, prepare_values};
use super::scratch::{PrepWs, RowWs};
use super::{check_cell_exact, dense_candidate, expect_dense};
use crate::error::{Error, Result};
use crate::format::{DICT, GAP, N_MODELS, RANK};
use crate::scans::tables;
use crate::scans::types::{CellRef, EncodeOptions, Indices, LayoutChoice, MatrixRef, RowRef, Rows, ValueChoice};
use crate::unchecked::{get, set};

/// What the scan learned about one row.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RowInfo {
    /// Elements the row's records count as non-zeros.
    pub(super) nnz: usize,
    /// Elements its sparse record holds.
    pub(super) stored: usize,
    /// Whether it may be stored dense.
    pub(super) allowed: bool,
    /// Whether its symbols are counted for the dense layout.
    pub(super) dense: bool,
}

/// Statistics of a selection and the decisions taken from them.
pub(super) struct Model {
    /// Source gene of each coded id; `None` without renumbering.
    pub(super) source_of_coded: Option<Vec<u32>>,
    /// Most frequent value of each source gene, when the chunk keeps them.
    pub(super) modes: Option<Vec<u64>>,
    /// Normalized frequency table of each model.
    pub(super) freqs: [[u16; 256]; N_MODELS],
    /// Largest non-zero count of a record.
    pub(super) max_nnz: usize,
    /// Largest dictionary of any row (of the mode coding, when it is kept).
    pub(super) max_k: usize,
    /// Largest element count of a record: every gene of a dense one.
    pub(super) max_elems: usize,
    /// Shape and expected coding of each selected row.
    pub(super) rows: Vec<RowInfo>,
    /// Each row's prepared sparse record, when kept for coding.
    pub(super) preps: Vec<SparsePrep>,
}

impl Model {
    /// Non-zero count of each row, in selection order.
    pub(super) fn nnz(&self) -> impl Iterator<Item = u64> + '_ {
        self.rows.iter().map(|r| r.nnz as u64)
    }
}

/// Result of the scan pass.
pub(super) struct Scan {
    /// Shape and expected coding of each row.
    pub(super) rows: Vec<RowInfo>,
    /// Occurrences of each gene, when the chunk renumbers genes.
    presence: Option<Vec<u32>>,
    /// Column-mode votes ([`Votes`]).
    vote: Option<Vec<u64>>,
    /// Largest non-zero count of a record.
    pub(super) max_nnz: usize,
    /// Largest element count of a record, which sizes the row scratch.
    pub(super) max_elems: usize,
}

/// Mutable borrows of the tables the scan pass fills, for the columns the
/// chunk keeps.
struct ScanCtx<'a> {
    n_genes: u32,
    layout: LayoutChoice,
    presence: Option<&'a mut [u32]>,
    vote: Option<&'a mut [u64]>,
}

/// Boyer-Moore majority votes per gene: candidate and count side by side,
/// one word per gene for values of up to 4 bytes (`count << 32 | candidate`)
/// and two for wider ones.
struct Votes;

impl Votes {
    /// Words of the vote table per gene.
    fn words(vw: usize) -> usize {
        if vw <= 4 { 1 } else { 2 }
    }

    /// The candidate of each gene.
    fn candidates(votes: &[u64], vw: usize) -> Vec<u64> {
        if vw <= 4 {
            votes.iter().map(|&e| e & 0xffff_ffff).collect()
        } else {
            votes.chunks_exact(2).map(|e| e[0]).collect()
        }
    }
}

impl ScanCtx<'_> {
    /// One vote of `gene` for the `W`-byte value `v` (Boyer-Moore
    /// majority), without data-dependent branches.
    #[inline(always)]
    fn cast<const W: usize>(votes: &mut [u64], gene: usize, v: u64) {
        // SAFETY: callers pass genes below `n_genes`; the table holds
        // `Votes::words(W)` words per gene.
        unsafe {
            let (c, held) = if W <= 4 {
                let e = get(votes, gene);
                ((e >> 32) as u32, e & 0xffff_ffff)
            } else {
                (get(votes, 2 * gene + 1) as u32, get(votes, 2 * gene))
            };
            // Selections by masks: the outcome of a vote is data.
            let empty = ((c == 0) as u32).wrapping_neg();
            let same = (held == v) as u32;
            let held = held ^ ((held ^ v) & (empty as i32 as i64 as u64));
            let c = (c.wrapping_add(2 * same).wrapping_sub(1) & !empty) | (1 & empty);
            if W <= 4 {
                set(votes, gene, (c as u64) << 32 | held);
            } else {
                set(votes, 2 * gene, held);
                set(votes, 2 * gene + 1, c as u64);
            }
        }
    }
}

/// Validation, shape, presence and votes of a sparse row.
fn scan_sparse<I: Index, const W: usize>(idx: &[I], values: &[u8], cx: &mut ScanCtx) -> Result<RowInfo> {
    let n = idx.len();
    if values.len() != n * W {
        return Err(Error::InvalidInput("values length differs from nnz * value width"));
    }
    let g = cx.n_genes;
    let (nnz, allowed) = match cx.layout {
        LayoutChoice::Dense => {
            let z = zeros::<W>(values);
            (n - z, n > z)
        }
        choice => (n, dense_candidate(choice, n, g) && !any_zero::<W>(values)),
    };
    let dense = expect_dense(cx.layout, allowed, nnz, g);
    // Validation runs alongside presence and votes, whose genes are clamped
    // into the tables until the row proves valid.
    let last = g as usize - 1;
    let fused = nnz == n && !dense;
    let mut ok = true;
    let mut prev = -1i64;
    // SAFETY: clamped genes are below `n_genes`, the length of the
    // presence and vote tables; `t < n`, a value of the row.
    unsafe {
        macro_rules! pass {
            ($body:expr) => {
                for (t, &i) in idx.iter().enumerate() {
                    let s = i.id() as i64;
                    ok &= i.valid() & (s > prev);
                    prev = s;
                    let gene = (s as usize).min(last);
                    #[allow(clippy::redundant_closure_call)]
                    ($body)(t, gene);
                }
            };
        }
        match (fused, cx.presence.as_deref_mut(), cx.vote.as_mut()) {
            (true, Some(p), Some(votes)) => pass!(|t, gene| {
                *p.get_unchecked_mut(gene) += 1;
                ScanCtx::cast::<W>(votes, gene, load::<W>(values, t));
            }),
            (true, Some(p), None) => pass!(|_, gene| *p.get_unchecked_mut(gene) += 1),
            (true, None, Some(votes)) => pass!(|t, gene| ScanCtx::cast::<W>(votes, gene, load::<W>(values, t))),
            _ => pass!(|_, _| ()),
        }
    }
    if !ok || prev >= g as i64 {
        check_cell_exact(idx, g)?;
    }
    let info = RowInfo { nnz, stored: n, allowed, dense };
    if fused || nnz == 0 {
        return Ok(info);
    }
    // SAFETY: validated genes are below `n_genes`, the table lengths;
    // `nnz <= n`.
    unsafe {
        if let Some(p) = cx.presence.as_deref_mut() {
            for &i in idx.get_unchecked(..nnz) {
                *p.get_unchecked_mut(i.id() as usize) += 1;
            }
        }
        if let Some(votes) = cx.vote.as_deref_mut() {
            if dense {
                // Every gene votes, absent ones for zero.
                let mut t = 0;
                for gene in 0..g as usize {
                    let v = if t < n && get(idx, t).id() as usize == gene {
                        t += 1;
                        load::<W>(values, t - 1)
                    } else {
                        0
                    };
                    ScanCtx::cast::<W>(votes, gene, v);
                }
            } else {
                for t in 0..nnz {
                    ScanCtx::cast::<W>(votes, get(idx, t).id() as usize, load::<W>(values, t));
                }
            }
        }
    }
    Ok(info)
}

/// Shape, presence and votes of a dense row.
fn scan_dense<const W: usize>(values: &[u8], cx: &mut ScanCtx) -> Result<RowInfo> {
    let g = cx.n_genes;
    if values.len() != g as usize * W {
        return Err(Error::InvalidInput("dense row length differs from n_genes * value width"));
    }
    let nnz = g as usize - zeros::<W>(values);
    let allowed = nnz > 0 && dense_candidate(cx.layout, nnz, g);
    let dense = expect_dense(cx.layout, allowed, nnz, g);
    let info = RowInfo { nnz, stored: nnz, allowed, dense };
    if nnz == 0 {
        return Ok(info);
    }
    for gene in 0..g as usize {
        // SAFETY: `gene < n_genes`, within the row and both tables.
        let v = unsafe { load::<W>(values, gene) };
        if v != 0
            && let Some(p) = cx.presence.as_deref_mut()
        {
            p[gene] += 1;
        }
        if (dense || v != 0)
            && let Some(votes) = cx.vote.as_deref_mut()
        {
            ScanCtx::cast::<W>(votes, gene, v);
        }
    }
    Ok(info)
}

/// The scan pass over the selected rows.
pub(super) fn scan(src: &MatrixRef, rows: Rows, n: usize, opt: &EncodeOptions) -> Result<Scan> {
    let (vw, n_genes) = (src.value_width(), src.n_genes());
    let g = n_genes as usize;
    let mut presence = opt.renumber.then(|| vec![0u32; g]);
    let mut vote = opt.column_modes.map(|_| vec![0u64; g * Votes::words(vw)]);
    let mut cx = ScanCtx {
        n_genes,
        layout: opt.layout,
        presence: presence.as_deref_mut(),
        vote: vote.as_deref_mut(),
    };
    let mut infos = Vec::with_capacity(n);
    let (mut max_nnz, mut max_elems) = (0usize, 0usize);
    for i in 0..n {
        let row = src.row(rows.get(i))?;
        let info = match row {
            RowRef::Sparse(c) => with_indices!(c.indices, |idx| with_width!(vw, |W| scan_sparse::<_, W>(idx, c.values, &mut cx)))?,
            RowRef::Dense(v) => with_width!(vw, |W| scan_dense::<W>(v, &mut cx))?,
        };
        if opt.layout != LayoutChoice::Dense {
            max_nnz = max_nnz.max(info.stored);
        }
        max_nnz = max_nnz.max(info.nnz);
        let input_dense = matches!(row, RowRef::Dense(_)) as usize * g;
        max_elems = max_elems.max(if info.allowed { g } else { info.stored }).max(input_dense);
        infos.push(info);
    }
    Ok(Scan {
        rows: infos,
        presence,
        vote,
        max_nnz,
        max_elems: max_elems.max(max_nnz),
    })
}

/// Genes by descending presence, ties by gene id.
fn gene_order(presence: &[u32]) -> Vec<u32> {
    let g = presence.len();
    let top = presence.iter().copied().max().unwrap_or(0) as usize;
    let mut order = vec![0u32; g];
    if top <= 4 * g + 1024 {
        // Counting sort, stable over ascending genes.
        let mut start = vec![0u32; top + 2];
        for &p in presence {
            start[top - p as usize + 1] += 1;
        }
        for b in 1..start.len() {
            start[b] += start[b - 1];
        }
        for (gene, &p) in presence.iter().enumerate() {
            let b = top - p as usize;
            order[start[b] as usize] = gene as u32;
            start[b] += 1;
        }
    } else {
        let mut keys: Vec<u64> = presence
            .iter()
            .enumerate()
            .map(|(gene, &p)| ((u32::MAX - p) as u64) << 32 | gene as u64)
            .collect();
        keys.sort_unstable();
        for (o, k) in order.iter_mut().zip(keys) {
            *o = k as u32;
        }
    }
    order
}

/// Rough bytes of a value coding, to choose a coding per cell while the
/// models are still being counted: dictionary entries at full width plus
/// fixed-width ranks, never more than the raw values.
fn dict_cost(n: usize, k: usize, vw: usize) -> f64 {
    let ranks = if k > 1 {
        n as f64 * (k as f64).log2() / 8.0
    } else {
        0.0
    };
    ((k * vw) as f64 + ranks).min((n * vw) as f64)
}

/// Rough bytes of a mode coding of `n` elements, `e` of them exceptions with
/// `k` distinct exception values: the exception flags, the table and the
/// ranks. The dictionary counterpart is [`dict_cost`]; `e` is 0 when no
/// element differs from its mode.
fn mode_cost(n: usize, e: usize, k: usize, vw: usize) -> f64 {
    if e == 0 {
        return 0.0;
    }
    let p = e as f64 / n as f64;
    let flags = if p < 1.0 {
        -(p * p.log2() + (1.0 - p) * (1.0 - p).log2()) * n as f64 / 8.0
    } else {
        0.0
    };
    (k * vw) as f64
        + flags
        + if k > 1 {
            e as f64 * (k as f64).log2() / 8.0
        } else {
            0.0
        }
}

/// Dictionary and rank counts of one value coding.
type Counts = [Hist; 2];

/// Adds a prepared coding's dictionary and rank symbols to its counts.
fn count_coding(c: &CodingPrep, bytes: &[u8], h: &mut Counts) {
    h[0].add(c.dict.syms.of(bytes));
    if let Some(r) = c.ranks {
        h[1].add(r.syms.of(bytes));
    }
}

/// Every gene's value of a row in coded order.
pub(super) fn coded_view<const W: usize>(row: RowRef, cos: Option<&[u32]>, out: &mut [u8]) {
    match row {
        RowRef::Dense(v) => match cos {
            None => out[..v.len()].copy_from_slice(v),
            Some(cos) => {
                for (s, &c) in cos.iter().enumerate() {
                    // SAFETY: `cos` is a permutation of the row's genes, each
                    // with a value in both buffers.
                    unsafe { store::<W>(out, c as usize, load::<W>(v, s)) };
                }
            }
        },
        RowRef::Sparse(cell) => {
            out.fill(0);
            with_indices!(cell.indices, |idx| {
                for (t, &i) in idx.iter().enumerate() {
                    let s = i.id() as usize;
                    let c = cos.map_or(s, |cos| cos[s] as usize);
                    // SAFETY: validated genes and their coded ids lie below
                    // `n_genes`, the values in `out`; `t` is an element.
                    unsafe { store::<W>(out, c, load::<W>(cell.values, t)) };
                }
            })
        }
    }
}

/// The non-zeros of a dense row into `idx` and `val`; returns their count.
pub(super) fn nonzeros<const W: usize>(row: &[u8], idx: &mut [u32], val: &mut [u8]) -> usize {
    assert!(idx.len() >= row.len() / W && val.len() >= row.len());
    let mut n = 0;
    for g in 0..row.len() / W {
        // SAFETY: `g` is a gene of the row; at most one non-zero per gene,
        // which both buffers hold (checked above).
        unsafe {
            let v = load::<W>(row, g);
            if v != 0 {
                set(idx, n, g as u32);
                store::<W>(val, n, v);
                n += 1;
            }
        }
    }
    n
}

/// A row's non-zeros as a cell: a sparse row itself, a dense row's in `idx`
/// and `val`.
pub(super) fn cell_of<'r>(row: RowRef<'r>, vw: usize, idx: &'r mut [u32], val: &'r mut [u8]) -> CellRef<'r> {
    match row {
        RowRef::Sparse(c) => c,
        RowRef::Dense(v) => {
            let n = with_width!(vw, |W| nonzeros::<W>(v, idx, val));
            CellRef {
                indices: Indices::U32(&idx[..n]),
                values: &val[..n * vw],
            }
        }
    }
}

/// Prepares the sparse record of a row into `arena`.
pub(super) fn prepare_row(
    row: RowRef,
    vw: usize,
    n_genes: u32,
    cos: Option<&[u32]>,
    modes: Option<&[u8]>,
    ws: &mut PrepWs,
    arena: &mut Arena,
) -> SparsePrep {
    let cell = cell_of(row, vw, ws.sp_idx, ws.sp_val);
    let n = cell.indices.len();
    with_indices!(cell.indices, |idx| order(idx, cos, &mut ws.order));
    with_width!(vw, |W| prepare_sparse::<W>(cell.values, n, cos.is_some(), modes, n_genes, &ws.order, &mut ws.values, arena))
}

/// Prepares the dense record of a row into `arena`, its values in coded
/// order in `dense`.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_dense(
    row: RowRef,
    vw: usize,
    n_genes: u32,
    cos: Option<&[u32]>,
    modes: Option<&[u8]>,
    vws: &mut ValuesWs,
    dense: &mut [u8],
    arena: &mut Arena,
) -> ValuesPrep {
    let g = n_genes as usize;
    arena.clear();
    with_width!(vw, |W| {
        coded_view::<W>(row, cos, &mut dense[..g * W]);
        prepare_values::<W, _>(&Dense::<W> { values: &dense[..g * W] }, modes, vws, arena)
    })
}

/// The prepare pass and the model. With `cache`, each row's sparse record
/// is kept there for coding; otherwise rows are prepared in row scratch.
pub(super) fn build(
    src: &MatrixRef,
    rows: Rows,
    opt: &EncodeOptions,
    scan: Scan,
    ws: &mut RowWs,
    mut cache: Option<&mut Arena>,
) -> Result<Model> {
    let (vw, n_genes) = (src.value_width(), src.n_genes());
    let g = n_genes as usize;
    let Scan { rows: infos, presence, vote, max_nnz, max_elems } = scan;
    let vote = vote.map(|v| Votes::candidates(&v, vw));
    let source_of_coded = presence.map(|p| gene_order(&p));
    let coded_of_source = source_of_coded.as_ref().map(|soc| {
        let mut cos = vec![0u32; g];
        for (c, &s) in soc.iter().enumerate() {
            cos[s as usize] = c as u32;
        }
        cos
    });
    let cos = coded_of_source.as_deref();
    // Mode candidates in coded order, as records compare against them.
    let cand_coded = vote.as_ref().map(|cand| {
        let mut t = vec![0u8; g * vw];
        with_width!(vw, |W| {
            for c in 0..g {
                let s = source_of_coded.as_ref().map_or(c, |soc| soc[c] as usize);
                // SAFETY: `c < n_genes`, a value of the table.
                unsafe { store::<W>(&mut t, c, cand[s]) };
            }
        });
        t
    });
    let modes = cand_coded.as_deref();

    let mut gaps = Hist::new();
    // Dictionary-coding counts of rows that do not prefer modes, and both
    // codings of those that do.
    let (mut plain, mut pref_dict, mut pref_mode): (Counts, Counts, Counts) =
        ([Hist::new(), Hist::new()], [Hist::new(), Hist::new()], [Hist::new(), Hist::new()]);
    let (mut max_kd, mut max_km) = (0usize, 0usize);
    let (mut hits, mut stored, mut saving) = (0u64, 0u64, 0f64);
    let allow_sparse = opt.layout != LayoutChoice::Dense;
    let keep = cache.is_some() && allow_sparse;
    let mut preps = Vec::with_capacity(if keep { infos.len() } else { 0 });
    for (i, info) in infos.iter().enumerate() {
        if info.nnz == 0 {
            if keep {
                preps.push(SparsePrep::default());
            }
            continue;
        }
        let row = src.row(rows.get(i))?;
        let sparse = allow_sparse.then(|| {
            let p = match cache.as_deref_mut() {
                Some(arena) => prepare_row(row, vw, n_genes, cos, modes, &mut ws.prep, arena),
                None => {
                    ws.arena.clear();
                    prepare_row(row, vw, n_genes, cos, modes, &mut ws.prep, &mut ws.arena)
                }
            };
            max_kd = max_kd.max(p.values.dict.k);
            if let Some(m) = p.values.mode {
                max_km = max_km.max(m.coding.k);
            }
            p
        });
        if keep {
            preps.push(sparse.unwrap_or_default());
        }
        let dense = info.allowed.then(|| {
            let d = prepare_dense(row, vw, n_genes, cos, modes, &mut ws.prep.values, ws.dense, &mut ws.dense_arena);
            max_kd = max_kd.max(d.dict.k);
            if let Some(m) = d.mode {
                max_km = max_km.max(m.coding.k);
            }
            d
        });
        // Counts follow the layout the row is expected to take.
        let sparse_bytes = match cache.as_deref() {
            Some(a) => a.contents(),
            None => ws.arena.contents(),
        };
        let (vp, bytes) = match (info.dense, sparse, dense) {
            (true, _, Some(d)) => (d, ws.dense_arena.contents()),
            (false, Some(p), _) => {
                gaps.add(p.gaps.syms.of(sparse_bytes));
                (p.values, sparse_bytes)
            }
            _ => unreachable!("a row with non-zeros has its expected candidate"),
        };
        let mut prefer = None;
        if vote.is_some() {
            let m = vp.mode.expect("rows are prepared against the mode candidates");
            let (e_n, e) = (vp.n, m.exceptions);
            hits += (e_n - e) as u64;
            stored += e_n as u64;
            let gain = dict_cost(e_n, vp.dict.k, vw) - mode_cost(e_n, e, m.coding.k, vw);
            if gain > 0.0 || opt.values == ValueChoice::Modes {
                saving += gain.max(0.0);
                prefer = Some(m);
            }
        }
        if opt.values != ValueChoice::Raw {
            match prefer {
                Some(m) => {
                    count_coding(&vp.dict, bytes, &mut pref_dict);
                    count_coding(&m.coding, bytes, &mut pref_mode);
                }
                None => count_coding(&vp.dict, bytes, &mut plain),
            }
        }
    }
    // Keep the column modes only if enough stored values match them and the
    // estimated saving over per-cell dictionaries clearly pays for the table
    // (always, when mode coding is forced).
    let modes = match (vote, opt.column_modes) {
        (Some(cand), Some(share)) => {
            let table = (g * vw) as f64;
            let keep = opt.values == ValueChoice::Modes
                || (hits as f64 >= share as f64 * stored as f64 && saving > 2.0 * table);
            keep.then_some(cand)
        }
        _ => None,
    };
    let preferred = if modes.is_some() && opt.values != ValueChoice::Dict { pref_mode } else { pref_dict };
    let mut counts = [[0u64; 256]; N_MODELS];
    counts[GAP] = gaps.finish();
    for ((model, a), b) in [DICT, RANK].into_iter().zip(plain).zip(preferred) {
        let (a, b) = (a.finish(), b.finish());
        for s in 0..256 {
            counts[model][s] = a[s] + b[s];
        }
    }
    let max_k = if modes.is_some() { max_kd.max(max_km) } else { max_kd };
    Ok(Model {
        source_of_coded,
        modes,
        freqs: counts.map(|c| tables::normalize(&c)),
        max_nnz,
        max_k,
        max_elems,
        rows: infos,
        preps,
    })
}
