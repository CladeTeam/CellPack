// A small model of the page-plan compiler (crates/cellpack/src/opt/engine.rs)
// that the planning scenes draw. Page mode only.
//
// Every pair of consecutive uses (u, v) of a page is a reuse interval, decided
// when the sweep reaches v. Keeping it holds a frame over [u − D, v], or over
// [u + 1, v] when the page's previous interval was kept and ended at u. It is
// kept when a frame is free over that whole span; otherwise the page is read
// again (bypass). With D = 0 this is the interval form of Belady with bypass.
// An online compiler with horizon H only decides intervals with v − u ≤ H.

import { hash } from "../engine.js";

export function compile(seq, { frames, gap = 0, horizon = Infinity }) {
  const occ = new Array(seq.length).fill(0);
  const last = new Map();
  const keptEnd = new Map();
  const intervals = [];
  seq.forEach((page, v) => {
    if (last.has(page)) {
      const u = last.get(page);
      const chained = keptEnd.get(page) === u;
      const from = chained ? u + 1 : Math.max(0, u - gap);
      const seen = v - u <= horizon;
      let peak = 0;
      for (let t = from; t <= v; t++) peak = Math.max(peak, occ[t]);
      const kept = seen && peak < frames;
      if (kept) {
        for (let t = from; t <= v; t++) occ[t] += 1;
        keptEnd.set(page, v);
      }
      intervals.push({ page, u, v, from, chained, kept, seen });
    }
    last.set(page, v);
  });
  return { intervals, occ, residencies: residencies(intervals, frames) };
}

// Chains of kept intervals become residencies: one page in one frame from its
// admission to its last use. Frames go to residencies in admission order, each
// to the lowest frame whose previous tenant has left (the spans already carry
// the gap). A frame's generation counts its tenants.
function residencies(intervals, frames) {
  const out = [];
  const open = new Map();
  for (const it of intervals) {
    if (!it.kept) continue;
    const r = it.chained ? open.get(it.page) : null;
    if (r) {
      r.end = it.v;
      r.uses.push(it.v);
    } else {
      const n = { page: it.page, from: it.from, start: it.u, end: it.v, uses: [it.v] };
      out.push(n);
      open.set(it.page, n);
    }
  }
  out.sort((a, b) => a.from - b.from || a.start - b.start);
  const free = new Array(frames).fill(-1);
  const tenants = new Array(frames).fill(0);
  for (const r of out) {
    let f = free.findIndex((end) => end < r.from);
    if (f < 0) f = free.indexOf(Math.min(...free));
    r.frame = f;
    r.generation = ++tenants[f];
    free[f] = r.end;
  }
  return out;
}

/** What the plan tells the worker at each position: admit, use or read. */
export function entries(seq, plan) {
  const out = seq.map((page) => ({ page, kind: "read" }));
  for (const r of plan.residencies) {
    out[r.start] = { page: r.page, kind: "admit", frame: r.frame, generation: r.generation };
    for (const v of r.uses) out[v] = { page: r.page, kind: "use", frame: r.frame, generation: r.generation };
  }
  return out;
}

export const reads = (seq, plan) => seq.length - plan.intervals.filter((i) => i.kept).length;

// The compile scene's sequence: 16 accesses over 2 frames, chosen so that a
// window of D = 2 costs exactly one interval that serial Belady keeps.
export const TOY = { seq: [..."ECFDBECFEGFEBABE"], frames: 2, gap: 2 };

/**
 * A long synthetic stream for the online scene: mostly short reuse, some reuse
 * farther back than the horizon, and fresh pages. Deterministic.
 */
export function stream(n, seed = 11) {
  const seq = [];
  const uses = [];
  let next = 0;
  for (let i = 0; i < n; i++) {
    const r = hash(i, seed);
    let page;
    if (r < 0.5 && i > 3) page = seq[i - 2 - Math.floor(hash(i, seed + 1) * Math.min(i - 2, 9))];
    else if (r < 0.62 && i > 26) page = seq[i - 19 - Math.floor(hash(i, seed + 2) * 8)];
    // A page is shared by a few cells, not by every cell.
    if (page === undefined || page === seq[i - 1] || uses[page] >= 3) page = next++;
    uses[page] = (uses[page] || 0) + 1;
    seq.push(page);
  }
  return seq;
}
