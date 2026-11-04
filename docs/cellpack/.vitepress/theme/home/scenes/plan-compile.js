import data from "../scenes.json";
import { alpha, clamp, compact, ease, fillRound, lerp, line, mix, seg, strokeRound, text } from "../engine.js";
import { arc, bracket, fit, hatch, pill } from "./plan-draw.js";
import { TOY, compile, entries, reads } from "./plan-model.js";

// Compiling a page plan on a 16-access toy sequence with two frames. Belady's
// serial optimum is the baseline. Out-of-order workers could admit a page into
// a frame while a slower worker still reads the old one, so execution runs in a
// window of D positions and every fresh admission reserves its frame from
// u − D: an approximation of Belady that is safe to execute. Then the offline
// compiler's steps, frame naming, and the report's measurement.
const SERIAL = compile(TOY.seq, { frames: TOY.frames, gap: 0 });
const PLAN = compile(TOY.seq, { frames: TOY.frames, gap: TOY.gap });
const ENTRIES = entries(TOY.seq, PLAN);
const N = TOY.seq.length;
const D = TOY.gap;
const key = (r) => `${r.page}${r.start}`;
const LOST = SERIAL.residencies.find((r) => !PLAN.residencies.some((g) => key(g) === key(r)));
const LOST_IV = PLAN.intervals.find((i) => i.page === LOST.page && i.u === LOST.start);
const PH = {
  arcs: [0.02, 0.1],
  sweep: [0.12, 0.3],
  race: [0.34, 0.44],
  window: [0.5, 0.58],
  plan: [0.62, 0.7],
  offline: [0.73, 0.8],
  freeze: [0.82, 0.88],
  chart: [0.9, 0.94],
};
const B = data.cache.budgets.indexOf(0.1);

function timeline(ctx, { p, box, c, wide }, a) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  const W = box.w;
  const H = box.h;
  const fs = wide ? 11 : 9;
  const lab = wide ? 92 : 30;
  const cw = (W - lab) / N;
  const X = (i) => box.x + lab + i * cw;
  const mid = (i) => X(i) + cw / 2;
  const arcH = clamp(H * 0.15, 50, 110);
  const th = Math.min(30, cw * 0.9);
  const lh = clamp(H * 0.04, 16, 26);
  const tapeY = box.y + 34 + arcH;
  const beladyY = tapeY + th + 28;
  const planY = beladyY + 2 * (lh + 4) + 22;
  const detailY = planY + 2 * (lh + 4) + 26;
  const detailH = box.y + H - detailY;

  const arcs = seg(p, ...PH.arcs);
  const sweep = seg(p, ...PH.sweep);
  const race = ease.out(seg(p, ...PH.race));
  const win = ease.inOut(seg(p, ...PH.window));
  const plan = ease.inOut(seg(p, ...PH.plan));
  const offline = ease.out(seg(p, ...PH.offline));
  const freeze = ease.out(seg(p, ...PH.freeze));
  const cursor = sweep * N;

  text(ctx, `${N} ACCESSES · ${TOY.frames} CACHE FRAMES`, box.x, box.y + 14, { size: fs, family: "mono", weight: 700, color: c.text2 });

  // Tape.
  for (let i = 0; i < N; i++) {
    const swept = sweep > 0 && i < cursor;
    fillRound(ctx, X(i) + 2, tapeY, cw - 4, th, 4, alpha(swept ? c.tealSoft : c.surface, 1));
    text(ctx, TOY.seq[i], mid(i), tapeY + th / 2 + 5, { size: Math.min(14, cw * 0.4), family: "mono", weight: 700, color: c.text, align: "center" });
    text(ctx, String(i), mid(i), tapeY + th + 11, { size: 8, family: "mono", color: c.text3, align: "center" });
  }

  // Reuse intervals: grey until decided, teal kept, dashed accent read again.
  SERIAL.intervals.forEach((s, k) => {
    const show = clamp(arcs * (SERIAL.intervals.length + 2) - k);
    if (show <= 0) return;
    const g = PLAN.intervals[k];
    const decided = sweep > 0 && s.v < cursor;
    const kept = plan > 0.5 ? g.kept : s.kept;
    let col = alpha(c.text3, 0.55);
    let dash;
    if (decided) {
      col = kept ? c.teal : c.accent;
      if (!kept) dash = [4, 3];
      if (s.kept && !g.kept && plan > 0 && plan < 1) col = mix(c.teal, c.danger, plan);
    }
    const h = Math.min(arcH, (s.v - s.u) * cw * 0.32 + 10);
    ctx.globalAlpha = a * show;
    arc(ctx, mid(s.u), mid(s.v), tapeY - 2, h, col, decided ? 2 : 1.4, dash);
    ctx.globalAlpha = a;
  });
  if (sweep > 0 && sweep < 1) {
    const x = box.x + lab + cursor * cw;
    line(ctx, x, tapeY - arcH - 8, x, beladyY + 2 * (lh + 4), c.text, 1.5);
    text(ctx, "sweep", x + 4, tapeY - arcH - 2, { size: fs - 1, family: "mono", weight: 700, color: c.text });
  }

  // Two sets of lanes: Belady's serial optimum, and the executable plan.
  const lanes = (y0, label, sub, tone, alphaRow) => {
    if (alphaRow <= 0) return;
    ctx.globalAlpha = a * alphaRow;
    if (wide) {
      text(ctx, label, box.x, y0 + 10, { size: fs - 1, family: "mono", weight: 700, color: tone });
      text(ctx, sub, box.x, y0 + 24, { size: fs - 2, family: "mono", color: c.text3 });
    }
    [0, 1].forEach((f) => {
      const y = y0 + f * (lh + 4);
      strokeRound(ctx, X(0), y, X(N) - X(0), lh, 4, alpha(c.border, 1), 1);
      text(ctx, `f${f}`, X(0) - 6, y + lh / 2 + 4, { size: fs - 2, family: "mono", weight: 700, color: tone, align: "right" });
    });
    ctx.globalAlpha = a;
  };
  const bar = (r, y0, color, fade, reserve) => {
    if (fade <= 0) return;
    const decided = [r.start, ...r.uses].filter((v, i) => i === 0 || v < cursor);
    if (decided.length < 2) return;
    const end = decided[decided.length - 1];
    const y = y0 + r.frame * (lh + 4) + 2;
    if (reserve > 0 && r.start > r.from) {
      const pre = reserve * (r.start - r.from);
      hatch(ctx, X(r.start - pre), y, X(r.start) - X(r.start - pre), lh - 4, alpha(color, 0.75 * fade));
    }
    fillRound(ctx, X(r.start), y, X(end + 1) - X(r.start), lh - 4, 3, alpha(color, 0.5 * fade));
    for (const v of decided) fillRound(ctx, mid(v) - 2, y + lh - 9, 4, 4, 2, alpha(c.bg, fade));
    const label = freeze > 0 ? `${r.page}·g${r.generation}` : r.page;
    text(ctx, label, X(r.start) + 4, y + lh / 2 + 2, { size: fs - 1, family: "mono", weight: 700, color: c.text, a: fade });
  };
  const beladyCol = mix(c.text3, c.teal, 0.35);
  lanes(beladyY, "BELADY", "serial optimum", beladyCol, sweep > 0 ? 1 : 0);
  SERIAL.residencies.forEach((r) => {
    const flagged = r === LOST && (race > 0 || plan > 0);
    bar(r, beladyY, flagged ? mix(beladyCol, c.danger, Math.max(race * (1 - win * 0.5), plan * 0.5)) : beladyCol, 1, 0);
  });
  lanes(planY, "CELLPACK", `window D = ${D}`, c.teal, plan);
  PLAN.residencies.forEach((r) => bar(r, planY, c.teal, plan, plan));
  if (plan > 0) {
    bracket(ctx, X(LOST_IV.from), X(LOST_IV.v + 1), planY + LOST.frame * (lh + 4) + lh + 6, c.danger, `${LOST.page} needs a frame over [${LOST_IV.from}, ${LOST_IV.v}]: both are busy at ${LOST_IV.from}–${LOST.start - 1}`, { a: plan * (1 - offline), size: fs - 2 });
  }

  // Page reads: the baseline and the plan.
  const seen = Math.min(N, Math.ceil(cursor));
  const savedS = SERIAL.intervals.filter((i) => sweep > 0 && i.v < cursor && i.kept).length;
  if (sweep > 0) {
    text(ctx, `Belady ${sweep >= 1 ? reads(TOY.seq, SERIAL) : seen - savedS} reads`, box.x + W, box.y + 14, { size: fs + 2, family: "head", weight: 600, color: beladyCol, align: "right" });
    text(ctx, `plan ${reads(TOY.seq, PLAN)} reads · approximate Belady`, box.x + W, box.y + 32, { size: fs, family: "mono", weight: 700, color: c.teal, align: "right", a: plan });
  }

  // The detail panel changes with the beat.
  const panel = { x: box.x, y: detailY, w: W, h: detailH };
  const raceA = race * (1 - plan);
  if (raceA > 0) execution(ctx, { box: panel, c, fs, wide }, raceA, win);
  const offA = offline * (1 - freeze);
  if (offA > 0) occupancy(ctx, { box: panel, c, fs, wide, X, mid }, offA);
  if (freeze > 0) {
    const ey = detailY + 4;
    text(ctx, "entry", X(0) - 6, ey + 12, { size: 8, family: "mono", color: c.text3, align: "right", a: freeze });
    ENTRIES.forEach((e, i) => {
      const v = clamp((freeze * (N + 4)) / 4 - i / 4);
      const label = e.kind === "read" ? "read" : `${e.kind === "admit" ? "+" : ""}f${e.frame}`;
      const col = e.kind === "read" ? c.text3 : e.kind === "admit" ? c.accent : c.teal;
      pill(ctx, X(i) + 2, ey, cw - 4, 18, label, col, { a: v, size: Math.min(9, cw * 0.3) });
    });
    fit(ctx, "+f = admit: read, keep, publish the generation · f = use: wait for it · read = bypass", box.x, ey + 40, W, { size: fs - 1, family: "mono", color: c.text2, a: freeze });
    fit(ctx, "Frames are named in admission order; each new tenant raises its frame's generation.", box.x, ey + 58, W, { size: fs - 1, family: "mono", color: c.text2, a: freeze });
  }
  ctx.restore();
}

// Out-of-order execution of positions 4–8 on three workers, in time. Without
// a window, position 7 admits F into f1 while position 6 still reads C from it.
// With a window of D, position u starts only after every position ≤ u − D has
// finished; 7 still overlaps 6, so the plan cannot give F the frame C holds.
const RUN = [
  { pos: 4, w: 0, s: 0.0, e: 2.2, what: "read D" },
  { pos: 5, w: 1, s: 0.3, e: 1.6, what: "use E · f0" },
  { pos: 6, w: 2, s: 0.6, e: 3.6, what: "use C · f1 · slow decode", reads: true },
  { pos: 7, w: 1, s: 1.7, e: 3.0, what: "admit F → f1", writes: true },
  { pos: 8, w: 0, s: 2.3, e: 3.5, what: "use E · f0" },
];
function execution(ctx, { box, c, fs, wide }, a, win) {
  ctx.save();
  ctx.globalAlpha = a;
  const T = 4;
  const lab = wide ? 70 : 30;
  const rows = 3;
  const rh = clamp((box.h - 90) / rows, 16, 26);
  const x = (time) => box.x + lab + (time / T) * (box.w - lab);
  const rowY = (w) => box.y + 26 + w * (rh + 4);
  text(ctx, "IN TIME: THREE WORKERS RUN POSITIONS 4–8", box.x, box.y + 10, { size: fs - 1, family: "mono", weight: 700, color: c.text2 });
  for (let w = 0; w < rows; w++) text(ctx, `W${w}`, box.x + lab - 8, rowY(w) + rh / 2 + 4, { size: fs - 1, family: "mono", weight: 700, color: c.text3, align: "right" });
  // With the window, 7 may start only once 4 and 5 (≤ 7 − D) are done.
  const gate = Math.max(RUN[0].e, RUN[1].e);
  for (const r of RUN) {
    let s = r.s;
    let e = r.e;
    if (r.pos === 7) {
      s = lerp(r.s, gate, win);
      e = s + (r.e - r.s);
    }
    const y = rowY(r.w);
    const col = r.writes ? (win > 0.5 ? c.text3 : c.danger) : r.reads ? c.accent : c.teal;
    fillRound(ctx, x(s), y, x(e) - x(s) - 2, rh, 4, alpha(col, 0.35));
    strokeRound(ctx, x(s), y, x(e) - x(s) - 2, rh, 4, alpha(col, 0.9), 1);
    const what = r.writes && win > 0.5 ? "read F into a private frame" : r.what;
    fit(ctx, `${r.pos} · ${what}`, x(s) + 6, y + rh / 2 + 4, x(e) - x(s) - 10, { size: fs - 1, family: "mono", weight: 600, color: c.text });
  }
  const below = rowY(rows) + 14;
  // The collision: f1 written while it is still being read.
  const clash = 1 - win;
  if (clash > 0.02) {
    const w7 = RUN[3];
    const r6 = RUN[2];
    const s = Math.max(w7.s, r6.s);
    const e = Math.min(w7.e, r6.e);
    const y0 = rowY(w7.w);
    const y1 = rowY(r6.w) + rh;
    fillRound(ctx, x(s), y0 - 2, x(e) - x(s), y1 - y0 + 4, 4, alpha(c.danger, 0.12 * clash));
    strokeRound(ctx, x(s), y0 - 2, x(e) - x(s), y1 - y0 + 4, 4, alpha(c.danger, clash), 1.5);
    fit(ctx, "Belady's frames, run out of order: F is admitted into f1 before C is evicted —", box.x, below, box.w, { size: fs, family: "mono", weight: 700, color: c.danger, a: clash });
    fit(ctx, "position 7 overwrites the page that position 6 is still decoding.", box.x, below + 17, box.w, { size: fs, family: "mono", weight: 700, color: c.danger, a: clash });
  }
  if (win > 0) {
    line(ctx, x(gate), box.y + 18, x(gate), rowY(rows), alpha(c.teal, win), 1.5, [4, 3]);
    fit(ctx, `7 waits for positions ≤ ${7 - D}`, x(gate) + 6, box.y + 18, box.x + box.w - x(gate) - 6, { size: fs - 1, family: "mono", weight: 700, color: c.teal, a: win });
    fit(ctx, "Window: in-flight positions stay within D of the oldest unfinished one.", box.x, below, box.w, { size: fs, family: "mono", weight: 700, color: c.teal, a: win });
    fit(ctx, "A frame last used at or before u − D is free when u starts: eviction precedes admission.", box.x, below + 17, box.w, { size: fs - 1, family: "mono", color: c.text2, a: win });
    fit(ctx, "6 > 7 − D may still run, so the plan cannot hand C's frame to F.", box.x, below + 34, box.w, { size: fs - 1, family: "mono", color: c.text2, a: win });
  }
  ctx.restore();
}

// What the offline compiler computes: frames in use per position under the
// plan, against the budget; the rejected interval hits the budget.
function occupancy(ctx, { box, c, fs, wide, X, mid }, a) {
  ctx.save();
  ctx.globalAlpha = a;
  const ph = clamp(box.h * 0.3, 30, 54);
  const y0 = box.y + 22 + ph;
  const unit = ph / TOY.frames;
  text(ctx, "OFFLINE: FRAMES IN USE, POSITION BY POSITION", box.x, box.y + 10, { size: fs - 1, family: "mono", weight: 700, color: c.text2 });
  PLAN.occ.forEach((n, i) => fillRound(ctx, X(i) + 3, y0 - n * unit, X(i + 1) - X(i) - 6, n * unit, 2, alpha(c.teal, 0.45)));
  line(ctx, X(0), y0 - TOY.frames * unit, X(N), y0 - TOY.frames * unit, alpha(c.danger, 0.8), 1.2, [5, 3]);
  text(ctx, `budget ${TOY.frames}`, X(N), y0 - TOY.frames * unit - 4, { size: fs - 2, family: "mono", color: c.danger, align: "right" });
  line(ctx, X(0), y0, X(N), y0, c.border, 1);
  strokeRound(ctx, X(LOST_IV.from) + 1, y0 - ph - 2, X(LOST_IV.v + 1) - X(LOST_IV.from) - 2, ph + 4, 4, alpha(c.danger, 0.9), 1.5);
  text(ctx, `${LOST.page}: max over [${LOST_IV.from}, ${LOST_IV.v}] = ${TOY.frames} → read again`, mid(LOST_IV.from), y0 + 14, { size: fs - 2, family: "mono", weight: 700, color: c.danger });
  const steps = [
    "1  collect every reuse interval (u, v) of the stored sequence",
    "2  sweep by v: keep (u, v) if max in-use over [u − D, v] < frames, then add it there",
    "3  name frames in admission order and set generations",
    "4  write the plan, bound to this exact sequence; replay it to verify",
  ];
  steps.forEach((s, k) => fit(ctx, s, box.x, y0 + (wide ? 34 : 28) + k * (wide ? 17 : 13), box.w, { size: fs - 1, family: "mono", color: k === 1 ? c.text : c.text2, weight: k === 1 ? 600 : 400 }));
  if (wide) fit(ctx, "range-max / range-add tree over positions · page groups compile in parallel shards, independent of scheduling", box.x, y0 + 34 + 4 * 17, box.w, { size: fs - 2, family: "mono", color: c.text3 });
  ctx.restore();
}

// The report's cache-policy ablation at a 10% budget: page misses removed and
// delivered cells/s for every policy; Belady is the simulated optimum.
function chart(ctx, { p, box, c, wide }, a) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  const W = box.w;
  const fs = wide ? 11 : 9;
  const grow = ease.out(seg(p, PH.chart[1], 0.99));
  const methods = [...data.cache.methods].sort((x, y) => y.miss[B] - x.miss[B]);
  const belady = data.cache.belady[B];
  const nameW = wide ? 110 : 76;
  const split = wide ? 0.6 : 0.58;
  const missX = box.x + nameW;
  const missW = W * split - nameW - 10;
  const tpX = box.x + W * split + 10;
  const tpW = W - (tpX - box.x);
  const maxTp = Math.max(...methods.map((m) => m.cells_per_s[B]));
  const rh = clamp((box.h - 190) / (methods.length + 1), 22, 50);
  const top = box.y + Math.max(64, (box.h - rh * (methods.length + 1) - 60) * 0.42);
  const scale = (v) => (v / 0.45) * missW;

  text(ctx, "10% CACHE BUDGET · REPORT FIG. 6", box.x, box.y + 14, { size: fs, family: "mono", weight: 700, color: c.text2 });
  text(ctx, wide ? "page misses removed" : "misses removed", missX, top - 14, { size: fs - 1, family: "mono", color: c.text3 });
  text(ctx, wide ? "cells/s delivered" : "cells/s", tpX, top - 14, { size: fs - 1, family: "mono", color: c.text3 });
  // Belady first: the optimum every policy is measured against.
  text(ctx, "Belady", missX - 8, top + rh * 0.55, { size: fs, weight: 700, color: c.text, align: "right" });
  hatch(ctx, missX, top + rh * 0.2, scale(belady) * grow, rh * 0.5, alpha(c.text, 0.5));
  strokeRound(ctx, missX, top + rh * 0.2, scale(belady) * grow, rh * 0.5, 3, alpha(c.text, 0.7), 1);
  fit(ctx, `${(belady * 100).toFixed(1)}% · simulated optimum`, missX + scale(belady) * grow + 5, top + rh * 0.58, W - (missX + scale(belady)) + box.x - 5, { size: fs - 1, family: "mono", weight: 700, color: c.text, a: grow });
  methods.forEach((m, k) => {
    const y = top + (k + 1) * rh;
    const ours = m.key.startsWith("cellpack");
    const col = ours ? c.accent : mix(c.text3, c.bg, 0.15);
    text(ctx, m.name, missX - 8, y + rh * 0.55, { size: fs, weight: ours ? 700 : 500, color: ours ? c.accent : c.text2, align: "right" });
    const bh = rh * 0.5;
    fillRound(ctx, missX, y + rh * 0.2, scale(m.miss[B]) * grow, bh, 3, alpha(col, ours ? 0.85 : 0.55));
    text(ctx, `${(m.miss[B] * 100).toFixed(1)}%`, missX + scale(m.miss[B]) * grow + 5, y + rh * 0.2 + bh * 0.75, { size: fs - 1, family: "mono", weight: ours ? 700 : 400, color: ours ? c.accent : c.text2, a: grow });
    const tw = (m.cells_per_s[B] / maxTp) * (tpW - 46) * grow;
    fillRound(ctx, tpX, y + rh * 0.2, tw, bh, 3, alpha(col, ours ? 0.85 : 0.55));
    text(ctx, compact(m.cells_per_s[B]), tpX + tw + 5, y + rh * 0.2 + bh * 0.75, { size: fs - 1, family: "mono", weight: ours ? 700 : 400, color: ours ? c.accent : c.text2, a: grow });
  });
  const bx = missX + scale(belady);
  const bottom = top + (methods.length + 1) * rh;
  line(ctx, bx, top - 4, bx, bottom, alpha(c.text, 0.5 * grow), 1.2, [5, 4]);
  const nx = tpX + (data.cache.no_cache / maxTp) * (tpW - 46);
  line(ctx, nx, top + rh - 4, nx, bottom, alpha(c.text3, 0.8 * grow), 1.2, [3, 3]);
  fit(ctx, `no cache ${compact(data.cache.no_cache)}`, nx, bottom + 16, tpW, { size: fs - 1, family: "mono", color: c.text3, align: "center", a: grow });
  fit(ctx, `CellPack = offline plan · CellPack* = online plan · ${data.cache.repeats}-run means, ${compact(data.cache.cells)} cells · Belady: same-protocol simulation`, box.x, bottom + 40, W, { size: fs - 1, family: "mono", color: c.text3, a: grow });
  ctx.restore();
}

function draw(ctx, env) {
  const to = ease.inOut(seg(env.p, ...PH.chart));
  timeline(ctx, env, 1 - to);
  chart(ctx, env, to);
}

export default { draw, continuous: false };
