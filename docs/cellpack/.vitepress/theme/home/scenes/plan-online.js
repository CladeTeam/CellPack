import { alpha, clamp, ease, fillRound, line, mix, seg, strokeRound, text } from "../engine.js";
import { arc, bracket, fit, pill } from "./plan-draw.js";
import { compile, entries, stream } from "./plan-model.js";

// Online compilation over a sequence that is never stored. Four frontiers move
// together: the compiler at X decides intervals within the horizon H, freezes
// and names frames at Z = X − H, publishes entries at P = Z − D, and workers
// consume them at F. Page mode only.
const FRAMES = 3;
const D = 3;
const H = 12;
const C = 2;
const SEQ = stream(420, 23);
const PLAN = compile(SEQ, { frames: FRAMES, gap: D, horizon: H });
// The same stream compiled at every horizon the opening sweeps through.
const HMIN = 3;
const HMAX = 22;
const PLANS = Object.fromEntries([...Array(HMAX - HMIN + 1)].map((_, i) => [HMIN + i, compile(SEQ, { frames: FRAMES, gap: D, horizon: HMIN + i })]));
const FULL = compile(SEQ, { frames: FRAMES, gap: D }).intervals.filter((i) => i.kept).length;
const ENTRIES = entries(SEQ, PLAN);
const START = 60;
const TRAVEL = 120;
const PH = { sweepH: [0.03, 0.2], regions: [0.27, 0.35], publish: [0.43, 0.51], lost: [0.6, 0.68], early: [0.8, 0.9] };

function draw(ctx, { p, box, c, wide }) {
  const W = box.w;
  const Hh = box.h;
  const fs = wide ? 11 : 9;
  const lab = wide ? 34 : 24;
  const vis = wide ? 36 : 26;
  const cw = (W - lab) / vis;
  const regions = ease.out(seg(p, ...PH.regions));
  // Opening: the horizon grows, then settles at H for the rest of the scene.
  const sw = seg(p, ...PH.sweepH);
  const Hcur = Math.round(sw < 0.6 ? lerpN(HMIN, HMAX, ease.inOut(sw / 0.6)) : lerpN(HMAX, H, ease.inOut((sw - 0.6) / 0.4)));
  const opening = 1 - seg(p, PH.sweepH[1] + 0.01, PH.regions[0] - 0.01);
  const plan = regions > 0.5 ? PLAN : PLANS[Hcur];
  const publish = ease.out(seg(p, ...PH.publish));
  const lost = ease.out(seg(p, ...PH.lost));
  const early = ease.inOut(seg(p, ...PH.early));

  // Frontiers, in positions. The compiler runs at a fixed screen column; the
  // tape scrolls under it as the reader scrolls.
  const X = START + p * TRAVEL;
  const xCol = vis - 5;
  const h = regions > 0 ? lerpN(H, D + C, early) : Hcur; // freezing early shortens the horizon
  const Z = X - h;
  const P = Z - D;
  const F = P - lerpN(6, 1, early);
  const left = X - xCol; // position at the left edge
  const sx = (i) => box.x + lab + (i - left) * cw;
  const visible = (i) => sx(i) > box.x + lab - cw && sx(i) < box.x + W;

  const arcH = clamp(Hh * 0.2, 60, 140);
  const th = Math.min(26, cw * 1.1);
  const markY = box.y + 30;
  const tapeY = Math.max(markY + 40 + arcH, box.y + Hh * 0.4);
  const lh = clamp(Hh * 0.06, 16, 34);
  const laneY = tapeY + th + 34;
  const lanes = [...Array(FRAMES).keys()].map((f) => laneY + f * (lh + 6));
  const entY = lanes[FRAMES - 1] + lh + 22;

  // Region bands behind the tape.
  const band = (a, b, color, fill, label, y = tapeY - 6) => {
    const x0 = Math.max(box.x + lab, sx(a));
    const x1 = Math.min(box.x + W, sx(b));
    if (x1 <= x0) return;
    fillRound(ctx, x0, y, x1 - x0, th + 12, 6, alpha(color, fill));
    if (label) fit(ctx, label, (x0 + x1) / 2, y + th + 26, x1 - x0 - 4, { size: fs - 1, family: "mono", color, align: "center", a: regions });
  };
  band(left - 2, F, c.text3, 0.06 * regions);
  band(F, P, c.teal, 0.2 * regions, "plan ring");
  band(P, Z, c.teal, 0.1 * regions, "frozen");
  band(Z, X + 1, c.accent, 0.08 * Math.max(regions, opening * 0.8), regions > 0 ? "deciding" : "");

  // Tape: pages up to the compiler frontier; nothing is known beyond it.
  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x + lab, box.y, W - lab, Hh);
  ctx.clip();
  for (let i = Math.floor(left) - 1; i <= Math.ceil(left + vis); i++) {
    if (i < 0) continue;
    const x = sx(i);
    const known = i <= X;
    const done = i < F;
    const hue = (SEQ[i] * 47) % 360;
    const fill = known ? `hsla(${hue}, 32%, ${c.bg === "#171a18" ? 42 : 72}%, ${done ? 0.35 : 0.9})` : alpha(c.divider, 0.5);
    fillRound(ctx, x + 1.5, tapeY, cw - 3, th, 3, fill);
    if (!known) strokeRound(ctx, x + 1.5, tapeY, cw - 3, th, 3, alpha(c.border, 0.6), 1);
    if (known && wide && cw > 16) text(ctx, String(SEQ[i] % 100), x + cw / 2, tapeY + th / 2 + 4, { size: 8, family: "mono", color: c.text, align: "center", a: done ? 0.4 : 0.85 });
  }

  // Decided intervals as arcs: kept, read again, or beyond the horizon.
  for (const it of plan.intervals) {
    if (it.v > X || (!visible(it.u) && !visible(it.v))) continue;
    const far = !it.seen;
    let col = it.kept ? c.teal : c.accent;
    let dash = it.kept ? null : [4, 3];
    if (far) {
      col = mix(c.accent, c.danger, Math.max(lost, opening));
      dash = [2, 3];
    }
    const hh = Math.min(arcH, (it.v - it.u) * cw * 0.3 + 8);
    const done = it.v < F ? 0.3 : 1;
    const strong = Math.max(lost, opening);
    ctx.globalAlpha = done * (far ? 0.6 + 0.4 * strong : 0.85);
    arc(ctx, sx(it.u) + cw / 2, sx(it.v) + cw / 2, tapeY - 2, hh, col, far ? 1.2 + strong : 1.6, dash);
    ctx.globalAlpha = 1;
    if (far && lost > 0 && it.v > Z - 2) text(ctx, `${it.v - it.u} > H`, (sx(it.u) + sx(it.v)) / 2 + cw / 2, tapeY - hh - 6, { size: 8, family: "mono", weight: 700, color: c.danger, align: "center", a: lost });
  }

  ctx.globalAlpha = regions;
  // Frames: named lanes left of Z; right of it only occupancy counts exist.
  for (const r of PLAN.residencies) {
    const a0 = Math.max(r.from, left - 1);
    const a1 = Math.min(r.end + 1, Z);
    if (a1 <= a0) continue;
    const y = lanes[r.frame] + 1;
    const pre = r.start > a0 ? Math.min(r.start, a1) : a0;
    if (pre > a0) fillRound(ctx, sx(a0), y, sx(pre) - sx(a0), lh - 2, 3, alpha(c.teal, 0.18));
    fillRound(ctx, sx(pre), y, sx(a1) - sx(pre), lh - 2, 3, alpha(c.teal, r.end < F ? 0.25 : 0.55));
  }
  // Occupancy as the compiler sees it at X: intervals decided so far.
  const decided = PLAN.intervals.filter((it) => it.kept && it.v <= X && it.v >= Z);
  for (let i = Math.max(0, Math.ceil(Z)); i <= Math.floor(X); i++) {
    const used = decided.filter((it) => it.from <= i && i <= it.v).length;
    for (let k = 0; k < FRAMES; k++) {
      const y = lanes[FRAMES - 1 - k] + 1;
      fillRound(ctx, sx(i) + 1.5, y, cw - 3, lh - 2, 3, k < used ? alpha(c.teal, 0.3) : alpha(c.divider, 0.35));
    }
  }
  ctx.restore();
  fit(ctx, "frames in use · not yet named", sx(Z) + 6, lanes[FRAMES - 1] + lh + 14, sx(X) - sx(Z) - 6, { size: fs - 2, family: "mono", color: c.teal, a: regions });
  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x + lab, box.y, W - lab, Hh);
  ctx.clip();
  // Published entries, consumed at F.
  for (let i = Math.max(0, Math.floor(F) - 3); i < P; i++) {
    const e = ENTRIES[i];
    const col = e.kind === "read" ? c.text3 : e.kind === "admit" ? c.accent : c.teal;
    const a = (i < F ? 0.25 : 1) * publish;
    if (cw > 15) pill(ctx, sx(i) + 1, entY, cw - 2, 14, e.kind === "read" ? "r" : `${e.kind === "admit" ? "+" : ""}${e.frame}`, col, { a, size: 7 });
    else fillRound(ctx, sx(i) + 1.5, entY, cw - 3, 10, 3, alpha(col, 0.5 * a));
  }
  ctx.restore();

  ctx.globalAlpha = 1;
  // Lane labels and the frame budget.
  [...Array(FRAMES).keys()].forEach((f) => text(ctx, `f${f}`, box.x, lanes[f] + lh / 2 + 4, { size: fs - 1, family: "mono", weight: 700, color: c.teal, a: regions }));
  if (cw > 15 || publish > 0) text(ctx, "entry", box.x, entY + 10, { size: 8, family: "mono", color: c.text3, a: publish });

  // Frontier markers.
  const marks = [
    ["F", F, "workers", c.text],
    ["P", P, "published", c.teal],
    ["Z", Z, "frozen", c.teal],
    ["X", X, "compiler", c.accent],
  ];
  marks.forEach(([name, at, what, col], k) => {
    if (k < 3 && regions <= 0) return;
    ctx.globalAlpha = k < 3 ? regions : 1;
    const x = sx(at);
    const crowded = k < 3 && sx(marks[k + 1][1]) - x < 64;
    line(ctx, x, markY + 18, x, entY + 16, alpha(col, 0.7), 1.4, k === 3 ? null : [3, 3]);
    text(ctx, name, x, markY + 12, { size: fs + 3, family: "head", weight: 700, color: col, align: k < 3 && sx(marks[k + 1][1]) - x < 16 ? "right" : "center" });
    if (wide && !crowded) text(ctx, what, x, markY - 4, { size: fs - 2, family: "mono", color: col, align: "center", a: regions });
  });
  ctx.globalAlpha = 1;
  bracket(ctx, sx(Z), sx(X), markY + 28, c.accent, "", { a: 1, dir: -1 });
  const hLabel = early > 0.05 ? "H shrinks: freeze early" : regions > 0 ? "horizon H" : `horizon H = ${Hcur}`;
  text(ctx, hLabel, sx(X) - 6, markY + 42, { size: fs - 1, family: "mono", weight: 700, color: c.accent, align: "right" });
  bracket(ctx, sx(P), sx(Z), markY + 28, c.teal, "D", { a: publish, size: fs - 1, dir: -1 });

  // Opening: what a horizon buys and what it costs.
  if (opening > 0.02) {
    const kept = PLANS[Hcur].intervals.filter((i) => i.kept).length;
    const my = laneY;
    const bw = Math.min(W, 520);
    const rowsO = [
      [`reuse kept: ${kept} of ${FULL} the full sequence would keep`, kept / FULL, c.teal],
      [`look-ahead held in memory: ${Hcur} accesses`, Hcur / HMAX, c.accent],
      [`first batch waits for the first ${Hcur} accesses to compile`, Hcur / HMAX, c.text2],
    ];
    ctx.globalAlpha = opening;
    text(ctx, "THE HORIZON TRADE-OFF", box.x, my - 6, { size: fs - 1, family: "mono", weight: 700, color: c.text2 });
    rowsO.forEach(([label, v, col], k) => {
      const y = my + 10 + k * 40;
      fillRound(ctx, box.x, y, bw, 10, 3, alpha(c.divider, 0.8));
      fillRound(ctx, box.x, y, bw * v, 10, 3, alpha(col, 0.8));
      fit(ctx, label, box.x, y + 26, W, { size: fs, family: "mono", weight: 600, color: col });
    });
    ctx.globalAlpha = 1;
  }
  // Bottom note: what each frontier guarantees.
  const ny = entY + 46;
  const notes = [
    ["decide (u, v) at v, only if v − u ≤ H; occupancy counts only", regions],
    ["at Z: name frames, set generations; at P = Z − D: publish", publish],
    ["new admissions reserve from u − D ≥ P: nothing published changes", publish],
    [`workers catch up → Z moves ahead, never past X − D − C`, early],
  ];
  notes.forEach(([n, a], k) => fit(ctx, n, box.x, ny + k * 17, W, { size: fs - 1, family: "mono", color: k === 3 ? c.accent : c.text2, a }));
}

const lerpN = (a, b, t) => a + (b - a) * t;

export default { draw, continuous: false };
