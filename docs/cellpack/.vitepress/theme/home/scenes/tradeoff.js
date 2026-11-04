import data from "../scenes.json";
import { alpha, clamp, ease, fillRound, glow, lerp, line, mix, seg, strokeRound, text } from "../engine.js";

// Compression ratio R(k) against cells per independently compressed unit k
// (report Fig. 2b), and what a random read then costs: about k / R(k) cells'
// worth of uncompressed bytes (report §2.2).
const CODECS = data.units.codecs;
const CELLPACK = data.units.cellpack;
const X1 = Math.log2(2048);
const YMAX = 6.5;
const COST = [Math.log10(0.1), Math.log10(1000)];

function valueAt(points, lx) {
  const xs = points.map(([u]) => Math.log2(u));
  if (lx <= xs[0]) return points[0][1];
  for (let i = 1; i < xs.length; i++)
    if (lx <= xs[i]) return lerp(points[i - 1][1], points[i][1], (lx - xs[i - 1]) / (xs[i] - xs[i - 1]));
  return points.at(-1)[1];
}

function curve(ctx, X, Y, fn, lx, color, width, a) {
  ctx.beginPath();
  for (let k = 0; k <= 80; k++) {
    const x = (lx * k) / 80;
    if (k === 0) ctx.moveTo(X(x), Y(fn(x)));
    else ctx.lineTo(X(x), Y(fn(x)));
  }
  ctx.strokeStyle = alpha(color, a);
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.stroke();
}

function star(ctx, x, y, r, color) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function draw(ctx, { p, t, box, c, wide }) {
  const unitH = Math.min(70, box.h * 0.12);
  const left = 46;
  const right = wide ? 110 : 100;
  const x0 = box.x + left;
  const w = box.w - left - right;
  const top = box.y + unitH + 30;
  const avail = box.y + box.h - top - 44;
  // The ratio chart first fills the area; the read-cost chart then joins below.
  const split = ease.inOut(seg(p, 0.4, 0.47));
  const A = { y: top, h: lerp(avail - 30, avail * 0.46, split) };
  const B = { y: top + avail * 0.46 + 64, h: avail * 0.54 - 64 };
  const X = (lx) => x0 + (lx / X1) * w;
  const YA = (v) => A.y + A.h - (v / YMAX) * A.h;
  const YB = (v) => B.y + B.h - ((Math.log10(Math.max(v, 0.1)) - COST[0]) / (COST[1] - COST[0])) * B.h;

  const axes = ease.out(seg(p, 0, 0.06));
  const sweep = ease.inOut(seg(p, 0.06, 0.36));
  const sweepB = ease.inOut(seg(p, 0.48, 0.66));
  const costOn = split;
  const drop = seg(p, 0.7, 0.8);
  const beam = ease.inOut(seg(p, 0.8, 0.9));
  const lx = lerp(0, X1, sweep);
  const lxB = lerp(0, X1, sweepB);
  const palette = [c.text3, c.teal, mix(c.teal, c.text, 0.45), c.danger, mix(c.accent, c.text3, 0.5)];

  // Panel A: compression ratio.
  for (let v = 0; v <= 6; v += 2) {
    line(ctx, x0, YA(v), x0 + w, YA(v), alpha(c.divider, axes), 1);
    text(ctx, `${v}×`, x0 - 8, YA(v) + 4, { size: 11, family: "mono", color: c.text3, align: "right", a: axes });
  }
  text(ctx, wide ? "COMPRESSION RATIO  ↑ better" : "COMPRESSION RATIO ↑", x0, A.y - 8, { size: 11, family: "mono", weight: 600, color: c.text2, a: axes });
  // Panel B: bytes read per random cell.
  ctx.globalAlpha = costOn;
  for (const v of [0.1, 1, 10, 100, 1000]) {
    line(ctx, x0, YB(v), x0 + w, YB(v), alpha(c.divider, 1), 1);
    text(ctx, v >= 1 ? `${v}×` : "0.1×", x0 - 8, YB(v) + 4, { size: 11, family: "mono", color: c.text3, align: "right" });
  }
  text(ctx, wide ? "BYTES READ FOR ONE RANDOM CELL  ↓ better" : "RANDOM READ COST ↓", x0, B.y - 8, { size: 11, family: "mono", weight: 600, color: c.text2 });
  if (wide) text(ctx, "× one uncompressed cell", x0 + w, B.y - 8, { size: 10, family: "mono", color: c.text3, align: "right" });
  ctx.globalAlpha = 1;
  const axisY = lerp(A.y + A.h, B.y + B.h, split);
  for (const u of [1, 8, 64, 512, 2048]) text(ctx, String(u), X(Math.log2(u)), axisY + 18, { size: 11, family: "mono", color: c.text3, align: "center", a: axes });
  text(ctx, wide ? "cells per compression unit  =  cells decoded to read one" : "cells per unit", x0 + w / 2, axisY + 36, { size: 11, family: "mono", color: c.text2, align: "center", a: axes });

  const labelsA = [];
  const labelsB = [];
  if (sweep > 0)
    CODECS.forEach((codec, i) => {
      const color = palette[i];
      const fade = 1 - beam * 0.5;
      curve(ctx, X, YA, (x) => valueAt(codec.points, x), lx, color, 2, fade);
      const cost = (x) => 2 ** x / Math.max(valueAt(codec.points, x), 0.05);
      if (sweepB > 0) curve(ctx, X, YB, cost, lxB, color, 2, fade * costOn);
      const v = valueAt(codec.points, lx);
      labelsA.push({ x: X(lx), y: YA(v), text: `${codec.name} ${v.toFixed(2)}×`, color });
      if (sweepB > 0) labelsB.push({ x: X(lxB), y: YB(cost(lxB)), text: `${cost(lxB) >= 10 ? cost(lxB).toFixed(0) : cost(lxB).toFixed(1)}×`, color });
      for (const [Y, val, on, at] of [[YA, v, 1, lx], [YB, cost(lxB), sweepB, lxB]]) {
        if (on <= 0) continue;
        ctx.beginPath();
        ctx.arc(X(at), Y(val), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    });
  for (const labels of [labelsA, labelsB]) {
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
    labels.forEach((l) => text(ctx, l.text, l.x + 8, l.y + 4, { size: 10, family: "mono", weight: 500, color: l.color, a: 1 - beam * 0.5, maxWidth: right - 10 }));
  }
  if (sweep > 0 && sweep < 1) line(ctx, X(lx), A.y, X(lx), A.y + A.h, alpha(c.text2, 0.4), 1, [3, 3]);
  if (sweepB > 0 && sweepB < 1) line(ctx, X(lxB), B.y, X(lxB), B.y + B.h, alpha(c.text2, 0.4), 1, [3, 3]);

  // The unit being decoded.
  const units = Math.round(2 ** (sweepB > 0 ? lxB : lx));
  const ub = unitH - 22;
  text(ctx, beam > 0.5 ? "CELLPACK · 1 CELL PER RECORD" : `UNIT = ${units.toLocaleString("en-US")} CELLS`, x0, box.y + 12, { size: wide ? 12 : 10, family: "mono", color: c.text2, a: axes });
  text(ctx, units > 1 ? `decode ${units.toLocaleString("en-US")} to read 1` : "decode 1 to read 1", x0 + w, box.y + 12, { size: 12, family: "mono", color: units > 1 ? c.danger : c.teal, align: "right", a: axes * (1 - beam) * (wide ? 1 : 0), maxWidth: w * 0.5 });
  strokeRound(ctx, x0, box.y + 20, w, ub, 6, alpha(c.divider, axes), 1);
  const shown = Math.min(units, 256);
  const cw = w / 256;
  for (let k = 0; k < shown; k++) {
    const wave = 0.5 + 0.5 * Math.sin(t * 3 + k * 0.4);
    ctx.fillStyle = k === 0 ? c.accent : alpha(c.text3, 0.25 + 0.2 * wave);
    ctx.fillRect(x0 + 3 + k * cw, box.y + 24, Math.max(1, cw - 0.6), ub - 8);
  }
  if (units > 256) text(ctx, `+${(units - 256).toLocaleString("en-US")}`, x0 + w - 8, box.y + 20 + ub / 2 + 4, { size: 11, family: "mono", color: c.text2, align: "right", a: 1 - beam });
  if (beam > 0) {
    fillRound(ctx, x0, box.y + 20, w, ub, 6, alpha(c.bg, beam * 0.92));
    fillRound(ctx, x0 + 3, box.y + 24, Math.max(cw, 8), ub - 8, 2, alpha(c.accent, beam));
    text(ctx, wide ? "CellPack: statistics shared by 4,096 cells · decode 1 to read 1" : "CellPack: decode 1 to read 1", x0 + 22, box.y + 20 + ub / 2 + 4, { size: wide ? 12 : 10, family: "mono", color: c.accent, a: beam, maxWidth: w - 26 });
  }

  // CellPack: one cell per unit, highest ratio and lowest read cost.
  if (drop > 0) {
    const b = ease.back(drop);
    const sx = X(0);
    const ya = YA(CELLPACK);
    const yb = YB(1 / CELLPACK);
    const burst = seg(p, 0.76, 0.86);
    if (burst > 0 && burst < 1)
      for (const y of [ya, yb]) {
        ctx.beginPath();
        ctx.arc(sx, y, 10 + burst * 60, 0, Math.PI * 2);
        ctx.strokeStyle = alpha(c.accent, 0.6 * (1 - burst));
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    glow(ctx, sx, lerp(A.y - 40, ya, b), 46, c.accent, 0.4 * drop);
    star(ctx, sx, lerp(A.y - 40, ya, b), 11, c.accent);
    star(ctx, sx, lerp(B.y - 40, yb, b), 9, c.accent);
    text(ctx, `CellPack ${CELLPACK.toFixed(2)}×`, sx + 16, ya + 24, { size: 17, family: "head", weight: 600, color: c.accent, a: clamp(drop * 2) });
    text(ctx, `reads ${(1 / CELLPACK).toFixed(2)}× of one cell`, sx + 16, yb - 12, { size: 12, family: "mono", weight: 600, color: c.accent, a: clamp(drop * 2) });
  }
  if (beam > 0) {
    line(ctx, X(0), YA(CELLPACK), lerp(X(0), x0 + w, beam), YA(CELLPACK), alpha(c.accent, 0.8), 2, [6, 5]);
    line(ctx, X(0), YB(1 / CELLPACK), lerp(X(0), x0 + w, beam), YB(1 / CELLPACK), alpha(c.accent, 0.8), 2, [6, 5]);
  }
}

export default { draw };
