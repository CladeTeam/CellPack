import home from "../../home-data.json";
import data from "../scenes.json";
import { alpha, clamp, compact, ease, fillRound, glow, lerp, line, mix, packets, seg, text } from "../engine.js";

// Single-H200 demand of eleven models against loader supply (report Figs. 1, 5).
// A lane's packets thin out past the supply line: the GPU waits for the rest.
const MODELS = [...home.models].sort((a, b) => b.rate - a.rate);
const LOADERS = home.loaders;
const BEST = Math.max(...LOADERS.filter((l) => !l.block).map((l) => l.rate));
const CP = data.cellpack.nvme_16;
const CP_MAX = data.cellpack.raid0_32;
const LOG0 = 2;
const LOG1 = 7;
const AUDIENCES = ["Foundation models", "Perturbation models", "Virtual cells"];

function draw(ctx, { p, t, box, c, wide }) {
  const labelW = wide ? 112 : 80;
  const rateW = wide ? 150 : 96;
  const top = box.y + (wide ? 104 : 118);
  const gx = box.x + labelW;
  const gw = box.w - labelW - rateW;
  const X = (v) => gx + ((Math.log10(Math.max(v, 10 ** LOG0)) - LOG0) / (LOG1 - LOG0)) * gw;
  const rowH = Math.min(48, (box.y + box.h - top - 40) / MODELS.length);
  const bh = Math.min(22, rowH * 0.5);
  const bottom = top + rowH * MODELS.length;

  const bars = seg(p, 0.02, 0.22);
  const drop = ease.out(seg(p, 0.25, 0.38));
  const sweep = ease.inOut(seg(p, 0.44, 0.5));
  const extra = ease.inOut(seg(p, 0.7, 0.74));
  const mission = ease.out(seg(p, 0.88, 0.96));
  const cpRate = sweep >= 1 ? CP : 10 ** lerp(Math.log10(BEST), Math.log10(CP), sweep);
  const supply = sweep > 0 ? Math.max(BEST, cpRate) : BEST;
  const limiting = drop > 0.5;

  for (let k = LOG0; k <= LOG1; k++) {
    const x = X(10 ** k);
    line(ctx, x, top - 6, x, bottom, alpha(c.divider, 0.8), 1);
    if (box.w >= 320 || k % 2 === 0) text(ctx, ["100", "1K", "10K", "100K", "1M", "10M"][k - LOG0], x, bottom + 18, { size: 11, family: "mono", color: c.text3, align: "center" });
  }
  text(ctx, "cells / second (log)", gx + gw, bottom + 34, { size: 11, family: "mono", color: c.text3, align: "right" });
  text(ctx, wide ? "MODEL DEMAND · 1 H200 · GPU" : "DEMAND · 1 H200", box.x, top - 18, { size: 11, family: "mono", color: c.text2, a: 1 - mission });

  MODELS.forEach((m, i) => {
    const y = top + i * rowH + rowH / 2;
    const grow = ease.out(clamp((bars - i * 0.05) / 0.45));
    const end = lerp(gx, X(m.rate), grow);
    const share = limiting ? Math.min(1, supply / m.rate) : 1;
    const starved = share < 1;
    const color = starved ? c.danger : sweep > 0 ? c.teal : c.text3;
    text(ctx, m.name, gx - 10, y + 4, { size: wide ? 13 : 10, color: c.text2, align: "right", a: clamp(grow * 3) });
    // A plain bar: the lane, with cells moving through it.
    ctx.fillStyle = alpha(color, 0.14);
    ctx.fillRect(gx, y - bh / 2, Math.max(0, end - gx), bh);
    ctx.fillStyle = color;
    ctx.fillRect(Math.max(gx, end - 2), y - bh / 2, 2, bh);
    const cut = X(supply);
    packets(ctx, {
      x0: gx, x1: end - 3, y, h: bh * 0.56, t, seed: i + 1, gap: 26, length: 1.8, radius: 1,
      speed: 40 + 18 * Math.log10(m.rate / 100), color,
      pass: (x) => (x > cut ? share : 1),
    });
    if (grow > 0.9) {
      text(ctx, compact(m.rate, 0), gx + gw + (wide ? 50 : 40), y + 4, { size: wide ? 12 : 10, family: "mono", weight: 600, color, align: "right" });
      if (limiting) {
        const idle = Math.round((1 - share) * 100);
        text(ctx, idle > 0 ? `idle ${idle}%` : "fed", gx + gw + (wide ? 60 : 46), y + 4, { size: wide ? 11 : 9, family: "mono", color: idle > 0 ? c.danger : c.teal });
      }
    }
  });

  // Existing loaders drop in as supply lines.
  LOADERS.forEach((l, i) => {
    const d = ease.out(clamp((drop - i * 0.03) / 0.6));
    if (d <= 0) return;
    const x = X(l.rate);
    const best = l.rate === BEST;
    line(ctx, x, lerp(top - 40, top - 8, d), x, lerp(top - 8, bottom, d), alpha(best ? c.text : c.text3, (best ? 0.9 : 0.3) * d * (1 - sweep * 0.6)), best ? 2 : 1, best ? [5, 4] : [2, 3]);
  });
  const starvedCount = MODELS.filter((m) => m.rate > supply).length;
  if (drop > 0) {
    const a = drop * (1 - seg(p, 0.44, 0.48));
    text(ctx, `Lance ${compact(BEST, 1)}/s · best existing`, box.x, top - 46, { size: 12, family: "mono", weight: 600, color: c.text, a, maxWidth: box.w });
    const big = Math.min(40, box.w / 15);
    text(ctx, `${starvedCount} of ${MODELS.length} models wait for data`, box.x, box.y + big * 0.9, { size: big, family: "head", color: c.danger, a });
  }

  // CellPack sweeps past every model.
  if (sweep > 0) {
    const big = Math.min(46, box.w / 12);
    const shown = seg(p, 0.44, 0.48) * (1 - seg(p, 0.84, 0.88));
    const value = extra > 0 ? lerp(CP, CP_MAX, extra) : cpRate;
    line(ctx, X(cpRate), top - 12, X(cpRate), bottom, c.accent, 3);
    glow(ctx, X(cpRate), top + (bottom - top) * (0.5 + 0.4 * Math.sin(t * 1.4)), 60, c.accent, 0.2);
    text(ctx, `CellPack ${compact(value, 2)} cells/s`, box.x, box.y + big * 0.9, { size: big, family: "head", color: c.accent, a: shown, maxWidth: box.w });
    text(ctx, extra > 0 ? "2 NVMe SSDs (RAID-0) · 32 workers" : sweep < 1 ? "1 NVMe SSD · 16 workers" : `${(CP / BEST).toFixed(1)}× the best existing loader · 1 NVMe SSD · 16 workers`, box.x, box.y + big * 0.9 + 22, { size: 11, family: "mono", color: c.text2, a: shown, maxWidth: box.w });
    if (sweep >= 1) text(ctx, "1.17M", X(CP) + 6, top - 4, { size: 11, family: "mono", weight: 600, color: c.accent });
  }
  if (extra > 0) {
    const x2 = lerp(X(CP), X(CP_MAX), extra);
    line(ctx, x2, top - 12, x2, bottom, alpha(c.accent, 0.7), 2, [6, 4]);
    text(ctx, "2.52M", x2 + 6, top - 18, { size: 11, family: "mono", weight: 600, color: c.accent, a: extra });
  }

  // Who it is for.
  if (mission > 0) {
    const big = Math.min(36, box.w / (wide ? 16 : 19));
    text(ctx, wide ? "Data supply, no longer the bottleneck." : "No longer the bottleneck.", box.x, box.y + big * 0.9, { size: big, family: "head", color: c.text, a: mission, maxWidth: box.w });
    let x = box.x;
    AUDIENCES.forEach((name, i) => {
      const a = clamp(mission * 2 - i * 0.3);
      const label = wide ? name : name.split(" ")[0];
      const w = Math.min(label.length * 7.4 + 26, (box.w - 20) / 3);
      fillRound(ctx, x, box.y + big + 14, w, 26, 13, alpha(mix(c.tealSoft, c.bg, 0.2), a));
      text(ctx, label, x + w / 2, box.y + big + 31, { size: 12, family: "mono", weight: 600, color: c.teal, align: "center", a, maxWidth: w - 10 });
      x += w + 10;
    });
  }
}


export default { draw };
