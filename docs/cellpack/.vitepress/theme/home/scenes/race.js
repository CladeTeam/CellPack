import home from "../../home-data.json";
import data from "../scenes.json";
import { alpha, compact, ease, fillRound, glow, lerp, line, packets, seg, text } from "../engine.js";

// Loader throughput on real data (report Fig. 5), on a linear axis so the
// distance is shown at true scale. Baselines: one NVMe SSD, 16 workers,
// 20 GB cache, best variant each.
const CP = data.cellpack;
const ROWS = [
  { name: "CellPack · RAID-0 · 32 workers", rate: CP.raid0_32, cellpack: true },
  { name: "CellPack · 1 NVMe · 16 workers", rate: CP.nvme_16, cellpack: true },
  ...[...home.loaders].sort((a, b) => b.rate - a.rate).map((l) => ({ name: l.name, rate: l.rate })),
];
const MAX = 2.6e6;
const BEST = Math.max(...home.loaders.filter((l) => !l.block).map((l) => l.rate));

function draw(ctx, { p, t, box, c, wide }) {
  const labelW = wide ? 210 : Math.min(146, box.w * 0.47);
  const rateW = wide ? 70 : 50;
  const gx = box.x + labelW;
  const gw = box.w - labelW - rateW;
  const X = (v) => gx + (v / MAX) * gw;
  // The axis sits above the bars, next to the fastest rows.
  const top = box.y + 56;
  const rowH = Math.min(38, (box.h - 70) / ROWS.length);
  const bh = rowH * 0.56;
  const run = seg(p, 0.08, 0.62);
  const zoom = ease.inOut(seg(p, 0.72, 0.84));
  const intro = ease.out(seg(p, 0, 0.06));

  for (let v = 0; v <= 2.5e6; v += 5e5) {
    line(ctx, X(v), top - 6, X(v), top + rowH * ROWS.length, alpha(c.divider, intro), 1);
    text(ctx, v ? `${v / 1e6}M` : "0", X(v), top - 14, { size: 11, family: "mono", color: c.text3, align: "center", a: intro });
  }
  text(ctx, "cells / second", gx, top - 34, { size: 11, family: "mono", color: c.text3, a: intro });

  // Every bar grows at the same speed and stops at its own value.
  const front = ease.inOut(run) * MAX;
  ROWS.forEach((r, i) => {
    const y = top + i * rowH + (rowH - bh) / 2;
    const v = Math.min(r.rate, front);
    const done = front >= r.rate;
    const color = r.cellpack ? c.accent : c.text3;
    text(ctx, r.name, gx - 10, y + bh / 2 + 4, { size: wide ? 12 : 9, weight: r.cellpack ? 700 : 500, color: r.cellpack ? c.accent : c.text2, align: "right", a: intro, maxWidth: labelW - 10 });
    fillRound(ctx, gx, y, Math.max(2, X(v) - gx), bh, 3, alpha(color, r.cellpack ? 0.9 : 0.55));
    if (r.cellpack && X(v) - gx > 40) packets(ctx, { x0: gx + 4, x1: X(v) - 4, y: y + bh / 2, h: bh * 0.4, t, seed: i + 3, gap: 26, speed: 160, color: c.bg });
    text(ctx, compact(v, r.rate >= 1e6 ? 2 : 1), gx + gw + rateW, y + bh / 2 + 4, { size: wide ? 12 : 10, family: "mono", weight: done ? 600 : 400, color: r.cellpack ? c.accent : c.text2, align: "right", a: intro, maxWidth: labelW - 10 });
    if (r.cellpack && done) glow(ctx, X(r.rate), y + bh / 2, 40, c.accent, 0.3 * (0.6 + 0.4 * Math.sin(t * 3 + i)));
  });

  // Bracket: CellPack versus the best existing loader on the same hardware.
  if (zoom > 0) {
    // The empty space right of the baseline bars holds the comparison.
    const y = top + rowH * 7.5;
    const xb = X(BEST);
    const xc = X(CP.nvme_16);
    const yb = top + rowH * (ROWS.findIndex((r) => r.rate === BEST) + 0.5);
    const yc = top + rowH * 1.5;
    line(ctx, xb, yb, xb, y, alpha(c.text, zoom), 1.5, [4, 3]);
    line(ctx, xc, yc, xc, y, alpha(c.accent, zoom), 1.5, [4, 3]);
    line(ctx, xb, y, lerp(xb, xc, zoom), y, c.accent, 2);
    const big = Math.min(48, box.w / 12);
    text(ctx, `${(CP.nvme_16 / BEST).toFixed(1)}×`, (xb + xc) / 2, y - 10, { size: big, family: "head", weight: 600, color: c.accent, align: "center", a: zoom });
    text(ctx, wide ? "same SSD, same 16 cores" : "same SSD / 16 cores", (xb + xc) / 2, y + 18, { size: wide ? 11 : 10, family: "mono", color: c.text2, align: "center", a: zoom, maxWidth: gw });
  }
}

export default { draw };
