import data from "../scenes.json";
import { alpha, clamp, ease, fillRound, glow, lerp, line, mix, seg, strokeRound, text } from "../engine.js";

// Each loader's own storage (bytes stored per cell) above, and the device
// reads it needs to deliver one random cell below (report Fig. 2a). Pages
// fall from each loader's storage into its pile; one square = 4 KiB.
const PAGE = 4.096;
const SHORT = { "HF Datasets": "HF", annbatch: "annb.", memmap: "mmap", Parquet: "Parq." };
const METHODS = data.reads.map((m) => ({ ...m, pages: m.read_kb / PAGE, needed: m.stored_kb / PAGE, cellpack: m.name === "CellPack" }));
const MAX_READ = Math.max(...METHODS.map((m) => m.pages));
const MAX_STORED = Math.max(...METHODS.map((m) => m.stored_kb));
const FLIGHT = 10;

function layout(box) {
  const n = METHODS.length;
  const narrow = box.w < 600;
  const label = narrow ? 58 : 40;
  const gap = Math.max(5, box.w * 0.016);
  const cw = (box.w - gap * (n - 1)) / n;
  const storeTop = box.y + 4;
  const storeH = Math.min(110, box.h * 0.18);
  const top = storeTop + storeH + 100;
  const bottom = box.y + box.h - label;
  let best = { ps: 0, per: 1 };
  for (let per = 2; per <= 16; per++) {
    const ps = Math.min(cw / per, (bottom - top) / Math.ceil(MAX_READ / per));
    if (ps > best.ps) best = { ps, per };
  }
  return { n, gap, cw, ps: best.ps, per: best.per, top, bottom, storeTop, storeH, narrow };
}

function draw(ctx, { p, t, box, c }) {
  const L = layout(box);
  const intro = ease.out(seg(p, 0, 0.08));
  const fall = seg(p, 0.1, 0.7);
  const reveal = ease.inOut(seg(p, 0.78, 0.86));
  const fs = L.narrow ? 9 : 12;

  const landed = fall * (MAX_READ + FLIGHT) * 1.02;
  METHODS.forEach((m, i) => {
    const x0 = box.x + i * (L.cw + L.gap);
    const dim = m.cellpack ? 0 : reveal * 0.6;
    const a = 1 - dim;
    // CellPack is visible from the start in teal, highlighted at the end.
    const mine = m.cellpack ? mix(c.teal, c.accent, reveal) : null;
    // This loader's storage block: height = stored bytes per cell.
    const sh = (m.stored_kb / MAX_STORED) * L.storeH * intro;
    const sy = L.storeTop;
    const shimmer = 0.75 + 0.25 * Math.sin(t * 2 + i);
    fillRound(ctx, x0, sy, L.cw, sh, 4, alpha(mine || mix(c.text3, c.teal, 0.25), (m.cellpack ? 0.7 : 0.35) * a * shimmer));
    strokeRound(ctx, x0, sy, L.cw, sh, 4, alpha(mine || c.text3, 0.8 * a), 1);

    // Pages falling from this storage into the read pile.
    const total = Math.ceil(m.pages);
    const shown = Math.min(total, Math.max(0, landed - FLIGHT));
    for (let k = 0; k < Math.min(total, Math.ceil(landed)); k++) {
      const col = k % L.per;
      const row = Math.floor(k / L.per);
      const tx = x0 + col * L.ps;
      const ty = L.bottom - (row + 1) * L.ps;
      const flight = clamp((landed - k) / FLIGHT);
      if (flight <= 0) continue;
      const fy = lerp(sy + sh, ty, ease.in(flight));
      const fx = lerp(x0 + L.cw / 2 - L.ps / 2, tx, ease.in(flight));
      const partial = m.pages < 1.5 && k === total - 1 ? m.pages - Math.floor(m.pages) : 1;
      const needed = k < m.needed;
      const color = needed ? mine || c.teal : mix(c.bg, c.text3, 0.4);
      ctx.globalAlpha = a * (flight < 1 ? 0.55 + flight * 0.45 : 1);
      ctx.fillStyle = color;
      ctx.fillRect(fx + 0.5, fy + 0.5, Math.max(1, (L.ps - 1) * partial), L.ps - 1);
    }
    ctx.globalAlpha = 1;

    const kb = Math.min(m.read_kb, (shown / Math.max(m.pages, 1e-9)) * m.read_kb);
    const height = Math.ceil(shown / L.per) * L.ps;
    const ty = L.bottom - height - 8;
    const done = shown >= m.pages;
    const unit = L.narrow ? "" : " KB";
    text(ctx, `${kb >= 10 ? kb.toFixed(0) : kb.toFixed(1)}${unit}`, x0 + L.cw / 2, ty, { size: fs, family: "mono", weight: m.cellpack ? 700 : 400, color: mine || c.text2, align: "center", a: (shown > 0 ? 1 : 0) * a });
    if (done) {
      const amp = clamp((landed - FLIGHT - m.pages) / 8) * a;
      text(ctx, `${Math.round(m.amp)}×`, x0 + L.cw / 2, ty - fs - 4, { size: fs + 2, family: "head", weight: 600, color: mine || c.danger, align: "center", a: amp });
    }
    // Loader name, rotated on narrow screens.
    ctx.save();
    ctx.translate(x0 + L.cw / 2, L.bottom + 14);
    if (L.narrow) ctx.rotate(-Math.PI / 4);
    text(ctx, L.narrow ? SHORT[m.name] || m.name : m.name, 0, L.narrow ? 4 : 0, { size: L.narrow ? 10 : 12, weight: m.cellpack ? 700 : 500, color: mine || c.text2, align: L.narrow ? "right" : "center", a: Math.max(0.15, a) });
    ctx.restore();
  });
  // Stored sizes, drawn above the falling pages.
  METHODS.forEach((m, i) => {
    const x0 = box.x + i * (L.cw + L.gap);
    const sh = (m.stored_kb / MAX_STORED) * L.storeH * intro;
    const mine = m.cellpack ? mix(c.teal, c.accent, reveal) : null;
    const a = (m.cellpack ? 1 : 1 - reveal * 0.6) * intro;
    const label = m.stored_kb.toFixed(1);
    const w = label.length * fs * 0.62 + 10;
    fillRound(ctx, x0 + L.cw / 2 - w / 2, L.storeTop + sh + 3, w, fs + 6, 4, alpha(c.bg, 0.92 * a));
    text(ctx, label, x0 + L.cw / 2, L.storeTop + sh + fs + 5, { size: fs, family: "mono", weight: m.cellpack ? 700 : 400, color: mine || c.text2, align: "center", a });
  });
  line(ctx, box.x, L.bottom + 1, box.x + box.w, L.bottom + 1, c.border, 1);
  text(ctx, L.narrow ? "▲ KB stored per cell" : "▲ each loader's storage: KB stored per cell", box.x, L.storeTop + L.storeH + 36, { size: 11, family: "mono", color: c.text3, a: intro });
  text(ctx, L.narrow ? "▼ KB read per random cell" : "▼ KB read from storage to deliver one random cell", box.x, L.storeTop + L.storeH + 52, { size: 11, family: "mono", color: c.text3, a: intro });

  if (reveal > 0) {
    const i = METHODS.findIndex((m) => m.cellpack);
    const x = box.x + i * (L.cw + L.gap) + L.cw / 2;
    glow(ctx, x, L.bottom - L.ps, 80, c.accent, 0.3 * reveal);
    glow(ctx, x, L.storeTop + 10, 60, c.accent, 0.25 * reveal);
    const big = Math.min(54, box.w / 10);
    const right = box.x + box.w;
    const y = L.top + (L.bottom - L.top) * 0.42;
    text(ctx, "2.0 KB stored", right, y, { size: big * 0.62, family: "head", color: c.accent, align: "right", a: reveal });
    text(ctx, "4.4 KB read", right, y + big * 0.9, { size: big, family: "head", color: c.accent, align: "right", a: reveal });
    text(ctx, "CellPack, per random cell", right, y + big * 0.9 + 24, { size: 12, family: "mono", color: c.text2, align: "right", a: reveal });
  }
}

export default { draw };
