import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, seg, strokeRound, text, paragraph } from "../engine.js";
import { B, CSR, FINAL, ladder } from "./fmt-common.js";

// Records laid end to end cross page boundaries; packed, every record fits
// within one 4 KiB page, at the cost of some padding. Sizes are illustrative.
const PAGE = 4096;
const SIZES = Array.from({ length: 18 }, (_, i) => Math.round(900 + hash(i, 13) * 2700));
const NAIVE = SIZES.reduce((a, s) => [...a, a.at(-1) + s], [0]);
const PACKED = (() => {
  const out = [];
  let page = 0;
  let used = 0;
  for (const s of SIZES) {
    if (used + s > PAGE) {
      page++;
      used = 0;
    }
    out.push({ page, off: used });
    used += s;
  }
  return out;
})();
const PAGES = PACKED.at(-1).page + 1;
const CROSS = SIZES.findIndex((s, i) => Math.floor(NAIVE[i] / PAGE) !== Math.floor((NAIVE[i] + s - 1) / PAGE));
const PICK = 11;

function draw(ctx, env) {
  const { p, t, box, c, wide } = env;
  const intro = ease.out(seg(p, 0, 0.08));
  const straddle = seg(p, 0.1, 0.22);
  const pack = seg(p, 0.26, 0.48);
  const dir = ease.out(seg(p, 0.52, 0.66));
  const read = seg(p, 0.7, 0.92);
  const y0 = ladder(ctx, env, B["7_plus_chunk_metadata"], B["8_packed_4k_pages"], ease.inOut(pack));

  const cols = 4;
  const gap = 12;
  const pw = (box.w - gap * (cols - 1)) / cols;
  const ph = Math.min(36, Math.max(18, (box.h - 330) / 4));
  const rowGap = 24;
  const py0 = y0 + 126;
  const pagePos = (k) => ({ x: box.x + (k % cols) * (pw + gap), y: py0 + Math.floor(k / cols) * (ph + rowGap) });
  const xOf = (off) => {
    const k = Math.floor(off / PAGE);
    const { x, y } = pagePos(k);
    return { x: x + ((off % PAGE) / PAGE) * pw, y, k };
  };

  // Page frames.
  for (let k = 0; k < PAGES; k++) {
    const { x, y } = pagePos(k);
    fillRound(ctx, x, y, pw, ph, 6, alpha(c.surface, intro));
    strokeRound(ctx, x, y, pw, ph, 6, alpha(c.border, intro), 1);
    text(ctx, `p${k} · 4 KiB`, x, y + ph + 14, { size: 9, family: "mono", color: c.text3, a: intro });
  }

  // Records: naive positions → packed positions.
  SIZES.forEach((s, i) => {
    const local = ease.inOut(clamp((pack - (i / SIZES.length) * 0.4) / 0.6));
    const naive = xOf(NAIVE[i]);
    const tgt = pagePos(PACKED[i].page);
    const tx = tgt.x + (PACKED[i].off / PAGE) * pw;
    const w = (s / PAGE) * pw;
    const crossing = i === CROSS && local < 0.5;
    const picked = read > 0 && i === PICK;
    const dim = read > 0.2 && !picked ? 0.35 : 1;
    let color = picked ? c.accent : alpha(c.teal, 0.45 + 0.4 * hash(i, 4));
    if (crossing && straddle > 0) color = alpha(c.danger, 0.4 + 0.6 * straddle);
    ctx.globalAlpha = intro * dim;
    if (local === 0) {
      // End-to-end layout: a record can be split by a page boundary.
      let off = NAIVE[i];
      let left = s;
      while (left > 0) {
        const room = PAGE - (off % PAGE);
        const part = Math.min(room, left);
        const at = xOf(off);
        fillRound(ctx, at.x + 1, at.y + 5, (part / PAGE) * pw - 2, ph - 10, 3, color);
        off += part;
        left -= part;
      }
    } else {
      const x = lerp(naive.x, tx, local);
      const y = lerp(naive.y, tgt.y, local) - Math.sin(local * Math.PI) * 30;
      fillRound(ctx, x + 1, y + 5, w - 2, ph - 10, 3, color);
    }
    ctx.globalAlpha = 1;
  });
  if (straddle > 0 && pack < 0.3) {
    const at = xOf(NAIVE[CROSS]);
    text(ctx, "crosses a page boundary → 2 page reads", at.x, at.y - 10, { size: 12, family: "mono", weight: 600, color: c.danger, a: straddle * (1 - pack * 3) });
  }
  // Padding after packing.
  if (pack > 0.6) {
    const a = seg(pack, 0.6, 1);
    for (let k = 0; k < PAGES; k++) {
      const last = PACKED.map((r, i) => (r.page === k ? r.off + SIZES[i] : 0)).reduce((m, v) => Math.max(m, v), 0);
      const { x, y } = pagePos(k);
      const px = x + (last / PAGE) * pw;
      ctx.save();
      ctx.beginPath();
      ctx.rect(px, y + 5, x + pw - px - 2, ph - 10);
      ctx.clip();
      for (let hx = px - ph; hx < x + pw; hx += 6) line(ctx, hx, y + ph - 5, hx + ph, y + 5, alpha(c.text3, 0.35 * a), 1);
      ctx.restore();
    }
    text(ctx, `padding +${Math.round(B["8_packed_4k_pages"] - B["7_plus_chunk_metadata"])} B/cell`, box.x + box.w, py0 - 8, { size: 11, family: "mono", color: c.text3, align: "right", a });
  }

  // Page reads per random cell: a record of mean size L crosses a boundary
  // with probability (L − 1) / 4096 when laid end to end; packed, never.
  if (straddle > 0) {
    const mean = B["5_values_dictionary"];
    const before = 1 + (mean - 1) / PAGE;
    const shown = lerp(before, 1, ease.inOut(pack));
    const a = straddle;
    const rx = box.x + box.w;
    text(ctx, "PAGE READS / CELL", rx, y0 + 8, { size: 11, family: "mono", weight: 600, color: c.text2, align: "right", a });
    text(ctx, `${before.toFixed(2)} → ${shown.toFixed(2)}`, rx, y0 + 34, { size: wide ? 26 : 20, family: "head", weight: 600, color: pack > 0.5 ? c.teal : c.danger, align: "right", a });
    text(ctx, `mean record ${Math.round(mean).toLocaleString("en-US")} B on 4 KiB pages`, rx, y0 + 50, { size: 10, family: "mono", color: c.text3, align: "right", a });
  }

  // In-chunk directory: offsets point at records out of row order.
  const dy = y0 + 82;
  if (dir > 0) {
    const ew = Math.min(28, box.w / SIZES.length - 3);
    text(ctx, "DIRECTORY · row → offset", box.x, dy - 8, { size: 11, family: "mono", weight: 600, color: c.text2, a: dir });
    SIZES.forEach((_, i) => {
      const x = box.x + i * (ew + 3);
      const hi = read > 0 && i === PICK;
      fillRound(ctx, x, dy, ew, 18, 3, alpha(hi ? c.accent : c.text3, (hi ? 1 : 0.5) * dir));
      if (i % 3 === 0 && !hi && read === 0) {
        const tgt = pagePos(PACKED[i].page);
        line(ctx, x + ew / 2, dy + 18, tgt.x + (PACKED[i].off / PAGE) * pw + 6, tgt.y + 4, alpha(c.text3, 0.25 * dir), 1);
      }
    });
  }

  // A random read: one entry, one page, one record.
  if (read > 0) {
    const ew = Math.min(28, box.w / SIZES.length - 3);
    const dx = box.x + PICK * (ew + 3) + ew / 2;
    const tgt = pagePos(PACKED[PICK].page);
    const rx = tgt.x + (PACKED[PICK].off / PAGE) * pw;
    const a1 = ease.out(seg(read, 0, 0.3));
    line(ctx, dx, dy + 18, lerp(dx, rx + 8, a1), lerp(dy + 18, tgt.y, a1), c.accent, 2, [5, 4]);
    const a2 = seg(read, 0.25, 0.5);
    strokeRound(ctx, tgt.x - 3, tgt.y - 3, pw + 6, ph + 6, 8, alpha(c.accent, a2), 2.5);
    glow(ctx, tgt.x + pw / 2, tgt.y + ph / 2, pw * 0.7, c.accent, 0.2 * a2 * (0.7 + 0.3 * Math.sin(t * 4)));
    const a3 = ease.out(seg(read, 0.5, 0.85));
    const last = pagePos(PAGES - 1);
    const vy = last.y + ph + 34;
    const cw = Math.min(50, (box.w - 42) / 7);
    for (let k = 0; k < 7; k++) {
      const x = lerp(rx, box.x + (box.w - (7 * cw + 36)) / 2 + k * (cw + 6), a3);
      const y = lerp(tgt.y + 5, vy, a3);
      fillRound(ctx, x, y, cw, 26, 4, alpha(c.accentSoft, a3));
      text(ctx, (Math.log1p(1 + ((k * 7) % 5))).toFixed(2), x + cw / 2, y + 17, { size: 11, family: "mono", color: c.accent, align: "center", a: a3 });
    }
    text(ctx, "1 page → 1 record decoded", box.x + box.w / 2, vy + 47, { size: 11, family: "mono", weight: 600, color: c.accent, align: "center", a: a3 });
    text(ctx, `${Math.round(FINAL).toLocaleString("en-US")} B/cell · ${(CSR / FINAL).toFixed(2)}× smaller`, box.x + box.w / 2, vy + 66, { size: 11, family: "mono", color: c.teal, align: "center", a: a3 });
  }
}

export default { draw };
