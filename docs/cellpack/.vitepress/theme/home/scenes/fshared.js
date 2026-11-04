import { alpha, clamp, ease, fillRound, font, grouped, hash, lerp, mix, seg, strokeRound, text } from "../engine.js";
import { B, STEPS, ladder } from "./fmt-common.js";

// Act 1: three shared rANS tables, a chunk of records, and one record decoded
// on its own (its coding state and value dictionary + shared tables).
// Act 2: the same cells compressed one by one instead (measured).
// Act 3: the chunk metadata this sharing costs, per cell.
const K = STEPS.contrast;
const ALTS = [
  { name: "CellPack · shared tables", short: "CellPack", v: K.shared, us: true },
  { name: "own rANS model per cell (tables free)", short: "per-cell rANS", v: K.per_cell_rans },
  { name: "own rANS model per cell, with tables", short: "+ its tables", v: K.per_cell_rans_with_tables },
  { name: "zstd-19 per cell", short: "zstd-19 / cell", v: K.zstd19_per_cell },
  { name: "zstd-3 per cell", short: "zstd-3 / cell", v: K.zstd3_per_cell },
];
const PARTS = [
  ["gene order", STEPS.metadata_parts.dims],
  ["offsets", STEPS.metadata_parts.offsets],
  ["nnz totals", STEPS.metadata_parts.nnz],
  ["lengths", STEPS.metadata_parts.lengths],
  ["column modes", STEPS.metadata_parts.modes],
  ["record kinds", STEPS.metadata_parts.kinds],
  ["rANS tables", STEPS.metadata_parts.models],
];
const TABLES = ["gaps", "ranks", "dict. bytes"];
// Two rows of records with varied lengths.
const RECS = (() => {
  const out = [];
  for (let r = 0; r < 2; r++) {
    let u = 0;
    for (let k = 0; u < 0.995; k++) {
      const w = Math.min(1 - u, 0.035 + hash(r * 40 + k, 9) * 0.055);
      out.push({ row: r, u, w });
      u += w;
    }
  }
  return out;
})();

function draw(ctx, env) {
  const { p, t, box, c, wide } = env;
  const intro = ease.out(seg(p, 0, 0.08));
  const act1 = intro * (1 - ease.inOut(seg(p, 0.34, 0.4)));
  const act2 = ease.inOut(seg(p, 0.38, 0.44));
  const race = seg(p, 0.42, 0.66);
  const meta = ease.inOut(seg(p, 0.74, 0.86));
  const y0 = ladder(ctx, env, B["5_values_dictionary"], B["7_plus_chunk_metadata"], meta);
  const area = { y: y0 + 16, h: box.y + box.h - y0 - 16 };

  if (act1 > 0) {
    const cycle = t / 6.4;
    const active = Math.floor(cycle) % RECS.length;
    const phase = cycle % 1;
    const pulse = Math.sin(Math.PI * seg(phase, 0, 0.12)) ** 2;
    const lookupStart = (i) => 0.12 + i * 0.1;
    const lookupAt = (i) => {
      const start = lookupStart(i);
      return ease.out(seg(phase, start, start + 0.03)) * (1 - ease.inOut(seg(phase, start + 0.11, start + 0.16)));
    };
    // Shared models.
    const tw = (box.w - 24) / 3;
    const th = Math.min(96, area.h * 0.2);
    const gapR = Math.max(80, area.h * 0.12);
    const used = 24 + th + gapR + 74 + 20 + 64 + 20 + 50;
    const ty = area.y + 24 + Math.max(0, (area.h - used) / 2);
    text(ctx, "SHARED BY THE 4,096 CELLS OF A CHUNK", box.x, ty - 10, { size: 11, family: "mono", weight: 700, color: c.teal, a: act1 });
    if (wide) text(ctx, `3 × 512 B of tables → ${STEPS.metadata_parts.models.toFixed(2)} B per cell`, box.x + box.w, ty - 10, { size: 11, family: "mono", color: c.text2, align: "right", a: act1 });
    const models = TABLES.map((name, i) => {
      const x = box.x + i * (tw + 12);
      fillRound(ctx, x, ty, tw, th, 8, alpha(c.tealSoft, act1));
      text(ctx, name, x + 12, ty + 20, { size: wide ? 12 : 10, family: "mono", weight: 600, color: c.teal, a: act1, maxWidth: tw - 24 });
      const bars = 18;
      const barW = (tw - 24) / bars;
      const slot = Math.floor(hash(active, i + 71) * 8);
      const focus = lookupAt(i);
      const hit = focus * ease.out(seg(phase, lookupStart(i) + 0.08, lookupStart(i) + 0.1));
      const slotX = x + 12 + slot * barW;
      fillRound(ctx, slotX - 2, ty + 30, barW + 2, th - 36, 2, alpha(c.accent, 0.16 * hit * act1));
      for (let k = 0; k < bars; k++) {
        const h = (th - 38) * Math.exp(-k * (0.22 + i * 0.08)) * (0.8 + 0.2 * hash(k, i));
        ctx.fillStyle = alpha(k === slot ? mix(c.teal, c.accent, hit) : c.teal, (0.75 + (k === slot ? hit * 0.25 : 0)) * act1);
        ctx.fillRect(x + 12 + k * barW, ty + th - 10 - h, barW - 2, h);
      }
      strokeRound(ctx, slotX - 2, ty + 30, barW + 2, th - 36, 2, alpha(c.accent, hit * act1), 1.2);
      return { cx: slotX + (barW - 2) / 2, by: ty + th, focus };
    });

    // The chunk's records.
    const fy = ty + th + gapR;
    const rh = 22;
    const fh = 2 * rh + 6 + 24;
    strokeRound(ctx, box.x, fy, box.w, fh, 8, alpha(c.border, act1), 1);
    text(ctx, "CHUNK · RECORDS", box.x + 10, fy + 15, { size: 10, family: "mono", weight: 700, color: c.text3, a: act1 });
    const rx0 = box.x + 10;
    const rw = box.w - 20;
    const recY = (row) => fy + 22 + row * (rh + 6);
    RECS.forEach((r, i) => {
      const x = rx0 + r.u * rw;
      const w = r.w * rw - 3;
      const on = i === active;
      ctx.fillStyle = alpha(on ? c.accent : mix(c.tealSoft, c.teal, 0.45), act1 * (on ? 1 : 0.8));
      ctx.fillRect(x, recY(r.row), w, rh);
      ctx.fillStyle = alpha(on ? mix(c.accent, c.text, 0.45) : mix(c.teal, c.text, 0.5), act1);
      ctx.fillRect(x, recY(r.row), 4, rh);
      if (on) strokeRound(ctx, x - 2, recY(r.row) - 2, w + 4, rh + 4, 3, alpha(c.accent, act1 * (0.65 + 0.35 * pulse)), 1.5 + pulse);
    });
    // Lookup paths stay above the chunk so they never cross neighbouring records.
    const ar = RECS[active];
    const ax = rx0 + ar.u * rw + (ar.w * rw - 3) / 2;
    const ay = fy - 5;
    models.forEach((m, i) => {
      if (m.focus <= 0) return;
      const arrive = ease.inOut(seg(phase, lookupStart(i), lookupStart(i) + 0.1));
      const endY = m.by + 4;
      const revealY = lerp(ay, endY, arrive);
      ctx.save();
      ctx.beginPath();
      ctx.rect(box.x - 4, revealY - 4, box.w + 8, ay - revealY + 8);
      ctx.clip();
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.bezierCurveTo(ax, ay - gapR * 0.45, m.cx, endY + gapR * 0.45, m.cx, endY);
      ctx.strokeStyle = alpha(c.teal, 0.65 * m.focus * act1);
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 5]);
      ctx.stroke();
      ctx.restore();
      ctx.beginPath();
      ctx.moveTo(m.cx - 3, endY + 5);
      ctx.lineTo(m.cx, endY);
      ctx.lineTo(m.cx + 3, endY + 5);
      ctx.strokeStyle = alpha(c.teal, seg(arrive, 0.85, 1) * m.focus * act1);
      ctx.lineWidth = 1.2;
      ctx.stroke();
    });
    const caption = phase < 0.12 ? `Select cell ${active}` : phase < 0.5 ? "Look up shared tables" : phase < 0.78 ? "Look up this cell's dictionary" : phase < 0.94 ? "Decode this cell" : "Cell decoded";
    const captionSize = wide ? 12 : 10;
    font(ctx, captionSize, "mono", 600);
    const captionW = ctx.measureText(caption).width;
    const captionX = box.x + box.w / 2;
    const captionY = ty + th + gapR / 2;
    fillRound(ctx, captionX - captionW / 2 - 10, captionY - 14, captionW + 20, 23, 5, alpha(c.bg, act1));
    text(ctx, caption, captionX, captionY + 2, { size: captionSize, family: "mono", weight: 600, color: phase < 0.5 ? c.teal : c.accent, align: "center", a: act1 });

    // The selected record owns this rank-to-value dictionary; the chunk's
    // probability tables decode its encoded bytes, not its rank lookups.
    const localY = fy + fh + 20;
    const localH = 64;
    const dict = [1, 2, 3, 5].map((count) => Math.log1p(count + active % 3));
    const entryW = (box.w - 20 - 6 * (dict.length - 1)) / dict.length;
    const entryX = (rank) => box.x + 10 + rank * (entryW + 6);
    const reveal = ease.out(seg(phase, 0.48, 0.54));
    const localFocus = ease.out(seg(phase, 0.52, 0.54)) * (1 - ease.inOut(seg(phase, 0.74, 0.78)));
    const rank = Math.min(dict.length - 1, Math.floor(seg(phase, 0.52, 0.76) * dict.length));
    fillRound(ctx, box.x, localY, box.w, localH, 8, alpha(c.surface, act1));
    strokeRound(ctx, box.x, localY, box.w, localH, 8, alpha(c.accent, act1 * (0.25 + 0.45 * reveal)), 1);
    text(ctx, `CELL ${active} · VALUE DICTIONARY`, box.x + 10, localY + 16, { size: wide ? 10 : 9, family: "mono", weight: 600, color: c.accent, a: act1 });
    if (wide) text(ctx, "rank → value · stored in this record", box.x + box.w - 10, localY + 16, { size: 10, family: "mono", color: c.text3, align: "right", a: act1 });
    dict.forEach((value, i) => {
      const hit = i === rank ? localFocus : 0;
      fillRound(ctx, entryX(i), localY + 26, entryW, 26, 4, alpha(c.accent, act1 * (0.06 + 0.1 * reveal + 0.15 * hit)));
      strokeRound(ctx, entryX(i), localY + 26, entryW, 26, 4, alpha(c.accent, act1 * hit), 1.4);
      text(ctx, `${i} → ${value.toFixed(2)}`, entryX(i) + entryW / 2, localY + 43, { size: wide ? 12 : 9, family: "mono", weight: hit > 0 ? 600 : 400, color: c.text, align: "center", a: act1 * reveal, maxWidth: entryW - 8 });
    });
    if (localFocus > 0) {
      const endX = entryX(rank) + entryW / 2;
      const startY = fy + fh + 2;
      const endY = localY - 3;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(ax, startY);
      ctx.bezierCurveTo(ax, startY + 8, endX, endY - 8, endX, endY);
      ctx.strokeStyle = alpha(c.accent, 0.75 * localFocus * act1);
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(endX - 3, endY - 5);
      ctx.lineTo(endX, endY);
      ctx.lineTo(endX + 3, endY - 5);
      ctx.stroke();
      ctx.restore();
    }
    // Decoded output: only this cell's values.
    const oy = localY + localH + 20;
    const n = 14;
    const ow = Math.min(34, (box.w - 80 - 3 * n) / n);
    const filled = seg(phase, 0.78, 0.92) * n;
    text(ctx, "decode →", box.x, oy + 16, { size: 12, family: "mono", weight: 700, color: c.accent, a: act1 });
    for (let k = 0; k < n; k++) fillRound(ctx, box.x + 80 + k * (ow + 3), oy, ow, 22, 3, alpha(c.accent, act1 * lerp(0.12, 0.85, clamp(filled - k))));
    text(ctx, wide ? "this cell only · its own dictionary + shared models · no neighbours decoded" : "own dictionary · no neighbours", box.x + 80, oy + 42, { size: wide ? 12 : 9, family: "mono", color: c.text2, a: act1 });
    if (wide) text(ctx, "▌ = the record's own rANS state", box.x + box.w, fy + 15, { size: 10, family: "mono", color: c.text3, align: "right", a: act1 });
  }

  if (act2 > 0) {
    // Same cells, compressed one by one.
    const chartH = Math.min(area.h * 0.52, 300);
    const top = area.y + 30 + Math.max(0, (area.h - (30 + chartH + 54 + 90)) / 2);
    const rowH = chartH / ALTS.length;
    const labelW = wide ? 290 : 104;
    const barX = box.x + labelW;
    const barW = box.w - labelW - (wide ? 140 : 130);
    const max = Math.max(...ALTS.map((r) => r.v));
    text(ctx, wide ? "BYTES PER CELL · EVERY CELL DECODABLE ON ITS OWN" : "BYTES / INDEPENDENT CELL", box.x, top - 12, { size: 11, family: "mono", weight: 700, color: c.text2, a: act2 });
    ALTS.forEach((r, i) => {
      const y = top + i * rowH;
      const v = Math.min(r.v, max * ease.inOut(race));
      const bh = Math.min(28, rowH * 0.6);
      text(ctx, wide ? r.name : r.short, barX - 12, y + rowH / 2 + 4, { size: wide ? 13 : 10, weight: r.us ? 700 : 500, color: r.us ? c.teal : c.text2, align: "right", a: act2 });
      fillRound(ctx, barX, y + (rowH - bh) / 2, Math.max(2, (v / max) * barW), bh, 4, alpha(r.us ? c.teal : c.text3, (r.us ? 0.9 : 0.45) * act2));
      text(ctx, `${grouped(v)} B`, barX + (v / max) * barW + 10, y + rowH / 2 + 4, { size: 12, family: "mono", weight: 600, color: r.us ? c.teal : c.text2, a: act2 });
      if (!r.us) text(ctx, `${(r.v / K.shared).toFixed(2)}×`, box.x + box.w, y + rowH / 2 + 4, { size: 13, family: "head", weight: 700, color: c.danger, align: "right", a: act2 * seg(p, 0.64, 0.68) });
    });

    // Act 3: chunk metadata, amortized per cell.
    if (meta > 0) {
      const my = top + chartH + 54;
      const total = PARTS.reduce((a, [, v]) => a + v, 0);
      text(ctx, `+ ${STEPS.metadata_per_cell.toFixed(1)} B per cell of chunk metadata`, box.x, my - 12, { size: wide ? 13 : 11, family: "mono", weight: 700, color: c.accent, a: meta });
      if (wide) text(ctx, `${((STEPS.metadata_per_cell / B["7_plus_chunk_metadata"]) * 100).toFixed(1)}% of the chunk`, box.x + box.w, my - 12, { size: 12, family: "mono", color: c.text2, align: "right", a: meta });
      let x = box.x;
      const legend = [];
      PARTS.forEach(([name, v], i) => {
        const w = (v / total) * box.w * meta;
        const col = alpha(c.accent, 0.9 - i * 0.11);
        fillRound(ctx, x, my, Math.max(0, w - 3), 18, 3, col);
        legend.push([name, v, col]);
        x += w;
      });
      // Legend in two columns, so every part is named.
      const columns = Math.max(2, Math.min(4, Math.floor(box.w / 190)));
      const lw = box.w / columns;
      legend.forEach(([name, v, col], i) => {
        const lx = box.x + (i % columns) * lw;
        const ly = my + 38 + Math.floor(i / columns) * 20;
        ctx.fillStyle = col;
        ctx.fillRect(lx, ly - 9, 10, 10);
        text(ctx, `${name} ${v.toFixed(1)} B`, lx + 16, ly, { size: 11, family: "mono", color: c.text2, a: meta });
      });
    }
  }
}

export default { draw };
