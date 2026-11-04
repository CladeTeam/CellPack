import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, mix, seg, strokeRound, text, paragraph } from "../engine.js";

// A 48 × 96 slice of the cell-by-gene matrix at ~5% density: one cell, a
// dataset and a corpus; then CSR built in two moves (pack each row's nonzeros
// to its left end, then concatenate the rows into arrays), chunked, compressed
// and read back at random.
const R = 48;
const C = 96;
const ROWS = Array.from({ length: R }, (_, r) => {
  const cols = [];
  for (let c = 0; c < C; c++) if (hash(r + 11, c + 3) < 0.05) cols.push(c);
  for (let k = 0; cols.length < 3; k++) {
    const c = Math.floor(hash(r, k + 500) * C);
    if (!cols.includes(c)) cols.push(c);
  }
  return cols.sort((a, b) => a - b).map((c) => ({ c, v: hash(c, r + 99) }));
});
const OFFSET = ROWS.reduce((acc, row) => [...acc, acc.at(-1) + row.length], [0]);
const NNZ = OFFSET.at(-1);
const PICK = 29;
// The single cell shown first: a row with a nonzero near the centre.
const FOCUS = ROWS.findIndex((row) => row.some((e) => e.c >= C / 2 - 3 && e.c <= C / 2 + 2));
const CHUNKS = 6;
const PER_CHUNK = Math.ceil(NNZ / CHUNKS);
// Zarr (PCodec-8), the report's pooled ratio for chunk-compressed arrays.
const RATIO = 4.79;

// Timeline: every concept holds on screen before the next one starts.
const T = {
  dataset: [0.07, 0.14],
  corpus: [0.21, 0.28],
  back: [0.35, 0.4],
  zeros: [0.42, 0.48],
  pack: [0.52, 0.59],
  concat: [0.63, 0.72],
  chunk: [0.76, 0.79],
  squeeze: [0.79, 0.82],
  repack: [0.82, 0.85],
  read: [0.88, 0.95],
};
const STEPS = ["dense", "sparse rows", "three arrays", "compressed chunks", "random read"];

function layout(box) {
  const hud = 44;
  const g = { x: box.x, y: box.y + hud, w: box.w, h: box.h - hud };
  const s1 = Math.min(g.w / C, (g.h * 0.86) / R);
  return { hud, g, s1, cx: g.x + g.w / 2, cy: g.y + g.h / 2 };
}

function setup({ box, c, dpr }) {
  const L = layout(box);
  const s = L.s1;
  const tile = document.createElement("canvas");
  tile.width = Math.ceil(C * s * dpr);
  tile.height = Math.ceil(R * s * dpr);
  const t = tile.getContext("2d");
  t.scale(dpr, dpr);
  const zero = mix(c.bg, c.divider, 0.75);
  for (let r = 0; r < R; r++)
    for (let col = 0; col < C; col++) {
      t.fillStyle = zero;
      t.fillRect(col * s + s * 0.07, r * s + s * 0.07, s * 0.86, s * 0.86);
    }
  ROWS.forEach((row, r) =>
    row.forEach(({ c: col, v }) => {
      t.fillStyle = mix(c.tealSoft, c.teal, 0.45 + v * 0.55);
      t.fillRect(col * s + s * 0.07, r * s + s * 0.07, s * 0.86, s * 0.86);
    }),
  );
  return { L, tile };
}

// Arrays wrap two chunks per line so chunk boundaries line up with lines.
function arrays(g, s1) {
  const perRow = PER_CHUNK * 2;
  const rows = Math.ceil(NNZ / perRow);
  const sq = Math.min(s1 * 1.2, (g.w - 90) / perRow);
  const block = rows * (sq + 2 + 9);
  const x = g.x + (g.w - 80 - perRow * sq) / 2;
  const indptr = g.y + g.h * 0.12;
  const indices = indptr + Math.max(70, g.h * 0.14);
  return { perRow, rows, sq, x, indptr, indices, values: indices + block + Math.max(56, g.h * 0.1) };
}

// A label pinned next to what it describes.
function callout(ctx, c, x, y, title, sub, color, a, align = "center", box) {
  if (a <= 0) return;
  const size = box.w < 500 ? 22 : 26;
  ctx.font = `600 ${size}px 'Space Grotesk Variable', sans-serif`;
  const w1 = ctx.measureText(title).width;
  ctx.font = "500 12px 'IBM Plex Mono', monospace";
  const w = Math.min(box.w, Math.max(w1, ctx.measureText(sub).width) + 32);
  const bx = clamp(align === "right" ? x - w : x - w / 2, box.x, box.x + box.w - w);
  const by = Math.min(y, box.y + box.h - 88);
  fillRound(ctx, bx, by, w, 88, 10, alpha(c.bg, 0.94 * a));
  strokeRound(ctx, bx, by, w, 88, 10, alpha(color, 0.7 * a), 1.5);
  text(ctx, title, bx + 12, by + 28, { size, family: "head", weight: 600, color, a, maxWidth: w - 24 });
  paragraph(ctx, sub, bx + 12, by + 50, w - 24, { size: 11, family: "mono", color: c.text2, a });
}

function draw(ctx, { p, box, c, state }) {
  const note = (x, y, title, sub, color, a, align) => callout(ctx, c, x, y, title, sub, color, a, align, box);
  const { L, tile } = state;
  const { g, s1, cx, cy } = L;
  const at = (k) => ease.inOut(seg(p, ...T[k]));

  // Camera: one cell → a dataset → a mosaic of datasets, then back to one slice.
  const sCell = g.w / 11;
  const sFar = s1 * 0.26;
  let s = Math.exp(lerp(Math.log(sCell), Math.log(s1), at("dataset")));
  if (at("corpus") > 0) s = Math.exp(lerp(Math.log(s1), Math.log(sFar), at("corpus")));
  if (at("back") > 0) s = Math.exp(lerp(Math.log(sFar), Math.log(s1), at("back")));
  const focusY = lerp(FOCUS + 0.5, R / 2, at("dataset"));
  const others = at("dataset");
  const zeroFade = 1 - ease.out(seg(p, ...T.zeros));
  const pack = at("pack");
  const concat = seg(p, ...T.concat);
  const chunk = at("chunk");
  const squeeze = at("squeeze");
  const repack = at("repack");
  const pick = ease.out(seg(p, ...T.read));

  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x, g.y - 8, box.w, g.h + 16);
  ctx.clip();
  const gx = (col) => cx + (col - C / 2) * s;
  const gy = (row) => cy + (row - focusY) * s;
  const zero = mix(c.bg, c.divider, 0.75);
  const valueColor = (v) => mix(c.tealSoft, c.teal, 0.45 + v * 0.55);

  if (s < s1 * 0.999) {
    // Mosaic of identical slices; neighbours fade in as the camera pulls back.
    const tw = C * s;
    const th = R * s;
    const ghost = clamp((s1 / s - 1) / 2);
    for (let i = -5; i <= 5; i++)
      for (let j = -7; j <= 7; j++) {
        const x = cx - tw / 2 + i * (tw + s * 2);
        const y = cy - th / 2 + j * (th + s * 2);
        if (x > box.x + box.w || x + tw < box.x || y > g.y + g.h + 8 || y + th < g.y - 8) continue;
        ctx.globalAlpha = i === 0 && j === 0 ? 1 : ghost * (0.85 - 0.08 * Math.hypot(i, j));
        if (ctx.globalAlpha > 0) ctx.drawImage(tile, x, y, tw, th);
      }
    ctx.globalAlpha = 1;
  } else if (concat < 1) {
    const digits = s > 24;
    const r0 = Math.max(0, Math.floor(focusY - (cy - g.y) / s) - 1);
    const r1 = Math.min(R, Math.ceil(focusY + (g.y + g.h - cy) / s) + 1);
    const c0 = Math.max(0, Math.floor(C / 2 - (cx - box.x) / s) - 1);
    const c1 = Math.min(C, Math.ceil(C / 2 + (box.x + box.w - cx) / s) + 1);
    // Zeros fall away.
    if (zeroFade > 0)
      for (let r = r0; r < r1; r++) {
        const rowA = r === FOCUS ? 1 : others;
        if (rowA <= 0) continue;
        const nz = new Set(ROWS[r].map((e) => e.c));
        for (let col = c0; col < c1; col++) {
          if (nz.has(col)) continue;
          const fall = (1 - zeroFade) * (0.6 + hash(r, col) * 0.8);
          const y = gy(r) + fall * s * 3;
          ctx.globalAlpha = rowA * zeroFade;
          ctx.fillStyle = zero;
          ctx.fillRect(gx(col) + s * 0.07, y + s * 0.07, s * 0.86, s * 0.86);
          if (digits) text(ctx, "0", gx(col) + s / 2, y + s * 0.6, { size: s * 0.3, family: "mono", color: c.text3, align: "center", a: rowA * zeroFade });
        }
      }
    ctx.globalAlpha = 1;
    // Nonzeros slide to the left end of their row as (value, gene id) pairs.
    // Once concatenation starts, these squares are drawn in flight instead.
    for (let r = concat > 0 ? r1 : r0; r < r1; r++) {
      const rowA = r === FOCUS ? 1 : others;
      if (rowA <= 0) continue;
      ROWS[r].forEach(({ c: col, v }, j) => {
        const k = ease.inOut(clamp((pack - (r / R) * 0.35) / 0.65));
        const x = lerp(gx(col), gx(0) + j * 2 * s, k);
        ctx.globalAlpha = rowA;
        ctx.fillStyle = valueColor(v);
        ctx.fillRect(x + s * 0.07, gy(r) + s * 0.07, s * 0.86, s * 0.86);
        if (digits) text(ctx, (0.5 + v * 4).toFixed(1), x + s / 2, gy(r) + s * 0.6, { size: s * 0.28, family: "mono", color: c.bg, align: "center", a: rowA });
        if (k > 0) {
          ctx.fillStyle = alpha(c.accent, k);
          ctx.fillRect(gx(0) + (j * 2 + 1) * s + s * 0.07, gy(r) + s * 0.07, s * 0.86, s * 0.86);
        }
      });
    }
    ctx.globalAlpha = 1;
  }

  // Packed rows concatenate into the CSR arrays: squares stay squares.
  if (concat > 0) {
    const A = arrays(g, s1);
    const gapC = 14 * chunk;
    const pos = (k, top) => {
      const row = Math.floor(k / A.perRow);
      const col = k % A.perRow;
      const half = Math.floor(col / PER_CHUNK);
      return [A.x + col * A.sq + half * gapC, top + row * (A.sq + 2 + gapC * 0.6)];
    };
    const size = (ch) => Math.min(PER_CHUNK, NNZ - ch * PER_CHUNK);
    // Before compression: each chunk where its elements sit. After: a block
    // a 1/RATIO as wide, then all blocks repacked end to end.
    const blockAt = (ch, top) => {
      const [x0, y0] = pos(ch * PER_CHUNK, top);
      let cx = A.x;
      for (let j = 0; j < ch; j++) cx += (size(j) * A.sq) / RATIO + 10;
      const w = lerp(size(ch) * A.sq, (size(ch) * A.sq) / RATIO, squeeze);
      return { x: lerp(x0, cx, repack), y: lerp(y0, top, repack), w, h: A.sq };
    };
    const touched = [...new Set(Array.from({ length: ROWS[PICK].length }, (_, j) => Math.floor((OFFSET[PICK] + j) / PER_CHUNK)))];
    ROWS.forEach((row, r) => {
      const fT = ease.inOut(clamp((concat - (r / R) * 0.5) / 0.5));
      row.forEach(({ v }, j) => {
        const k = OFFSET[r] + j;
        const ch = Math.floor(k / PER_CHUNK);
        for (const [dx, top, color] of [[0, A.values, valueColor(v)], [1, A.indices, c.accent]]) {
          let [x, y] = pos(k, top);
          x = lerp(gx(0) + (j * 2 + dx) * s + s * 0.07, x, fT);
          y = lerp(gy(r) + s * 0.07, y, fT);
          // Squeezing pulls the squares into their chunk's block.
          if (squeeze > 0) {
            const b = blockAt(ch, top);
            const [bx] = pos(ch * PER_CHUNK, top);
            x = lerp(x, b.x + (x - bx) / RATIO, squeeze);
          }
          const sz = lerp(s * 0.86, A.sq - 1, fT) * (1 - squeeze * 0.5);
          ctx.globalAlpha = 1 - squeeze;
          ctx.fillStyle = color;
          ctx.fillRect(x, y, sz, sz);
        }
      });
    });
    ctx.globalAlpha = 1;
    const labelA = seg(concat, 0.6, 1);
    for (const [top, name, sizeLabel] of [[A.indices, "indices", "gene ids · 8 TB"], [A.values, "values", "expression · 8 TB"]]) {
      text(ctx, name, A.x, top - 10, { size: 14, family: "mono", weight: 600, color: c.text, a: labelA });
      text(ctx, squeeze > 0.5 ? `compressed · ${(8 / RATIO).toFixed(1)} TB` : sizeLabel, A.x + 78, top - 10, { size: 12, family: "mono", color: squeeze > 0.5 ? c.teal : c.text3, a: labelA });
      if (chunk > 0)
        for (let ch = 0; ch < CHUNKS; ch++) {
          const b = blockAt(ch, top);
          const hit = pick > 0 && touched.includes(ch);
          if (squeeze > 0) {
            ctx.globalAlpha = squeeze;
            fillRound(ctx, b.x, b.y, b.w, b.h, 3, mix(top === A.values ? c.teal : c.accent, c.bg, 0.2));
            for (let q = 0; q < b.w; q += 3) {
              ctx.fillStyle = alpha(c.bg, 0.25 * hash(q, ch));
              ctx.fillRect(b.x + q, b.y, 1.5, b.h);
            }
            ctx.globalAlpha = 1;
          }
          strokeRound(ctx, b.x - 3, b.y - 3, b.w + 6, b.h + 6, 4, alpha(hit ? c.danger : c.text3, chunk * (hit ? 1 : 0.6)), hit ? 2 : 1);
          if (b.w > 34 || repack > 0.5) text(ctx, repack > 0.5 ? `c${ch}` : `chunk ${ch}`, b.x + b.w / 2, b.y + b.h + 14, { size: 10, family: "mono", color: hit ? c.danger : c.text3, align: "center", a: chunk });
        }
      // A read decompresses every touched chunk in full, then keeps a few values.
      if (pick > 0) {
        const dy = top + A.sq + 26;
        let dx = A.x;
        touched.forEach((ch) => {
          const b = blockAt(ch, top);
          const n = size(ch);
          for (let q = 0; q < n; q++) {
            const k = ch * PER_CHUNK + q;
            const own = k >= OFFSET[PICK] && k < OFFSET[PICK + 1];
            const u = ease.out(clamp((pick - 0.15 - (q / n) * 0.25) / 0.45));
            const x = lerp(b.x + (q / n) * b.w, dx + q * A.sq, u);
            const y = lerp(b.y, dy, u);
            ctx.globalAlpha = u;
            ctx.fillStyle = own ? c.danger : mix(top === A.values ? c.teal : c.accent, c.bg, 0.65);
            ctx.fillRect(x, y, A.sq - 1, A.sq - 1);
          }
          dx += n * A.sq + 12;
        });
        ctx.globalAlpha = 1;
        text(ctx, `decoded ${touched.length * PER_CHUNK} · needed ${ROWS[PICK].length}`, A.x, dy + A.sq + 18, { size: 10, family: "mono", weight: 600, color: c.danger, a: seg(pick, 0.5, 0.8) });
      }
    }
    // indptr: one offset per row.
    text(ctx, "indptr", A.x, A.indptr - 10, { size: 14, family: "mono", weight: 600, color: c.text, a: labelA });
    text(ctx, "row offsets · 8 GB", A.x + 70, A.indptr - 10, { size: 12, family: "mono", color: c.text3, a: labelA });
    for (let r = 0; r <= R; r++) {
      const a = clamp((concat - (Math.min(r, R - 1) / R) * 0.5) / 0.5);
      const hl = pick > 0 && (r === PICK || r === PICK + 1);
      ctx.globalAlpha = a * (pick > 0 && !hl ? 1 - pick * 0.6 : 1);
      ctx.fillStyle = hl ? c.danger : c.text2;
      ctx.fillRect(A.x + r * (A.sq + 1), A.indptr, A.sq, A.sq);
    }
    ctx.globalAlpha = 1;
    if (pick > 0) {
      const px = A.x + PICK * (A.sq + 1) + A.sq / 2;
      const u = ease.out(seg(pick, 0, 0.2));
      for (const top of [A.indices, A.values])
        touched.forEach((ch) => {
          const b = blockAt(ch, top);
          line(ctx, px, A.indptr + A.sq, lerp(px, b.x + b.w / 2, u), lerp(A.indptr + A.sq, b.y - 3, u), alpha(c.danger, 0.8), 1.5, [4, 3]);
        });
    }
  }
  ctx.restore();

  // Labels pinned to the visual, one per concept.
  const hold = (a, b) => seg(p, a, a + 0.02) * (1 - seg(p, b - 0.02, b));
  const rowBottom = gy(FOCUS) + s + 18;
  note( cx, Math.min(rowBottom, g.y + g.h - 70), "1 cell · 160 KB", "40,000 genes × 4 bytes, dense", c.text, 1 - seg(p, 0.05, 0.07));
  note( cx, g.y + g.h * 0.06, "1 dataset · 16 GB", "100,000 cells × 160 KB, dense", c.text, hold(0.14, 0.21));
  note( cx, g.y + g.h * 0.06, "1 corpus · 160 TB", "1,000,000,000 cells × 160 KB, dense", c.text, hold(0.28, 0.35));
  note( cx, g.y + g.h * 0.02, "160 TB → 16 TB", "keep only nonzeros · ~2,000 per cell", c.accent, hold(0.47, 0.52));
  note( cx, g.y + g.h * 0.02, "16 KB per cell", "each row packed · ■ value  ■ gene id", c.accent, hold(0.59, 0.63));
  note( box.x + box.w, g.y + g.h * 0.84, "8 TB + 8 TB + 8 GB", "values · indices · indptr", c.accent, hold(0.72, 0.76), "right");
  note( box.x + box.w, g.y + g.h * 0.84, `16 TB → ${(16 / RATIO).toFixed(1)} TB`, `chunks compressed on their own · Zarr (PCodec), ${RATIO}×`, c.teal, hold(0.84, 0.88), "right");
  note( box.x + box.w, g.y + g.h * 0.84, `cell ${PICK} · ${ROWS[PICK].length} values`, "→ decompress every chunk it touches, in full", c.danger, seg(p, 0.93, 0.95), "right");

  // Progress across the concepts.
  const stepAt = [0, T.zeros[0], T.concat[0], T.chunk[0], T.read[0]];
  let cur = 0;
  stepAt.forEach((a, i) => {
    if (p >= a) cur = i;
  });
  const shown = box.w < 620 ? [[STEPS[cur], cur]] : STEPS.map((name, i) => [name, i]);
  let x = box.x;
  shown.forEach(([name, i]) => {
    const on = i === cur;
    ctx.font = "500 12px 'IBM Plex Mono', monospace";
    const label = box.w < 620 ? `${i + 1} / ${STEPS.length} · ${name}` : name;
    const w = ctx.measureText(label).width + 20;
    fillRound(ctx, x, box.y, w, 24, 12, on ? c.accent : c.surface);
    text(ctx, label, x + w / 2, box.y + 16, { size: 12, family: "mono", weight: on ? 600 : 500, color: on ? c.bg : c.text2, align: "center" });
    x += w + 12;
  });
}

export default { setup, draw, continuous: false };
