import { alpha, clamp, ease, fillRound, glow, lerp, line, mix, seg, strokeRound, text, paragraph } from "../engine.js";
import { B, CSR, ladder } from "./fmt-common.js";

// Native CSR for an eight-cell chunk, turned into CellPack records:
// 1. indptr splits indices and values into per-cell slices (boxed);
// 2. pack: each cell's two slices come together as one record;
// 3. narrow: 2-byte gene ids, values in their stored dtype;
// 4. merge: records sit end to end in a chunk, found by an offset table;
// 5. a read: one offset → one record.
const NNZ = [5, 3, 6, 4, 7, 3, 5, 4];
const START = NNZ.reduce((a, n) => [...a, a.at(-1) + n], [0]);
const E = START.at(-1);
const PICK = 3;
const T = { box: [0.08, 0.22], pack: [0.28, 0.46], narrow: [0.52, 0.62], merge: [0.68, 0.8], access: [0.85, 0.95] };
const CAPTIONS = [
  [0, "Native CSR: indptr points into two separate arrays"],
  [T.box[0], "1 · indptr splits both arrays into per-cell slices"],
  [T.pack[0], "2 · pack: each cell's ids and values become one record"],
  [T.narrow[0], "3 · 2-byte gene ids, values in their stored dtype"],
  [T.merge[0], "4 · records end to end in a chunk + an offset table"],
  [T.access[0], "5 · one cell → one offset → one contiguous read"],
];

function draw(ctx, env) {
  const { p, box, c, wide } = env;
  const at = (k) => ease.inOut(seg(p, ...T[k]));
  const boxA = at("box");
  const pack = seg(p, ...T.pack);
  const narrow = at("narrow");
  const merge = at("merge");
  const access = ease.out(seg(p, ...T.access));
  const intro = ease.out(seg(p, 0, 0.06));
  const y0 = ladder(ctx, env, CSR, B["2_records_raw"], narrow);

  // Geometry.
  const labelW = 70;
  const sx = box.x + labelW;
  const hdrW = 12;
  const recGap = lerp(14, 2, merge);
  const ew = Math.min(18, (box.w - labelW) / E, (box.w - labelW - 4 * (hdrW + 16)) / 38);
  const eh = Math.min(30, box.h * 0.05);
  const iw = ew * lerp(1, 0.5, narrow);
  const vw = ew * lerp(1, 0.75, narrow);
  const rowGap = eh + Math.max(44, box.h * 0.07);
  const rows = { indptr: y0 + 40, indices: y0 + 40 + rowGap, values: y0 + 40 + 2 * rowGap };
  const recTop = rows.values + eh + Math.max(40, box.h * 0.07);
  const recRowGap = eh + 34;
  const color = (r) => (access > 0 && r === PICK ? c.accent : mix(c.tealSoft, c.teal, 0.35 + (r % 3) * 0.25));
  const arrA = 1 - seg(pack, 0.8, 1) * 0.8;

  // Array labels and indptr.
  const lab = (y, name, sub) => {
    text(ctx, name, box.x, y + eh / 2 + 2, { size: 12, family: "mono", weight: 600, color: c.text, a: intro * arrA });
    text(ctx, sub, box.x, y + eh / 2 + 16, { size: 10, family: "mono", color: c.text3, a: intro * arrA });
  };
  lab(rows.indptr, "indptr", "u64");
  lab(rows.indices, "indices", "u32");
  lab(rows.values, "values", "f32");
  const pw = Math.min(30, (box.w - labelW) / 10);
  for (let r = 0; r <= NNZ.length; r++) fillRound(ctx, sx + r * pw, rows.indptr, pw - 3, eh, 3, alpha(c.text3, 0.7 * intro * arrA));
  const sliceX = (r) => sx + START[r] * ew;

  // 1. Pointers and boxes around each cell's slices.
  if (boxA > 0 && pack < 1) {
    const a = boxA * (1 - seg(pack, 0, 0.4));
    NNZ.forEach((n, r) => {
      const ra = clamp(boxA * NNZ.length - r) * a;
      line(ctx, sx + r * pw + pw / 2, rows.indptr + eh, sliceX(r) + 2, rows.indices - 6, alpha(c.text3, 0.6 * ra), 1, [3, 3]);
      for (const y of [rows.indices, rows.values]) strokeRound(ctx, sliceX(r) - 1, y - 4, n * ew + 1, eh + 8, 4, alpha(c.text2, ra), 1.2);
      text(ctx, `c${r}`, sliceX(r) + (n * ew) / 2, rows.indices - 10, { size: 10, family: "mono", weight: 600, color: c.text2, align: "center", a: ra });
    });
  }

  // Record positions: two rows of four, then end to end inside the chunk.
  const recW = (r) => hdrW + NNZ[r] * (iw + vw);
  const recPos = (r) => {
    const row = Math.floor(r / 4);
    let x = box.x + labelW;
    for (let i = row * 4; i < r; i++) x += recW(i) + recGap;
    return [x, recTop + row * recRowGap];
  };

  // 2–3. Elements fly from the arrays into records and narrow there.
  NNZ.forEach((n, r) => {
    const local = ease.inOut(clamp((pack - (r / NNZ.length) * 0.4) / 0.6));
    const [rx, ry] = recPos(r);
    const lift = access > 0 && r === PICK ? access * 10 : 0;
    const dim = access > 0 && r !== PICK ? 1 - access * 0.6 : 1;
    ctx.globalAlpha = intro * dim;
    for (let j = 0; j < n; j++) {
      const k = START[r] + j;
      const ix = lerp(sx + k * ew, rx + hdrW + j * iw, local);
      const iy = lerp(rows.indices, ry - lift, local);
      fillRound(ctx, ix + 0.5, iy, Math.max(1, lerp(ew, iw, local) - 1), eh, 2, color(r));
      const vx = lerp(sx + k * ew, rx + hdrW + n * iw + j * vw, local);
      const vy = lerp(rows.values, ry - lift, local);
      fillRound(ctx, vx + 0.5, vy, Math.max(1, lerp(ew, vw, local) - 1), eh, 2, alpha(color(r), 0.5));
    }
    ctx.globalAlpha = 1;
    if (local > 0.6) {
      const a = seg(local, 0.6, 1) * dim;
      const hl = r === PICK && access > 0;
      fillRound(ctx, rx, ry - lift, hdrW - 2, eh, 2, alpha(c.text, a));
      strokeRound(ctx, rx - 2, ry - lift - 3, recW(r) + 2, eh + 6, 4, alpha(hl ? c.accent : c.border, a), hl ? 2 : 1);
      text(ctx, wide ? `cell ${r}` : `c${r}`, rx + recW(r) / 2, ry - lift + eh + 16, { size: 10, family: "mono", color: hl ? c.accent : c.text3, align: "center", a });
    }
  });
  if (pack > 0.7) text(ctx, "■ header   ■ gene ids   ■ values", box.x, recTop + 2 * recRowGap + 4, { size: 11, family: "mono", color: c.text3, a: seg(pack, 0.7, 1) * (1 - seg(p, T.merge[0] - 0.04, T.merge[0])) });
  if (narrow > 0 && wide) {
    const [x] = recPos(3);
    text(ctx, "ids 4 → 2 B · counts f32 → u16", box.x, recTop - 24, { size: 11, family: "mono", weight: 600, color: c.accent, a: narrow * (1 - seg(p, T.merge[0] - 0.04, T.merge[0])) });
  }

  // 4. The chunk frame and its offset table.
  if (merge > 0) {
    const fx = box.x + labelW - 10;
    const fy = recTop - 18;
    const fw = Math.max(recPos(3)[0] + recW(3), recPos(7)[0] + recW(7)) - fx + 10;
    const fh = recRowGap + eh + 32;
    strokeRound(ctx, fx, fy, fw, fh, 8, alpha(c.teal, merge), 2);
    text(ctx, "CHUNK · records region", fx + 10, fy - 8, { size: 11, family: "mono", weight: 700, color: c.teal, a: merge });
    const oy = fy + fh + 34;
    const ow = Math.min(46, fw / 8 - 6);
    text(ctx, "offset + length per record", fx, oy - 8, { size: 11, family: "mono", color: c.text2, a: merge });
    NNZ.forEach((_, r) => {
      const x = fx + r * (ow + 6);
      const hl = access > 0 && r === PICK;
      fillRound(ctx, x, oy, ow, 18, 3, alpha(hl ? c.accent : c.text3, (hl ? 1 : 0.6) * merge));
      text(ctx, `c${r}`, x + ow / 2, oy + 13, { size: 9, family: "mono", weight: 700, color: c.bg, align: "center", a: merge });
      if (hl) {
        const [rx, ry] = recPos(r);
        line(ctx, x + ow / 2, oy, rx + recW(r) / 2, ry + eh + 4, alpha(c.accent, 0.9 * access), 2, [4, 3]);
      }
    });
  }

  // 5. The read.
  if (access > 0) {
    const [rx, ry] = recPos(PICK);
    glow(ctx, rx + recW(PICK) / 2, ry, 90, c.accent, 0.2 * access);
  }

  // Caption for the current step.
  let cap = CAPTIONS[0][1];
  CAPTIONS.forEach(([a, s]) => {
    if (p >= a) cap = s;
  });
  paragraph(ctx, cap, box.x, box.y + box.h - 38, box.w, { size: wide ? 12 : 11, family: "mono", weight: 600, color: c.accent, a: intro });
}

export default { draw, continuous: false };
