import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, mix, seg, strokeRound, text, paragraph } from "../engine.js";
import { B, STEPS, ladder } from "./fmt-common.js";

// One cell's values. Counts are small integers, so even normalized values
// repeat. The record keeps each distinct bit pattern once (a dictionary),
// replaces values with ranks, and entropy-codes the ranks with a shared
// model: frequent ranks take a fraction of a bit, rare ones several bits.
const N = 36;
const VALUES = Array.from({ length: N }, (_, i) => {
  const u = hash(i, 31);
  const count = u < 0.52 ? 1 : u < 0.76 ? 2 : u < 0.87 ? 3 : u < 0.93 ? 4 : u < 0.97 ? 6 : 11;
  return Math.log1p(count);
});
const DICT = [...new Set(VALUES)].sort((a, b) => a - b);
const RANKS = VALUES.map((v) => DICT.indexOf(v));
const FREQ = DICT.map((_, r) => RANKS.filter((x) => x === r).length / N);
// Ideal code length of each rank under the model: −log2 p.
const BITS = FREQ.map((p) => -Math.log2(p));
const CODED = RANKS.reduce((a, r) => a + BITS[r], 0);
const FIXED = Math.ceil(Math.log2(DICT.length));

function draw(ctx, env) {
  const { p, t, box, c, wide } = env;
  const intro = ease.out(seg(p, 0, 0.06));
  const gather = ease.inOut(seg(p, 0.1, 0.3));
  const code = ease.inOut(seg(p, 0.4, 0.6));
  const stats = ease.out(seg(p, 0.68, 0.82));
  const y0 = ladder(ctx, env, B["4_gap_renumbered"], B["5_values_dictionary"], ease.inOut(seg(p, 0.1, 0.6)));

  const palette = DICT.map((_, i) => mix(c.teal, c.accent, i / Math.max(1, DICT.length - 1)));
  const dictW = wide ? 160 : 110;
  const area = box.w - dictW - 24;
  const per = wide ? 12 : 9;
  const tw = Math.min(56, area / per - 6);
  const th = Math.min(30, tw * 0.6);
  const vy = y0 + 26;
  const rows = Math.ceil(N / per);
  text(ctx, gather < 0.5 ? "VALUES · 32 bits each" : wide ? "RANKS INTO THE CELL'S DICTIONARY" : "THIS CELL'S RANKS", box.x, vy - 10, { size: 11, family: "mono", weight: 600, color: c.text2, a: intro });

  // Dictionary of distinct bit patterns.
  const dx = box.x + box.w - dictW;
  text(ctx, "DICTIONARY", dx, vy - 10, { size: 11, family: "mono", weight: 600, color: c.accent, a: gather });
  const dh = Math.min(th, (rows * (th + 8)) / DICT.length - 4);
  DICT.forEach((v, i) => {
    const a = clamp(gather * 2 - i * 0.15);
    const y = vy + i * (dh + 4);
    fillRound(ctx, dx, y, dictW, dh, 5, alpha(palette[i], 0.18 * a));
    strokeRound(ctx, dx, y, dictW, dh, 5, alpha(palette[i], 0.8 * a), 1);
    text(ctx, `${i} → ${v.toFixed(4)}`, dx + 10, y + dh / 2 + 4, { size: 11, family: "mono", color: c.text, a });
  });

  // The coded stream: each rank becomes a code as wide as its bit length.
  const sy = vy + rows * (th + 8) + 60 - code * 48;
  const unit = Math.min(16, (box.w - 4) / (N * FIXED));
  const fixedW = N * FIXED * unit;
  const codedW = CODED * unit;
  const ch = 26;
  let acc = 0;
  RANKS.forEach((r, k) => {
    const col = k % per;
    const row = Math.floor(k / per);
    const gxp = box.x + col * (tw + 6);
    const gyp = vy + row * (th + 8);
    const local = ease.inOut(clamp((gather - (k / N) * 0.4) / 0.6));
    const codeK = ease.inOut(clamp((code - (k / N) * 0.5) / 0.5));
    const cx = box.x + acc * unit;
    acc += BITS[r];
    const w = lerp(lerp(tw, tw * 0.45, local), BITS[r] * unit - 1, codeK);
    const h = lerp(th, ch, codeK);
    const x = lerp(gxp, cx, codeK);
    const y = lerp(gyp, sy, codeK) - Math.sin(codeK * Math.PI) * 30;
    const flash = Math.sin(local * Math.PI);
    fillRound(ctx, x, y, Math.max(1, w), h, codeK > 0.5 ? 2 : 5, local > 0.5 ? palette[r] : mix(c.surface, palette[r], flash * 0.5));
    if (local < 0.5) strokeRound(ctx, x, y, w, h, 5, alpha(c.border, 1), 1);
    const label = local > 0.5 ? String(r) : VALUES[k].toFixed(2);
    if (w > 9) text(ctx, label, x + w / 2, y + h / 2 + 4, { size: Math.min(12, tw * 0.24), family: "mono", weight: local > 0.5 ? 700 : 400, color: local > 0.5 ? c.bg : c.text, align: "center", a: intro });
  });

  if (code > 0) {
    // Fixed-width ranks for comparison, and the model's code lengths.
    text(ctx, wide ? "ENTROPY-CODED RANKS · shared rANS model" : "rANS-CODED RANKS", box.x, sy - 46, { size: 10, family: "mono", weight: 600, color: c.text2, a: code });
    strokeRound(ctx, box.x - 1, sy - 15, fixedW + 2, 8, 3, alpha(c.text3, 0.7 * code), 1);
    text(ctx, `fixed ${FIXED}-bit ranks: ${N * FIXED} bits`, box.x, sy - 23, { size: 10, family: "mono", color: c.text3, a: code });
    text(ctx, `coded ${Math.round(CODED)} bits · ${N} values · float32: ${N * 32} bits`, box.x, sy + ch + 22, { size: wide ? 12 : 10, family: "mono", weight: 600, color: c.teal, a: seg(code, 0.7, 1), maxWidth: box.w });
    line(ctx, box.x + codedW, sy - 4, box.x + codedW, sy + ch + 4, alpha(c.teal, seg(code, 0.7, 1)), 2);
    // Code length per rank.
    const ky = sy + ch + 50;
    const cols = wide ? DICT.length : 3;
    const kw = Math.min(110, box.w / cols - 8);
    text(ctx, "rank · share · bits (−log₂ p)", box.x, ky - 8, { size: 10, family: "mono", color: c.text3, a: code });
    DICT.forEach((_, r) => {
      const x = box.x + (r % cols) * (kw + 8);
      const y = ky + Math.floor(r / cols) * 42;
      fillRound(ctx, x, y, Math.min(kw, BITS[r] * unit), 10, 2, alpha(palette[r], code));
      text(ctx, `${r} · ${Math.round(FREQ[r] * 100)}% · ${BITS[r].toFixed(1)}b`, x, y + 25, { size: 10, family: "mono", color: c.text2, a: code, maxWidth: kw });
    });
    glow(ctx, box.x + codedW, sy + ch / 2, 40, c.teal, 0.2 * seg(code, 0.7, 1) * (0.7 + 0.3 * Math.sin(t * 3)));
  }

  if (stats > 0) {
    const ty = sy + ch + (wide ? 120 : 164);
    const barX = box.x + (wide ? 200 : 118);
    const barW = box.w - (barX - box.x) - 80;
    const rowsS = [
      ["raw counts · u16", 16, c.text3],
      ["  dictionary + rANS", STEPS.value_bits.raw_u16, c.teal],
      ["normalized · f32", 32, c.text3],
      ["  dictionary + rANS", STEPS.value_bits.normed_f32, c.teal],
    ];
    text(ctx, wide ? "BITS PER VALUE, MEASURED ON THE SAMPLE" : "BITS PER VALUE · SAMPLE", box.x, ty - 8, { size: 11, family: "mono", weight: 600, color: c.text2, a: stats });
    rowsS.forEach(([name, v, color], i) => {
      const y = ty + i * 24;
      const a = clamp(stats * 3 - i * 0.4);
      text(ctx, name, barX - 10, y + 13, { size: 11, family: "mono", color: c.text2, align: "right", a });
      fillRound(ctx, barX, y + 2, (v / 32) * barW * a, 14, 4, alpha(color, 0.85 * a));
      text(ctx, v >= 10 ? `${v}` : v.toFixed(2), barX + (v / 32) * barW + 8, y + 14, { size: 12, family: "mono", weight: 600, color, a });
    });
  }
}

export default { draw };
