import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, seg, text, paragraph } from "../engine.js";
import { B, STEPS, ladder } from "./fmt-common.js";

// Gene ids within one chunk. Bars: how many of the chunk's cells express
// each gene (illustrative, Zipf-like). Highlighted: the genes one cell
// expresses. Gap coding stores differences between sorted ids; renumbering
// genes by frequency packs a cell's genes into small, adjacent ids.
const G = 40;
const GENES = (() => {
  const order = Array.from({ length: G }, (_, g) => g).sort((a, b) => hash(a, 5) - hash(b, 5));
  return Array.from({ length: G }, (_, g) => {
    const rank = order.indexOf(g);
    const freq = 1 / (rank + 1) ** 0.75;
    // A cell mostly expresses the chunk's common genes, plus a few rare ones.
    return { g, rank, freq, on: hash(g, 77) < (rank < 9 ? 0.85 : 0.1) };
  });
})();
const ON = GENES.filter((x) => x.on);

function gaps(key) {
  const ids = ON.map(key).sort((a, b) => a - b);
  return ids.map((v, i) => v - (i ? ids[i - 1] : -1) - 1);
}
const SRC_GAPS = gaps((x) => x.g);
const CODED_GAPS = gaps((x) => x.rank);

function draw(ctx, env) {
  const { p, box, c, wide } = env;
  const intro = ease.out(seg(p, 0, 0.06));
  const gapOn = ease.out(seg(p, 0.1, 0.24));
  const sort = ease.inOut(seg(p, 0.36, 0.56));
  const stats = ease.out(seg(p, 0.64, 0.8));
  const from = p < 0.32 ? B["2_records_raw"] : B["3_gap_source_order"];
  const to = p < 0.32 ? B["3_gap_source_order"] : B["4_gap_renumbered"];
  const y0 = ladder(ctx, env, from, to, p < 0.32 ? gapOn : sort);

  // Histogram of gene frequency in the chunk.
  const hx = box.x + 36;
  const hw = box.w - 36;
  // Reserve separate rows for the histogram legend and the gap heading.
  const sectionGap = 24;
  const hh = Math.max(48, Math.min(wide ? 170 : 130, box.h * (wide ? 0.26 : 0.22)) - sectionGap);
  const hy = y0 + (wide ? 28 : 42);
  const bw = hw / G;
  text(ctx, sort > 0.5 ? "GENES BY FREQUENCY ID" : "GENES IN SOURCE ORDER", box.x, hy - 4, { size: 11, family: "mono", weight: 600, color: c.text2, a: intro });
  text(ctx, "bar: cells in the chunk expressing the gene", box.x + box.w, hy - 20, { size: 10, family: "mono", color: c.text3, align: "right", a: intro });
  GENES.forEach((x) => {
    const pos = lerp(x.g, x.rank, sort);
    const bx = hx + pos * bw;
    const h = x.freq * hh * intro;
    fillRound(ctx, bx + 1, hy + 10 + hh - h, bw - 2, h, 2, alpha(x.on ? c.accent : c.text3, x.on ? 0.9 : 0.35));
  });
  line(ctx, hx, hy + 10 + hh, hx + hw, hy + 10 + hh, c.border, 1);
  // Id labels under a few bars.
  for (let i = 0; i < G; i += wide ? 4 : 8) text(ctx, String(i), hx + i * bw + bw / 2, hy + hh + 26, { size: 10, family: "mono", color: c.text3, align: "center", a: intro });
  const legendY = hy + hh + 44;
  text(ctx, "■ expressed by this cell", box.x, legendY, { size: 11, family: "mono", color: c.accent, a: intro });

  // The cell's gap sequence, recomputed as ids change.
  const gy = legendY + 40;
  const current = sort > 0.5 ? CODED_GAPS : SRC_GAPS;
  const flip = Math.abs(Math.cos(sort * Math.PI));
  text(ctx, "THIS CELL'S GAPS", box.x, gy - 8, { size: 11, family: "mono", weight: 600, color: c.text2, a: gapOn });
  let x = box.x;
  const th = 30;
  current.forEach((v, i) => {
    const w = Math.min(46, 16 + Math.log2(v + 1) * 7);
    const a = gapOn * clamp(gapOn * current.length - i);
    const hh2 = th * Math.max(0.15, flip);
    fillRound(ctx, x, gy + (th - hh2) / 2, w, hh2, 4, alpha(v === 0 ? c.teal : v < 3 ? c.tealSoft : c.accentSoft, a));
    if (flip > 0.4) text(ctx, String(v), x + w / 2, gy + th / 2 + 4, { size: 12, family: "mono", weight: 600, color: v === 0 ? c.bg : c.text, align: "center", a });
    x += w + 4;
  });
  const zeros = (arr) => Math.round((arr.filter((v) => v === 0).length / arr.length) * 100);
  text(ctx, `${zeros(current)}% zero in this cell`, box.x, gy + th + 20, { size: 11, family: "mono", color: c.teal, a: gapOn * flip });

  // Measured statistics over the sample.
  if (stats > 0) {
    const sy = gy + th + (wide ? 48 : 44);
    const rowH = wide ? 30 : 38;
    const barX = box.x + (wide ? 200 : 0);
    const barW = box.w - (barX - box.x) - 70;
    const rows = [
      ["bits per gene id · raw", STEPS.index_bits.raw, 16, c.text3],
      ["gaps, source order", STEPS.index_bits.source, 16, c.accent],
      ["gaps, renumbered", STEPS.index_bits.renumbered, 16, c.teal],
    ];
    rows.forEach(([name, v, max, color], i) => {
      const y = sy + i * rowH;
      const a = clamp(stats * 3 - i * 0.6);
      text(ctx, name, wide ? barX - 10 : box.x, wide ? y + 14 : y - 3, { size: 11, family: "mono", color: c.text2, align: wide ? "right" : "left", a });
      fillRound(ctx, barX, y + 2, (v / max) * barW * a, 16, 4, alpha(color, 0.85 * a));
      text(ctx, `${v.toFixed(2)} bits`, barX + (v / max) * barW + 8, y + 15, { size: 12, family: "mono", weight: 600, color, a });
    });
    const zy = sy + rows.length * rowH + 14;
    paragraph(ctx, `zero gaps: ${Math.round(STEPS.zero_gaps.source_order * 100)}% source → ${Math.round(STEPS.zero_gaps.renumbered * 100)}% ranked`, box.x, zy, box.w, { size: wide ? 13 : 11, family: "mono", weight: 600, color: c.teal, a: stats });
  }
  if (sort > 0 && sort < 1) glow(ctx, hx + hw * 0.15, hy + hh * 0.6, hw * 0.25, c.accent, 0.1 * Math.sin(sort * Math.PI));
}

export default { draw, continuous: false };
