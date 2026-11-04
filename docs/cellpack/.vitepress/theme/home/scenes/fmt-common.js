import data from "../scenes.json";
import { alpha, ease, fillRound, font, grouped, lerp, line, text } from "../engine.js";

// Measured per-cell bytes of each format step (pooled over the sample).
export const STEPS = data.steps;
export const B = STEPS.bytes;
export const CSR = B["1a_csr_u32_f32"];
export const FINAL = B["8b_on_disk"];
export const LADDER = [
  ["CSR", B["1a_csr_u32_f32"]],
  ["2-byte ids", B["1c_csr_u16_stored_dtype"]],
  ["gaps", B["3_gap_source_order"]],
  ["renumbered", B["4_gap_renumbered"]],
  ["dictionary", B["5_values_dictionary"]],
  ["+ metadata", B["7_plus_chunk_metadata"]],
  ["pages", B["8_packed_4k_pages"]],
];
export const SAMPLE = `${STEPS.datasets} CELLxGENE datasets · ${grouped(STEPS.cells)} cells`;

/**
 * The shared byte ladder at the top of every format scene: bytes per cell,
 * animated from `from` to `to` by `u`.
 */
export function ladder(ctx, { box, c, wide }, from, to, u, a = 1) {
  const v = lerp(from, to, ease.inOut(u));
  const big = Math.min(40, box.w / 14);
  const y = box.y;
  const label = `${grouped(v)} B`;
  text(ctx, label, box.x, y + big * 0.85, { size: big, family: "head", color: v < 3000 ? c.teal : c.text, a });
  font(ctx, big, "head");
  text(ctx, "per cell", box.x + ctx.measureText(label).width + 10, y + big * 0.85, { size: 12, family: "mono", color: c.text2, a });
  text(ctx, `${(CSR / v).toFixed(2)}× smaller than CSR`, box.x + box.w, y + big * 0.85 + (box.w < 340 ? 20 : 0), { size: wide ? 15 : 12, family: "head", weight: 600, color: c.accent, align: "right", a });
  const by = y + big + 14 + (box.w < 340 ? 20 : 0);
  const bw = box.w;
  fillRound(ctx, box.x, by, bw, 10, 5, alpha(c.divider, 0.6 * a));
  fillRound(ctx, box.x, by, (v / CSR) * bw, 10, 5, alpha(v < 3000 ? c.teal : c.accent, a));
  if (wide)
    LADDER.forEach(([name, bytes], i) => {
      const x = box.x + (bytes / CSR) * bw;
      line(ctx, x, by - 3, x, by + 13, alpha(c.text3, 0.5 * a), 1);
      if (Math.abs(bytes - to) > 1) return;
      text(ctx, name, x, by + 26, { size: 10, family: "mono", color: c.text3, align: "center", a });
    });
  return by + (wide ? 40 : 24);
}
