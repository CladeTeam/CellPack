import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, mix, seg, strokeRound, text, paragraph } from "../engine.js";

// Training corpora (report §1) at 16 KB per cell in CSR, against the 2 TB of
// host memory in a DGX H200.
const CORPORA = [
  { name: "scGPT", note: "training corpus", cells: 33e6, year: "2024" },
  { name: "Tahoe-x1", note: "training corpus", cells: 266e6, year: "2025" },
  { name: "scBaseCount", note: "public repository", cells: 502e6, year: "2026" },
  { name: "Arc Institute", note: "planned experiments", cells: 1e9, year: "next", planned: true },
];
const KB = 16e3;
const MEMORY_TB = 2;
const MAX_TB = 17;
const tb = (cells) => (cells * KB) / 1e12;

function draw(ctx, { p, t, box, c, wide }) {
  const sideW = wide ? Math.min(170, box.w * 0.24) : 0;
  const g = { x: box.x + 44, y: box.y + 24, w: box.w - 44 - sideW - (wide ? 24 : 0), h: box.h - 24 - 100 };
  const Y = (v) => g.y + g.h - (v / MAX_TB) * g.h;
  const intro = ease.out(seg(p, 0, 0.06));
  const grow = [seg(p, 0.04, 0.2), seg(p, 0.3, 0.46), seg(p, 0.5, 0.6), seg(p, 0.6, 0.72)];
  const spill = seg(p, 0.77, 0.9);

  // Axis and the memory line.
  for (let v = 0; v <= 16; v += 4) {
    line(ctx, g.x, Y(v), g.x + g.w, Y(v), alpha(c.divider, 0.8 * intro), 1);
    text(ctx, `${v} TB`, g.x - 8, Y(v) + 4, { size: 11, family: "mono", color: c.text3, align: "right", a: intro });
  }
  const my = Y(MEMORY_TB);
  fillRound(ctx, g.x, my, g.w, g.y + g.h - my, 0, alpha(c.tealSoft, 0.55 * intro));
  const crossed = grow[1] > 0.45;
  const flash = crossed ? Math.max(0, 1 - (p - 0.37) / 0.08) : 0;
  line(ctx, g.x, my, g.x + g.w, my, crossed ? c.danger : c.teal, 2 + flash * 2);

  // Bars.
  const n = CORPORA.length;
  const slot = g.w / n;
  const bw = Math.min(86, slot * 0.5);
  CORPORA.forEach((k, i) => {
    const gT = ease.out(grow[i]);
    const x = g.x + slot * i + (slot - bw) / 2;
    const size = tb(k.cells) * gT;
    const top = Y(size);
    const below = Math.min(size, MEMORY_TB);
    fillRound(ctx, x, Y(below), bw, g.y + g.h - Y(below), 3, alpha(c.teal, k.planned ? 0.45 : 0.85));
    if (size > MEMORY_TB) {
      // The part beyond memory: before the spill a solid bar; during the
      // spill it breaks into blocks (drawn below) that move to storage.
      const h = my - top;
      if (spill <= 0 || !wide) fillRound(ctx, x, top, bw, h, 3, alpha(c.danger, k.planned ? 0.5 : 0.85));
      strokeRound(ctx, x, top, bw, h, 3, alpha(c.danger, 0.8), 1.2);
      if (k.planned) strokeRound(ctx, x, top, bw, g.y + g.h - top, 3, alpha(c.text3, 0.8), 1);
    }
    if (gT > 0) {
      const label = size >= 1 ? `${size.toFixed(size >= 10 ? 0 : 2)} TB` : `${Math.round(size * 1000)} GB`;
      text(ctx, label, x + bw / 2, Math.abs(top - my) < 30 ? my - 34 : top - 10, { size: wide ? 15 : 12, family: "head", weight: 600, color: size > MEMORY_TB ? c.danger : c.teal, align: "center" });
    }
    const cells = k.cells >= 1e9 ? "1B" : `${Math.round(k.cells / 1e6)}M`;
    paragraph(ctx, k.name, x + bw / 2, g.y + g.h + 18, slot - 4, { size: wide ? 12 : 10, weight: 600, color: c.text, align: "center", a: intro });
    text(ctx, cells, x + bw / 2, g.y + g.h + 49, { size: 10, family: "mono", color: c.text2, align: "center", a: intro });
    text(ctx, k.year, x + bw / 2, g.y + g.h + 64, { size: 10, family: "mono", color: c.text3, align: "center", a: intro });

  });

  // Memory-line label on top of the bars.
  const label = "HOST MEMORY · DGX H200 · 2 TB";
  fillRound(ctx, g.x + 4, my - 22, label.length * 6.9 + 16, 18, 4, alpha(c.bg, 0.92 * intro));
  text(ctx, label, g.x + 12, my - 9, { size: 11, family: "mono", weight: 600, color: crossed ? c.danger : c.teal, a: intro });
  text(ctx, "fits in RAM", g.x + g.w - 8, my + 18, { size: 11, family: "mono", color: c.teal, align: "right", a: intro * 0.8 });

  // Storage and GPU: the overflow moves to SSD, then streams to training.
  if (wide) {
    const sx = g.x + g.w + 24;
    const gpu = { x: sx, y: g.y, w: sideW, h: g.h * 0.34 };
    const ssd = { x: sx, y: g.y + g.h * 0.58, w: sideW, h: g.h * 0.42 };
    const a = seg(p, 0.74, 0.8);
    const flow = seg(p, 0.9, 0.96);
    const tick = Math.floor(t * 14);

    // Storage slots, filled bottom-up.
    const SW = 8;
    const SH = 6;
    const scols = Math.floor((ssd.w - 20) / (SW + 2));
    const srows = Math.floor((ssd.h - 36) / (SH + 2));
    const cap = scols * srows;
    const slotPos = (i) => [ssd.x + 10 + (i % scols) * (SW + 2), ssd.y + ssd.h - 8 - (Math.floor(i / scols) + 1) * (SH + 2)];

    // Overflow blocks of every bar beyond memory.
    const blocks = [];
    CORPORA.forEach((k, i) => {
      const size = tb(k.cells);
      if (size <= MEMORY_TB) return;
      const x = g.x + slot * i + (slot - bw) / 2;
      const cols = 5;
      const bwk = (bw - 4) / cols;
      const bhk = bwk * 0.72;
      const rows = Math.floor((my - Y(size) - 2) / bhk);
      for (let r = 0; r < rows; r++)
        for (let q = 0; q < cols; q++) blocks.push({ x: x + 2 + q * bwk, y: my - (r + 1) * bhk, w: bwk - 2, h: bhk - 2, planned: k.planned });
    });
    let arrived = 0;
    blocks.forEach((b, j) => {
      const slotI = Math.floor((j * cap) / blocks.length);
      const d = hash(j, 23) * 0.6;
      const u = clamp((spill - d) / 0.4);
      if (u >= 1) {
        arrived = Math.max(arrived, slotI + 1);
        return;
      }
      const [tx, ty] = slotPos(slotI);
      const e = ease.inOut(u);
      const x = lerp(b.x, tx, e);
      const y = lerp(b.y, ty, e) - Math.sin(u * Math.PI) * 60;
      ctx.fillStyle = alpha(c.danger, (b.planned ? 0.55 : 0.85) * (spill > 0 ? 1 : 0));
      if (spill > 0) ctx.fillRect(x, y, lerp(b.w, SW, e), lerp(b.h, SH, e));
    });
    const filled = spill >= 1 ? cap : arrived;

    if (a > 0) {
      fillRound(ctx, ssd.x, ssd.y, ssd.w, ssd.h, 10, alpha(c.surface, a));
      strokeRound(ctx, ssd.x, ssd.y, ssd.w, ssd.h, 10, alpha(c.danger, a * 0.7), 1.5);
      text(ctx, "NVMe STORAGE", ssd.x + 12, ssd.y + 20, { size: 11, family: "mono", weight: 600, color: c.danger, a });
      // Many blocks flicker at once: each flash is a block being read.
      const lits = [];
      for (let i = 0; i < cap; i++) {
        const [x, y] = slotPos(i);
        if (i >= filled) {
          ctx.fillStyle = alpha(c.divider, 0.35 * a);
          ctx.fillRect(x, y, SW, SH);
          continue;
        }
        const lit = flow > 0 && hash(i, tick) < 0.09;
        if (lit) lits.push([x + SW / 2, y + SH / 2]);
        ctx.fillStyle = lit ? c.accent : alpha(mix(c.danger, c.surface, 0.45), a);
        ctx.fillRect(x, y, SW, SH);
      }
      // GPU: a training step counter, busy tensor cores and a falling loss.
      fillRound(ctx, gpu.x, gpu.y, gpu.w, gpu.h, 10, alpha(c.surface, a));
      strokeRound(ctx, gpu.x, gpu.y, gpu.w, gpu.h, 10, alpha(c.teal, a * 0.8), 1.5);
      text(ctx, "GPU · TRAINING", gpu.x + 12, gpu.y + 20, { size: 11, family: "mono", weight: 600, color: c.teal, a });
      const steps = Math.floor(t * 9 * flow);
      text(ctx, `step ${steps.toLocaleString("en-US")}`, gpu.x + gpu.w - 12, gpu.y + gpu.h - 10, { size: 10, family: "mono", color: c.text3, align: "right", a: a * flow });
      const cores = 6;
      const cs = (gpu.w - 24) / cores;
      for (let k = 0; k < cores * 2; k++) {
        const busy = flow * Math.max(0, Math.sin(t * 6 + k * 1.3));
        ctx.fillStyle = alpha(c.teal, (0.15 + 0.7 * busy) * a);
        ctx.fillRect(gpu.x + 12 + (k % cores) * cs, gpu.y + 32 + Math.floor(k / cores) * (cs * 0.6), cs - 4, cs * 0.6 - 4);
      }
      const ly = gpu.y + 32 + cs * 1.2 + 8;
      const lh = gpu.y + gpu.h - 10 - ly;
      if (lh > 14 && flow > 0) {
        ctx.beginPath();
        for (let k = 0; k <= 40; k++) {
          const u = k / 40;
          const x = gpu.x + 12 + u * (gpu.w - 24);
          const loss = Math.exp(-(u * 2 + t * 0.05 * flow)) * (0.85 + 0.15 * Math.sin(u * 37 + t * 3));
          const y = ly + lh - loss * lh;
          if (k) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        }
        ctx.strokeStyle = alpha(c.accent, a * flow);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        text(ctx, "loss", gpu.x + 12, ly + 8, { size: 9, family: "mono", color: c.text3, a: a * flow });
      }
      // Packets leave from lit blocks and rise to the GPU at their own speeds.
      if (flow > 0) {
        for (let k = 0; k < 40; k++) {
          const speed = 0.45 + hash(k, 7) * 0.7;
          const u = t * speed + hash(k, 11);
          const lap = Math.floor(u);
          const f = u - lap;
          const from = lits.length ? lits[Math.floor(hash(k * 13 + lap, 5) * lits.length)] : [ssd.x + ssd.w / 2, ssd.y + ssd.h / 2];
          const tx = gpu.x + 14 + hash(k, lap + 2) * (gpu.w - 28);
          const e = ease.inOut(f);
          const x = lerp(from[0], tx, e) + Math.sin(f * Math.PI) * (hash(k, 9) - 0.5) * 50;
          const y = lerp(from[1], gpu.y + gpu.h, e);
          ctx.fillStyle = alpha(c.accent, flow * Math.min(1, f * 8, (1 - f) * 8));
          ctx.fillRect(x - SW / 2, y - SH / 2, SW, SH);
        }
        text(ctx, "random reads", sx + sideW / 2, (gpu.y + gpu.h + ssd.y) / 2 + 5, { size: 11, family: "mono", weight: 700, color: c.accent, align: "center", a: flow });
      }
    }
  }
  if (crossed && flash > 0) glow(ctx, g.x + g.w / 2, my, g.w * 0.5, c.danger, 0.15 * flash);
}

export default { draw };
