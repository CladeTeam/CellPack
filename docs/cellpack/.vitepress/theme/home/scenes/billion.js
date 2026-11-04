import data from "../scenes.json";
import { alpha, clamp, compact, ease, fillRound, glow, grouped, lerp, line, packets, paragraph, seg, text } from "../engine.js";

// The report's billion-cell run (Fig. 7): five epochs over 10⁹ cells from two
// NVMe SSDs in RAID-0 with a 1 TB cache, replayed from its per-second log.
const B = data.billion;
const S = B.series;
const T_END = B.epochs.at(-1).end;
const PEAK = Math.max(...S.map((r) => r[2]));
const READ_PEAK = (2.6e6 * 4096) / 1e9;
const MEMORY = 1.887;
const CURVES = [
  { key: 2, ref: PEAK, label: "throughput", token: "accent" },
  { key: 4, ref: 1, label: "page hits", token: "teal" },
  { key: 3, ref: READ_PEAK, label: "device reads", token: "text3" },
  { key: 5, ref: MEMORY, label: "resident memory", token: "text2" },
];

function at(sec) {
  if (sec <= S[0][0]) return S[0];
  for (let i = 1; i < S.length; i++)
    if (sec <= S[i][0]) {
      const a = S[i - 1];
      const b = S[i];
      const u = (sec - a[0]) / (b[0] - a[0]);
      return a.map((v, k) => lerp(v, b[k], u));
    }
  return S.at(-1);
}

function clock(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function draw(ctx, { p, t, box, c, wide }) {
  const play = seg(p, 0.05, 0.9);
  const sec = play * T_END;
  const now = at(sec);
  const intro = ease.out(seg(p, 0, 0.05));
  const outro = ease.inOut(seg(p, 0.9, 0.98));
  const epoch = B.epochs.findIndex((e) => sec <= e.end) + 1 || B.epochs.length;

  // Counters.
  const finished = sec >= T_END - 0.001;
  const counters = [
    ["CELLS DELIVERED", grouped(finished ? 5e9 : now[1]), c.text],
    [finished ? "E5 MEAN RATE" : "THROUGHPUT", `${compact(finished ? B.last_epoch.cells_per_s : now[2], 2)}/s`, c.accent],
    [finished ? "E5 MEAN HITS" : "LIVE PAGE HITS", `${((finished ? B.last_epoch.page_hit_rate : now[4]) * 100).toFixed(0)}%`, c.teal],
    ["ELAPSED", clock(sec), c.text2],
  ];
  const cols = wide ? 4 : 2;
  // The cell counter needs the widest column.
  const widths = wide ? [0.37, 0.25, 0.2, 0.18] : [0.58, 0.42, 0.58, 0.42];
  const big = Math.min(wide ? 34 : 24, (box.w * widths[0]) / 8.2);
  counters.forEach(([label, value, color], i) => {
    const row = Math.floor(i / cols);
    const x = box.x + widths.slice(row * cols, i).reduce((a, w) => a + w * box.w, 0);
    const y = box.y + row * (big + 34);
    text(ctx, label, x, y + 10, { size: 11, family: "mono", color: c.text3, a: intro });
    text(ctx, value, x, y + 16 + big, { size: big, family: "head", weight: 500, color, a: intro });
  });
  const headH = Math.ceil(4 / cols) * (big + 34) + 10;

  // Cells streaming from storage to the model at the current rate.
  const fy = box.y + headH + 12;
  const fh = Math.min(64, box.h * 0.12);
  fillRound(ctx, box.x, fy, box.w, fh, 10, alpha(c.surface, intro));
  text(ctx, wide ? "RAID-0 · 1 TB CACHE" : "SSD", box.x + 12, fy + fh / 2 + 4, { size: 11, family: "mono", color: c.text3, a: intro });
  text(ctx, "→ TRAINING", box.x + box.w - 12, fy + fh / 2 + 4, { size: 11, family: "mono", weight: 600, color: c.text2, align: "right", a: intro });
  const rate = now[2] / PEAK;
  // Each lane carries cells: teal from cached pages, orange read from SSD.
  // Density follows the current throughput; every packet has its own speed.
  const lanes = 4;
  const fx0 = box.x + (wide ? 170 : 130);
  const fx1 = box.x + box.w - (wide ? 110 : 90);
  for (let l = 0; l < lanes; l++) {
    const y = fy + 12 + (l + 0.5) * ((fh - 24) / lanes);
    const h = Math.max(5, ((fh - 24) / lanes) * 0.55);
    const common = { x0: fx0, x1: fx1, y, h, t, gap: 18, speed: 120 };
    packets(ctx, { ...common, seed: l * 2 + 1, color: c.teal, pass: () => rate * now[4] });
    packets(ctx, { ...common, seed: l * 2 + 2, color: c.accent, pass: () => rate * (1 - now[4]) });
  }

  // Timeline chart.
  const legendY = fy + fh + 20;
  if (!wide) CURVES.forEach((curve, i) => {
    const x = box.x + (i % 2) * box.w / 2;
    const y = legendY + Math.floor(i / 2) * 18;
    line(ctx, x, y - 3, x + 12, y - 3, c[curve.token], 2);
    text(ctx, curve.label, x + 18, y, { size: 10, color: c[curve.token], a: intro });
  });
  const gy = fy + fh + (wide ? 34 : 72);
  const gh = box.y + box.h - gy - 54;
  const labelW = wide ? 150 : 0;
  const gw = box.w - labelW;
  const X = (s) => box.x + (s / T_END) * gw;
  const Y = (v) => gy + gh - clamp(v, 0, 1.12) / 1.12 * gh;
  line(ctx, box.x, Y(0), box.x + gw, Y(0), alpha(c.border, intro), 1);
  line(ctx, box.x, Y(1), box.x + gw, Y(1), alpha(c.divider, intro), 1, [2, 4]);
  text(ctx, "100% of reference", box.x, Y(1) - 6, { size: 10, family: "mono", color: c.text3, a: intro });
  let start = 0;
  B.epochs.forEach((e, i) => {
    const shown = sec >= start;
    if (e.end < T_END) line(ctx, X(e.end), gy - 10, X(e.end), Y(0), alpha(c.text3, shown ? 0.6 : 0.15), 1, [2, 3]);
    const mid = (X(start) + X(e.end)) / 2;
    const done = sec >= e.end;
    text(ctx, `E${e.epoch}`, mid, gy - 2, { size: 11, family: "mono", weight: 600, color: i + 1 === epoch && !done ? c.accent : c.text3, align: "center", a: intro });
    if (done && wide) text(ctx, `${(e.rate / 1e6).toFixed(2)}M/s`, mid, gy + 12, { size: 10, family: "mono", color: c.text2, align: "center", a: clamp((sec - e.end) / 40) });
    start = e.end;
  });
  CURVES.forEach((curve) => {
    const color = c[curve.token];
    ctx.beginPath();
    let started = false;
    for (const r of S) {
      if (r[0] > sec) break;
      const x = X(r[0]);
      const y = Y(r[curve.key] / curve.ref);
      if (!started) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      started = true;
    }
    if (started) {
      const y = Y(now[curve.key] / curve.ref);
      ctx.lineTo(X(Math.min(sec, S.at(-1)[0])), y);
      ctx.strokeStyle = color;
      ctx.lineWidth = curve.key === 2 ? 2.4 : 1.5;
      ctx.lineJoin = "round";
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(X(Math.min(sec, S.at(-1)[0])), y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  });
  if (wide) {
    // End labels follow each curve's head, spread apart.
    const heads = CURVES.map((curve) => ({ curve, y: Y(now[curve.key] / curve.ref) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < heads.length; i++) heads[i].y = Math.max(heads[i].y, heads[i - 1].y + 15);
    heads.forEach(({ curve, y }) => {
      const value = curve.key === 2 ? `${compact(now[2], 2)}/s` : curve.key === 4 ? `${(now[4] * 100).toFixed(0)}%` : curve.key === 3 ? `${now[3].toFixed(1)} GB/s` : `${now[5].toFixed(2)} TB`;
      text(ctx, `${value} ${curve.label}`, box.x + gw + 12, y + 4, { size: 10, family: "mono", weight: 500, color: c[curve.token], a: intro * (sec > 1 ? 1 : 0), maxWidth: labelW - 12 });
    });
  }
  text(ctx, "minutes since loader start", box.x + gw, Y(0) + 36, { size: 10, family: "mono", color: c.text3, align: "right", a: intro });
  for (let m = 0; m <= T_END / 60 + 0.01; m += 5) text(ctx, String(m), X(m * 60), Y(0) + 20, { size: 10, family: "mono", color: c.text3, align: "center", a: intro });

  // Finale.
  if (outro > 0) {
    fillRound(ctx, box.x - 10, gy - 20, box.w + 20, gh + 40, 14, alpha(c.bg, outro * 0.86));
    const size = Math.min(64, box.w / 9);
    const cy = gy + gh / 2;
    glow(ctx, box.x + box.w / 2, cy, box.w * 0.4, c.accent, 0.18 * outro);
    text(ctx, "5 billion cells", box.x + box.w / 2, cy - 6, { size, family: "head", color: c.text, align: "center", a: outro });
    paragraph(ctx, `in ${(T_END / 60).toFixed(1)} minutes · epoch 5 means: ${(B.last_epoch.cells_per_s / 1e6).toFixed(2)}M cells/s · ${Math.round(B.last_epoch.page_hit_rate * 100)}% page hits`, box.x + box.w / 2, cy + 28, box.w - 24, { size: wide ? 14 : 11, family: "mono", color: c.accent, align: "center", a: outro });
  }
}

export default { draw };
