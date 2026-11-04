import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, mix, seg, strokeRound, text } from "../engine.js";
import { fit } from "./plan-draw.js";

// Why plans are compiled online: at billion-cell scale the access sequence
// alone outgrows host memory, and set-based training multiplies it. The
// sequence is generated from a seed instead, and the compiler keeps a window.
const CELLS = 1e9;
const EPOCHS = 5;
const SET = 512;
const BYTES = 4; // one uint32 CellId per access
const TB = 1e12;
const ROWS = [
  { key: "corpus", label: "1B-cell corpus, compressed", tb: 1.85, note: "report, billion-cell run" },
  { key: "flat", label: `access ids · ${EPOCHS} epochs`, tb: (CELLS * EPOCHS * BYTES) / TB, note: `${EPOCHS}B accesses` },
  { key: "sets", label: `access ids · sets of ${SET}`, tb: (CELLS * EPOCHS * SET * BYTES) / TB, note: `${EPOCHS} × ${SET} × 1B = 2.56T accesses` },
];
const MEMORY = 2; // TB of host memory in a DGX H200, as in chapter I
const PH = { flat: [0.04, 0.16], sets: [0.24, 0.44], online: [0.54, 0.62] };
const fmt = (tb) => (tb >= 1 ? `${tb.toFixed(tb >= 10 ? 1 : 2)} TB` : `${Math.round(tb * 1000)} GB`);

function sizes(ctx, { p, box, c, wide }, a) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  const W = box.w;
  const fs = wide ? 11 : 9;
  const flat = ease.out(seg(p, ...PH.flat));
  const sets = ease.inOut(seg(p, ...PH.sets));
  const nameW = wide ? 210 : 0;
  const x0 = box.x + nameW;
  const maxTB = 11;
  const scale = (tb) => (tb / maxTB) * (W - nameW - 10);
  const rh = clamp(box.h * 0.13, 56, 90);
  const top = box.y + Math.max(60, (box.h - rh * ROWS.length - 60) * 0.45);

  text(ctx, "STORING THE ACCESS SEQUENCE, 1B CELLS", box.x, box.y + 14, { size: fs, family: "mono", weight: 700, color: c.text2 });
  ROWS.forEach((r, k) => {
    const y = top + k * rh;
    const grow = k === 0 ? 1 : k === 1 ? flat : sets;
    const over = r.tb > MEMORY;
    const col = k === 0 ? c.teal : over ? mix(c.accent, c.danger, sets) : c.accent;
    const labelX = wide ? box.x : box.x;
    const labelY = wide ? y + rh * 0.42 : y - 6;
    text(ctx, r.label, labelX, labelY, { size: fs, weight: 600, color: c.text, a: k === 0 ? 1 : Math.max(0.35, grow) });
    const bh = rh * 0.38;
    const by = y + (wide ? rh * 0.18 : 2);
    const w = Math.max(3, scale(r.tb) * grow);
    fillRound(ctx, x0, by, w, bh, 4, alpha(col, 0.75));
    if (over && sets > 0.6) glow(ctx, x0 + w, by + bh / 2, 60, c.danger, 0.3 * sets);
    const size = k === 2 ? fmt(r.tb * grow) : fmt(r.tb);
    const inside = w > 140;
    const after = k === 0 ? Math.max(x0 + w + 8, x0 + scale(MEMORY) + 8) : x0 + w + 8;
    text(ctx, size, inside ? x0 + w - 8 : after, by + bh * 0.7, { size: fs + 3, family: "head", weight: 700, color: inside ? c.bg : col, align: inside ? "right" : "left", a: grow > 0.02 ? 1 : 0 });
    fit(ctx, r.note, x0, by + bh + 14, W - nameW, { size: fs - 1, family: "mono", color: c.text3, a: grow > 0.02 ? 1 : 0 });
  });
  // Host memory, and how far past it the set sequence goes.
  const mx = x0 + scale(MEMORY);
  const bottom = top + ROWS.length * rh;
  line(ctx, mx, top - 16, mx, bottom, alpha(c.danger, 0.85), 1.5, [5, 4]);
  text(ctx, `host memory ${MEMORY} TB`, mx + 6, wide ? top - 8 : bottom + 4, { size: fs - 1, family: "mono", weight: 700, color: c.danger });
  if (sets > 0.9) {
    const ratio = ROWS[2].tb / ROWS[0].tb;
    fit(ctx, `The id list alone is ${ratio.toFixed(1)}× the compressed data, ${(ROWS[2].tb / MEMORY).toFixed(1)}× host memory, before any plan.`, box.x, bottom + 22, W, { size: fs + 1, family: "mono", weight: 700, color: c.danger, a: seg(p, 0.42, 0.48) });
  }
  ctx.restore();
}

// Offline against online as two scans over the same sequence. In both, the
// workers' execution window sweeps right and wraps around. Offline, every id
// and every plan entry is stored before the sweep and stays after it. Online,
// a planner runs a horizon ahead of the workers, generating ids from the seed
// and compiling them; once the workers pass, those ids are dropped.
function stream(ctx, { p, t, box, c, wide }, a) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  const W = box.w;
  const H = box.h;
  const fs = wide ? 11 : 9;
  const offA = ease.out(seg(p, 0.54, 0.62));
  const onA = ease.out(seg(p, 0.76, 0.84));
  const factsA = ease.out(seg(p, 0.86, 0.94));
  const n = wide ? 48 : 26;
  const EXEC = wide ? 5 : 3;
  const HZN = wide ? 11 : 5;
  const cw = W / n;
  const th = clamp(H * 0.06, 22, 34);
  const rowGap = clamp(H * 0.3, 150, 210);
  const y1 = box.y + Math.max(46, H * 0.14);
  const y2 = y1 + rowGap;
  const pos = (t * 6) % n; // left edge of the execution window, in cells
  const hue = (i) => `hsla(${(hash(i, 4) * 360) | 0}, 34%, 58%, 0.85)`;
  const rel = (i) => (((i - pos) % n) + n) % n; // distance ahead of the window's left edge

  // A window drawn over [from, from + len) cells, split where it wraps.
  const windowBox = (y, from, len, col, label, labelAbove) => {
    const parts = [];
    const s0 = ((from % n) + n) % n;
    if (s0 + len <= n) parts.push([s0, len]);
    else parts.push([s0, n - s0], [0, len - (n - s0)]);
    parts.forEach(([s, l]) => {
      fillRound(ctx, box.x + s * cw, y - 5, l * cw, th + 10, 5, alpha(col, 0.1));
      strokeRound(ctx, box.x + s * cw, y - 5, l * cw, th + 10, 5, alpha(col, 0.95), 1.8);
    });
    const [s, l] = parts[0][1] >= (parts[1]?.[1] ?? 0) ? parts[0] : parts[1];
    fit(ctx, label, box.x + s * cw + 3, labelAbove ? y - 10 : y + th + 18, Math.max(l * cw, 140), { size: fs - 1, family: "mono", weight: 700, color: col });
  };

  // Offline: the whole sequence and its plan, stored before the sweep.
  if (offA > 0) {
    ctx.globalAlpha = a * offA;
    text(ctx, "OFFLINE · stored sequence, precompiled plan", box.x, y1 - 30, { size: fs, family: "mono", weight: 700, color: c.text });
    for (let i = 0; i < n; i++) {
      const r = rel(i);
      const inExec = r < EXEC;
      fillRound(ctx, box.x + i * cw + 1, y1, cw - 3, th, 2, hue(i));
      if (inExec) fillRound(ctx, box.x + i * cw + 1, y1, cw - 3, th, 2, alpha(c.bg, 0.0));
      fillRound(ctx, box.x + i * cw + 2, y1 + th + 3, cw - 5, 3, 1, alpha(c.teal, 0.7));
    }
    windowBox(y1, pos, EXEC, c.teal, "workers", true);
    fit(ctx, `held in memory: all ${n} of ${n} ids + the whole plan · at 1B cells × sets of 512: 10.2 TB`, box.x, y1 + th + 28, W, { size: fs - 1, family: "mono", weight: 700, color: c.danger });
    const held = 1;
    fillRound(ctx, box.x, y1 + th + 38, W * held, 6, 3, alpha(c.danger, 0.7));
  }

  // Online: the planner a horizon ahead generates and compiles; the workers'
  // trail is dropped.
  if (onA > 0) {
    ctx.globalAlpha = a * onA;
    text(ctx, "ONLINE · planner ahead, ids dropped behind", box.x, y2 - 30, { size: fs, family: "mono", weight: 700, color: c.text });
    const front = EXEC + HZN; // cells from the window's left edge to the planner's front
    let held = 0;
    for (let i = 0; i < n; i++) {
      const r = rel(i);
      const x = box.x + i * cw;
      if (r < front) {
        held++;
        const born = clamp((front - r) / 1.2);
        fillRound(ctx, x + 1, y2 + (1 - born) * 8, cw - 3, th, 2, hue(i));
        if (r < front - 1) fillRound(ctx, x + 2, y2 + th + 3, cw - 5, 3, 1, alpha(c.teal, 0.7));
      } else if (r > n - 3) {
        // Just passed by the workers: dropping.
        const fade = (n - r) / 3;
        fillRound(ctx, x + 1, y2 + th * 0.5 * (1 - fade), cw - 3, th * fade, 2, alpha(c.text3, 0.45 * fade));
        strokeRound(ctx, x + 1.5, y2 + 0.5, cw - 4, th - 1, 2, alpha(c.border, 0.7), 1);
      } else {
        strokeRound(ctx, x + 1.5, y2 + 0.5, cw - 4, th - 1, 2, alpha(c.border, 0.7), 1);
      }
    }
    windowBox(y2, pos + EXEC, HZN, c.accent, "planner · horizon H", true);
    windowBox(y2, pos, EXEC, c.teal, "workers", true);
    // The generator at the planner's front.
    const fx = box.x + (((pos + front) % n) + 0) * cw;
    glow(ctx, fx, y2 + th / 2, 26, c.accent, 0.5);
    fit(ctx, "shuffle(seed, i) → new id", Math.min(fx + 4, box.x + W - 170), y2 + th + 22, 170, { size: fs - 1, family: "mono", weight: 700, color: c.accent });
    fit(ctx, `held in memory: ${held} of ${n} ids, the horizon and the window · billion-cell run: 4 GB`, box.x, y2 + th + 44, W, { size: fs - 1, family: "mono", weight: 700, color: c.teal });
    fillRound(ctx, box.x, y2 + th + 54, W, 6, 3, alpha(c.divider, 0.8));
    fillRound(ctx, box.x, y2 + th + 54, W * (held / n), 6, 3, alpha(c.teal, 0.8));
  }
  ctx.globalAlpha = a;

  if (factsA > 0) {
    const fy = y2 + th + 92;
    const lines = [
      ["memory grows with H, not with the length of training", c.teal, 700],
      ["billion-cell run: H = 1B accesses ahead, 8 planner cores", c.text, 500],
      ["first batch 106 s after start; 5B cells in 25.7 min", c.text2, 500],
    ];
    lines.forEach(([l, col, w], k) => fit(ctx, l, box.x, fy + k * 21, W, { size: fs + (k === 0 ? 1 : 0), family: "mono", weight: w, color: col, a: factsA }));
  }
  ctx.restore();
}

function draw(ctx, env) {
  const to = ease.inOut(seg(env.p, ...PH.online));
  sizes(ctx, env, 1 - to);
  stream(ctx, env, to);
}

export default { draw };
