// Shared helpers for the canvas scenes. Every scene is a pure function of scroll
// progress `p` (0..1) plus an ambient clock `t` (seconds) for idle motion.

export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
/** Local progress of `p` inside [a, b]. */
export const seg = (p, a, b) => clamp((p - a) / (b - a));
export const ease = {
  in: (t) => t * t * t,
  out: (t) => 1 - (1 - t) ** 3,
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
  back: (t) => 1 + 2.4 * (t - 1) ** 3 + 1.4 * (t - 1) ** 2,
};
/** Deterministic pseudo-random number in [0, 1) for integer inputs. */
export function hash(a, b = 0) {
  let h = (a * 374761393 + b * 668265263) ^ 0x5bd1e995;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const parser = typeof document !== "undefined" ? document.createElement("canvas").getContext("2d") : null;
const rgbCache = new Map();
/** Convert any CSS colour to [r, g, b]. */
export function rgb(color) {
  if (rgbCache.has(color)) return rgbCache.get(color);
  parser.fillStyle = "#000";
  parser.fillStyle = color;
  const value = parser.fillStyle;
  let out;
  if (value.startsWith("#")) out = [1, 3, 5].map((i) => parseInt(value.slice(i, i + 2), 16));
  else out = value.match(/[\d.]+/g).slice(0, 3).map(Number);
  rgbCache.set(color, out);
  return out;
}
export const alpha = (color, a) => {
  const [r, g, b] = rgb(color);
  return `rgba(${r},${g},${b},${clamp(a)})`;
};
export function mix(c1, c2, t) {
  const a = rgb(c1), b = rgb(c2);
  return `rgb(${a.map((v, i) => Math.round(lerp(v, b[i], clamp(t)))).join(",")})`;
}

export const FONT = {
  head: "'Space Grotesk Variable', 'DM Sans Variable', sans-serif",
  body: "'DM Sans Variable', sans-serif",
  mono: "'IBM Plex Mono', ui-monospace, monospace",
};
export function font(ctx, size, family = "body", weight = 500) {
  ctx.font = `${weight} ${size}px ${FONT[family]}`;
}
export function text(ctx, value, x, y, { size = 14, family = "body", weight = 500, color, align = "left", baseline = "alphabetic", a = 1, maxWidth } = {}) {
  if (a <= 0) return;
  font(ctx, size, family, weight);
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillStyle = a < 1 ? alpha(color, a) : color;
  if (maxWidth === undefined) ctx.fillText(value, x, y);
  else if (maxWidth > 0) ctx.fillText(value, x, y, maxWidth);
}

/** Wrap annotations into their reserved column, keeping the chosen type size. */
export function paragraph(ctx, value, x, y, maxWidth, opts = {}) {
  font(ctx, opts.size || 12, opts.family || "body", opts.weight || 500);
  const lines = [];
  let line = "";
  for (const word of String(value).split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > maxWidth) { lines.push(line); line = word; }
    else line = next;
  }
  if (line) lines.push(line);
  const leading = opts.leading || (opts.size || 12) * 1.45;
  lines.forEach((line, i) => text(ctx, line, x, y + i * leading, opts));
  return lines.length * leading;
}
export function roundRect(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}
export function fillRound(ctx, x, y, w, h, r, color) {
  if (w <= 0 || h <= 0) return;
  roundRect(ctx, x, y, w, h, r);
  ctx.fillStyle = color;
  ctx.fill();
}
export function strokeRound(ctx, x, y, w, h, r, color, width = 1) {
  if (w <= 0 || h <= 0) return;
  roundRect(ctx, x, y, w, h, r);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}
export function line(ctx, x1, y1, x2, y2, color, width = 1, dash) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.stroke();
  ctx.setLineDash([]);
}
/** A soft radial glow, used to mark the focus of a scene. */
export function glow(ctx, x, y, r, color, a = 0.35) {
  if (a <= 0 || r <= 0) return;
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, alpha(color, a));
  g.addColorStop(1, alpha(color, 0));
  ctx.fillStyle = g;
  ctx.fillRect(x - r, y - r, r * 2, r * 2);
}

// Round half up on the scaled integer, so 1,165,000 reads as 1.17M.
const fixed = (n, unit, digits) => (Math.round(n / (unit / 10 ** digits)) / 10 ** digits).toFixed(digits);
export function compact(n, digits = 2) {
  if (n >= 1e9) return `${fixed(n, 1e9, digits)}B`;
  if (n >= 1e6) return `${fixed(n, 1e6, digits)}M`;
  if (n >= 1e3) return `${fixed(n, 1e3, n >= 1e5 ? 0 : 1)}K`;
  return n.toFixed(0);
}
export const grouped = (n) => Math.round(n).toLocaleString("en-US");

/**
 * Packets flowing from x0 to x1 along a lane. Each packet has its own speed.
 * `pass(x)` gives the fraction of packets that survive past x (supply limits).
 */
export function packets(ctx, { x0, x1, y, h, t, seed = 0, gap = 34, speed = 90, color, pass, alphaScale = 1, length = 1.5, radius }) {
  const L = x1 - x0;
  if (L <= h * 2) return;
  const n = Math.max(1, Math.round(L / gap));
  const w = h * length;
  for (let k = 0; k < n; k++) {
    const r1 = hash(seed * 131 + k, 17);
    const r2 = hash(k, seed * 7 + 3);
    const v = speed * (0.6 + 0.8 * r1);
    const u = (t * v) / L + r2;
    const lap = Math.floor(u);
    const x = x0 + (u - lap) * (L - w);
    // A packet that loses the draw at the supply line vanishes there.
    if (pass) {
      const keep = pass(x);
      if (keep < 1 && hash(k * 977 + lap, seed + 41) > keep) continue;
    }
    const edge = Math.min(1, (x - x0) / 24, (x1 - w - x) / 24);
    if (edge <= 0) continue;
    ctx.globalAlpha = edge * alphaScale;
    fillRound(ctx, x, y - h / 2, w, h, radius ?? Math.min(3, h / 2), color);
  }
  ctx.globalAlpha = 1;
}
