// Drawing pieces shared by the planning scenes.
import { alpha, fillRound, strokeRound, text } from "../engine.js";

/** An arc above a tape, from x1 to x2, rising `h` above `y`. */
export function arc(ctx, x1, x2, y, h, color, width = 1.5, dash) {
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.bezierCurveTo(x1, y - h * 1.33, x2, y - h * 1.33, x2, y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.stroke();
  ctx.setLineDash([]);
}

/** Diagonal hatching inside a rectangle: frames reserved but not yet holding the page. */
export function hatch(ctx, x, y, w, h, color, step = 5) {
  if (w <= 0 || h <= 0) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = -h; k < w; k += step) {
    ctx.moveTo(x + k, y + h);
    ctx.lineTo(x + k + h, y);
  }
  ctx.stroke();
  ctx.restore();
}

/** A rounded label with a tinted fill. */
export function pill(ctx, x, y, w, h, label, color, { a = 1, size = 10, fill = 0.18, ink, weight = 600 } = {}) {
  if (a <= 0) return;
  fillRound(ctx, x, y, w, h, Math.min(6, h / 2), alpha(color, fill * a));
  strokeRound(ctx, x, y, w, h, Math.min(6, h / 2), alpha(color, 0.7 * a), 1);
  text(ctx, label, x + w / 2, y + h / 2 + size * 0.36, { size, family: "mono", weight, color: ink || color, align: "center", a });
}

/** A bracket under (or over, with negative `dir`) a span, with a centred label. */
export function bracket(ctx, x1, x2, y, color, label, { a = 1, size = 10, dir = 1 } = {}) {
  if (a <= 0) return;
  ctx.globalAlpha = a;
  ctx.beginPath();
  ctx.moveTo(x1, y - 5 * dir);
  ctx.lineTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.lineTo(x2, y - 5 * dir);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.globalAlpha = 1;
  if (label) text(ctx, label, (x1 + x2) / 2, y + (dir > 0 ? size + 5 : -6), { size, family: "mono", color, align: "center", a });
}

/** A small padlock. */
export function lock(ctx, x, y, s, color, a = 1) {
  if (a <= 0) return;
  ctx.globalAlpha = a;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.4, s * 0.14);
  ctx.beginPath();
  ctx.arc(x, y - s * 0.15, s * 0.32, Math.PI, 0);
  ctx.lineTo(x + s * 0.32, y + s * 0.1);
  ctx.moveTo(x - s * 0.32, y - s * 0.15);
  ctx.lineTo(x - s * 0.32, y + s * 0.1);
  ctx.stroke();
  ctx.globalAlpha = 1;
  fillRound(ctx, x - s * 0.5, y + s * 0.05, s, s * 0.62, s * 0.12, alpha(color, a));
}

/** Text that shrinks until it fits `max` pixels. */
export function fit(ctx, value, x, y, max, opts) {
  let size = opts.size || 12;
  const fam = opts.family === "mono" ? "'IBM Plex Mono', monospace" : opts.family === "head" ? "'Space Grotesk Variable', sans-serif" : "'DM Sans Variable', sans-serif";
  ctx.font = `${opts.weight || 500} ${size}px ${fam}`;
  while (size > 7 && ctx.measureText(value).width > max) {
    size -= 0.5;
    ctx.font = `${opts.weight || 500} ${size}px ${fam}`;
  }
  text(ctx, value, x, y, { ...opts, size, maxWidth: Math.max(0, max) });
}
