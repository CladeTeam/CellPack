import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, mix, seg, strokeRound, text, paragraph } from "../engine.js";
import { CSR, FINAL } from "./fmt-common.js";

// Top: a .cpk stream. Right: one chunk, stacked as in the report figure —
// header, shared models, offset tables, then 4 KiB pages of records.
// Left: the familiar AnnData picture, where only X changes, into a much
// smaller CellPack stream.
const STREAM = [
  ["superblock", 0.11, "text"],
  ["chunk 0", 0.21, "teal"],
  ["chunk 1", 0.21, "teal"],
  ["…", 0.08, "text3"],
  ["chunk n", 0.21, "teal"],
  ["directory", 0.14, "accent"],
];
const PAGE_ROWS = 4;
const RECORDS = Array.from({ length: PAGE_ROWS }, (_, r) => {
  const out = [];
  let used = 0;
  for (let k = 0; ; k++) {
    const w = 0.18 + hash(r * 7 + k, 3) * 0.32;
    if (used + w > 0.97) break;
    out.push([used, w]);
    used += w;
  }
  return out;
});
// AnnData's own colours, muted to sit on either theme.
const HUE = { obsp: "#d9695f", obsm: "#e3913f", X: "#5fae74", obs: "#d9b44a", var: "#4f97c2", varm: "#3f6f86", varp: "#8d6ab3" };

function block(ctx, x, y, w, h, color, label, a) {
  fillRound(ctx, x, y, w, h, 4, alpha(color, 0.8 * a));
  text(ctx, label, x + w / 2, y + h / 2 + 5, { size: 14, family: "mono", weight: 700, color: "#ffffff", align: "center", a });
}

function draw(ctx, { p, t, box, c, wide }) {
  const stream = ease.out(seg(p, 0.02, 0.12));
  const open = ease.inOut(seg(p, 0.16, 0.3));
  const ann = ease.out(seg(p, 0.42, 0.52));
  const swap = ease.inOut(seg(p, 0.6, 0.76));
  const fs = wide ? 12 : 9;

  // Sizes first, so the whole composition can be centred vertically.
  const sh = 40;
  const cw = wide ? box.w * 0.42 : box.w;
  const rowH = wide ? Math.min(40, box.h * 0.055) : Math.max(10, Math.min(24, (box.h - 408) / 7));
  const chunkH = (rowH + 4) * 3 + 8 + PAGE_ROWS * (rowH + 6) + 20;
  const aw = wide ? box.w - cw - 56 : box.w;
  const U = Math.min(aw / 4.3, (wide ? chunkH + 40 : Math.max(100, box.h - chunkH - 210)) / 3.5);
  const annH = wide ? U * 3.5 + 16 : 108;
  const mainH = wide ? Math.max(chunkH, annH) : chunkH + 40 + annH;
  const total = 42 + sh + 36 + mainH + 54;
  ctx.translate(0, Math.max(0, (box.h - total) / 2));

  // 1. The stream.
  paragraph(ctx, ".cpk · 4 KiB superblock · 4,096-cell chunks · directory", box.x, box.y + 10, box.w, { size: 11, family: "mono", color: c.text2, a: stream });
  const sy = box.y + 42;
  let x = box.x;
  const parts = STREAM.map(([name, f, tone], i) => {
    const w = box.w * f;
    const a = clamp(stream * STREAM.length - i * 0.7);
    if (name !== "…") {
      fillRound(ctx, x, sy, w - 6, sh, 6, alpha(c[tone], 0.16 * a));
      strokeRound(ctx, x, sy, w - 6, sh, 6, alpha(c[tone], 0.8 * a), 1.2);
    }
    text(ctx, name === "superblock" && !wide ? "4 KiB" : name, x + (w - 6) / 2, sy + sh / 2 + 4, { maxWidth: w - 8, size: fs, family: "mono", weight: 600, color: c[tone], align: "center", a });
    const r = { x, w: w - 6 };
    x += w;
    return r;
  });

  // 2. One chunk, stacked.
  const cx = box.x + box.w - cw;
  const cy = sy + sh + 36;
  if (open > 0) {
    const src = parts[4];
    ctx.globalAlpha = open * 0.5;
    line(ctx, src.x, sy + sh, cx, cy - 6, c.teal, 1, [3, 3]);
    line(ctx, src.x + src.w, sy + sh, cx + cw, cy - 6, c.teal, 1, [3, 3]);
    ctx.globalAlpha = 1;
    const rows = [
      [["header · 64 B", 1, "text"]],
      [["rANS models", 0.4, "teal"], ["gene order", 0.32, "teal"], ["column modes", 0.28, "accent"]],
      [["offsets · lengths · kinds · cumulative nnz", 1, "accent"]],
    ];
    let y = cy;
    rows.forEach((cells, i) => {
      const a = clamp(open * 4 - i * 0.6);
      let xx = cx;
      cells.forEach(([name, f, tone]) => {
        const w = cw * f;
        fillRound(ctx, xx, y, w - 4, rowH, 5, alpha(c[tone], 0.16 * a));
        strokeRound(ctx, xx, y, w - 4, rowH, 5, alpha(c[tone], 0.75 * a), 1);
        text(ctx, name, xx + (w - 4) / 2, y + rowH / 2 + 4, { size: wide ? 12 : 9, family: "mono", color: c[tone], align: "center", a });
        xx += w;
      });
      y += rowH + 4;
    });
    y += 8;
    for (let r = 0; r < PAGE_ROWS; r++) {
      const a = clamp(open * 4 - (3 + r) * 0.4);
      const py = y + r * (rowH + 6);
      strokeRound(ctx, cx, py, cw - 4, rowH, 5, alpha(c.border, a), 1);
      RECORDS[r].forEach(([u, w], k) => {
        fillRound(ctx, cx + 3 + u * (cw - 10), py + 4, w * (cw - 10) - 3, rowH - 8, 3, alpha(mix(c.teal, c.tealSoft, (k % 3) * 0.25), a));
      });
      const end = RECORDS[r].reduce((m, [u, w]) => Math.max(m, u + w), 0);
      for (let hx = cx + 3 + end * (cw - 10); hx < cx + cw - 10; hx += 5) line(ctx, hx, py + rowH - 5, hx + 5, py + 5, alpha(c.text3, 0.4 * a), 1);
      text(ctx, `p${r}`, cx + 6, py + rowH / 2 + 4, { size: 9, family: "mono", color: c.bg, a });
    }
    text(ctx, "4 KiB pages · ▨ padding", cx, y + PAGE_ROWS * (rowH + 6) + 12, { size: 10, family: "mono", color: c.text3, a: open });
  }

  // 3. AnnData, as usually drawn: only X changes.
  const ax = box.x;
  const ay = wide ? cy + 12 : cy + chunkH + 40;
  if (ann > 0 && wide) {
    const g = 6;
    const X = { x: ax + 1.1 * U + 0.6 * U + 2 * g, y: ay + 0.42 * U + g, w: 1.3 * U, h: 1.1 * U };
    const a = ann;
    text(ctx, "ANNDATA", ax, ay - 8, { size: 11, family: "mono", weight: 700, color: c.text2, a });
    block(ctx, X.x, ay, X.w, 0.42 * U, HUE.var, "var", a);
    block(ctx, ax, X.y, 1.1 * U, X.h, HUE.obsp, "obsp", a);
    block(ctx, ax + 1.1 * U + g, X.y, 0.6 * U, X.h, HUE.obsm, "obsm", a);
    block(ctx, X.x + X.w + g, X.y, 0.5 * U, X.h, HUE.obs, "obs", a);
    block(ctx, X.x, X.y + X.h + g + 30, X.w, 0.42 * U, HUE.varm, "varm", a);
    block(ctx, X.x, X.y + X.h + 2 * g + 30 + 0.42 * U, X.w, 0.9 * U, HUE.varp, "varp", a);
    text(ctx, "layers", X.x + 4, X.y + X.h + 21, { size: 11, family: "mono", weight: 500, color: c.text2, a });
    text(ctx, "{ uns … }", ax + 0.25 * U, X.y + X.h + 0.95 * U, { size: 18, family: "mono", weight: 700, color: c.text2, a });
    for (let k = 2; k >= 1; k--) fillRound(ctx, X.x - k * 3, X.y + k * 3, X.w, X.h, 4, alpha(HUE.X, 0.35 * a));
    // X: CSR arrays, replaced by a much smaller CellPack stream in its corner.
    const side = Math.sqrt(FINAL / CSR);
    const inset = 6 * swap;
    const xw = Math.max(70, X.w * side);
    const xh = Math.max(40, X.h * side);
    const w = lerp(X.w, xw, swap);
    const h = lerp(X.h, xh, swap);
    if (swap > 0) {
      fillRound(ctx, X.x - 6, X.y - 2, X.w + 8, X.h + 10, 4, alpha(c.bg, swap));
      strokeRound(ctx, X.x, X.y, X.w, X.h, 4, alpha(c.text3, 0.7 * swap), 1);
    }
    fillRound(ctx, X.x + inset, X.y + inset, w, h, 4, alpha(swap > 0.5 ? c.teal : HUE.X, 0.92 * a));
    if (swap < 0.5) {
      const ia = a * (1 - swap * 2);
      text(ctx, "X · CSR", X.x + 10, X.y + 20, { size: 13, family: "mono", weight: 700, color: "#ffffff", a: ia });
      const bh = (X.h - 40) / 3 - 5;
      ["indptr", "indices", "data"].forEach((n, i) => {
        const yy = X.y + 32 + i * (bh + 5);
        fillRound(ctx, X.x + 8, yy, X.w - 16, bh, 3, alpha("#ffffff", 0.3 * ia));
        text(ctx, n, X.x + X.w / 2, yy + bh / 2 + 4, { size: 10, family: "mono", weight: 600, color: "#ffffff", align: "center", a: ia });
      });
    } else {
      const ia = a * seg(swap, 0.5, 1);
      text(ctx, ".cpk", X.x + inset + xw / 2, X.y + inset + xh / 2 + 5, { size: 15, family: "mono", weight: 700, color: c.bg, align: "center", a: ia });
      const ly = X.y + inset + xh + (X.h - xh - inset) / 2 + 2;
      text(ctx, `${(CSR / FINAL).toFixed(1)}×`, X.x + X.w / 2, ly, { size: 16, family: "head", weight: 700, color: c.teal, align: "center", a: ia });
      text(ctx, "smaller", X.x + X.w / 2, ly + 14, { size: 10, family: "mono", color: c.text2, align: "center", a: ia });
    }
    if (swap > 0) {
      // The whole .cpk — stream and chunk — is outlined and drops into X.
      const u = ease.inOut(seg(swap, 0, 0.4));
      const pts = [
        [box.x - 8, sy - 32],
        [box.x + box.w + 8, sy - 32],
        [box.x + box.w + 8, cy + chunkH + 4],
        [cx - 56, cy + chunkH + 4],
        [cx - 56, sy + sh + 12],
        [box.x - 8, sy + sh + 12],
      ];
      ctx.beginPath();
      pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.closePath();
      ctx.strokeStyle = alpha(c.teal, 0.85 * u);
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      const ox = X.x + inset + xw / 2;
      line(ctx, ox, sy + sh + 12, ox, lerp(sy + sh + 12, X.y + inset, u), alpha(c.teal, 0.85 * u), 2, [6, 5]);
      glow(ctx, X.x + 30, X.y + 20, X.w * 0.6, c.teal, 0.1 * Math.sin(swap * Math.PI) + 0.03 * Math.sin(t * 2));
    }
  }
  if (ann > 0 && !wide) {
    const x0 = ax + box.w * 0.22, xw = box.w * 0.62;
    text(ctx, "ANNDATA", ax, ay - 8, { size: 10, family: "mono", weight: 700, color: c.text2, a: ann });
    fillRound(ctx, x0, ay, xw, 16, 3, alpha(HUE.var, ann * 0.8));
    text(ctx, "var", x0 + xw / 2, ay + 12, { size: 10, color: "#fff", align: "center", a: ann });
    fillRound(ctx, ax, ay + 22, box.w * 0.18, 40, 3, alpha(HUE.obs, ann * 0.8));
    text(ctx, "obs", ax + box.w * 0.09, ay + 46, { size: 11, color: c.text, align: "center", a: ann });
    strokeRound(ctx, x0, ay + 22, xw, 40, 3, alpha(c.border, ann));
    const width = lerp(xw, Math.max(48, xw * Math.sqrt(FINAL / CSR)), swap);
    fillRound(ctx, x0, ay + 22, width, 40, 3, alpha(c.teal, ann * 0.9));
    text(ctx, swap > 0.5 ? ".cpk" : "X · CSR", x0 + width / 2, ay + 46, { size: 12, family: "mono", color: c.bg, align: "center", a: ann });
    if (swap > 0.5) text(ctx, `${(CSR / FINAL).toFixed(1)}× smaller`, x0 + xw - 4, ay + 46, { size: 10, family: "mono", color: c.teal, align: "right", a: ann });
    text(ctx, "obsp · obsm · varm · varp", ax, ay + 82, { size: 10, family: "mono", color: c.text2, a: ann });
    text(ctx, "layers · uns · annotations unchanged", ax, ay + 100, { size: 10, family: "mono", color: c.text2, a: ann, maxWidth: box.w });
  }
  const cap = seg(p, 0.8, 0.88);
  paragraph(ctx, "H5CP: X is a .cpk stream; annotations keep their AnnData encoding.", box.x, cy + mainH + 26, box.w, { size: wide ? 12 : 10, family: "mono", color: c.teal, a: cap });
}

export default { draw };
