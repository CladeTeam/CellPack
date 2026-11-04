import { alpha, clamp, ease, fillRound, glow, hash, lerp, mix, seg, strokeRound, text } from "../engine.js";

// The loader's modules, following one access from the sequence to the model.
// The module in focus is lit; everything else keeps working faintly behind.
const PH = {
  plan: [0.18, 0.3],
  claim: [0.36, 0.46],
  submit: [0.52, 0.58],
  read: [0.58, 0.66],
  decode: [0.68, 0.78],
  out: [0.82, 0.9],
};
const STAGES = ["decode", "map genes", "convert", "process fn"];

function card(ctx, c, x, y, w, h, label, color, a = 1) {
  fillRound(ctx, x - w / 2, y - h / 2, w, h, 6, alpha(color, 0.95 * a));
  text(ctx, label, x, y + 4, { size: 11, family: "mono", weight: 700, color: c.bg, align: "center", a });
}

function module(ctx, c, r, title, sub, focus, color) {
  const a = 0.35 + 0.65 * focus;
  fillRound(ctx, r.x, r.y, r.w, r.h, 10, alpha(c.bg, 1));
  fillRound(ctx, r.x, r.y, r.w, r.h, 10, alpha(color, 0.06 + 0.1 * focus));
  strokeRound(ctx, r.x, r.y, r.w, r.h, 10, alpha(color, a), 1 + focus);
  text(ctx, title, r.x + 12, r.y + 20, { size: 12, family: "mono", weight: 700, color, a });
  if (sub) text(ctx, sub, r.x + 12, r.y + 36, { size: 10, family: "mono", color: c.text3, a });
}

function draw(ctx, { p, t, box, c, wide }) {
  const W = box.w;
  const H = box.h;
  if (W < 620) return compact(ctx, { p, t, box, c });
  const ph = (k) => ease.inOut(seg(p, ...PH[k]));
  const on = (a, b) => seg(p, a - 0.03, a) * (1 - seg(p, b, b + 0.03));
  const focus = {
    seq: on(0, 0.18),
    plan: on(0.18, 0.34),
    frames: Math.max(on(0.18, 0.34) * 0.6, on(0.66, 0.8)),
    worker: on(0.36, 0.8),
    io: on(0.52, 0.68),
    out: on(0.8, 1.05),
  };

  // Layout.
  const tape = { x: box.x, y: box.y + 20, w: W, h: 30 };
  const row2 = box.y + H * 0.15;
  const rowH = H * 0.24;
  const planner = { x: box.x, y: row2, w: W * 0.34, h: rowH };
  const ring = { x: box.x + W * 0.37, y: row2, w: W * 0.3, h: rowH };
  const frames = { x: box.x + W * 0.7, y: row2, w: W * 0.3, h: rowH };
  const wy = box.y + H * 0.46;
  const ww = (W - 24) / 3;
  const wh = H * 0.14;
  const workers = [0, 1, 2].map((i) => ({ x: box.x + i * (ww + 12), y: wy, w: ww, h: wh }));
  const ssd = { x: box.x, y: box.y + H * 0.68, w: W * 0.62, h: 44 };
  const outY = box.y + H * 0.84;
  const outH = Math.min(46, H * 0.08);
  const model = { x: box.x + W - (wide ? 100 : 64), y: outY, w: wide ? 100 : 64, h: outH };
  const slots = 6;
  const slotW = (model.x - box.x - 16) / slots;
  const me = workers[1];

  // Access sequence tape; our access is the highlighted one at the frontier.
  text(ctx, "ACCESS SEQUENCE · shuffled, known ahead", tape.x, tape.y - 6, { size: 11, family: "mono", color: c.text2 });
  const cw = 46;
  const n = Math.floor(W / cw);
  for (let i = 0; i < n; i++) {
    const x = tape.x + i * cw;
    const ours = i === 3;
    fillRound(ctx, x + 2, tape.y, cw - 4, tape.h, 4, ours ? alpha(c.accent, 0.25 + 0.75 * focus.seq) : alpha(c.surface, 1));
    text(ctx, String(Math.floor(hash(i, 4) * 9e6 + 1e6)).slice(0, 5), x + cw / 2, tape.y + 19, { size: 9, family: "mono", color: ours ? c.text : c.text3, align: "center" });
  }
  fillRound(ctx, tape.x + 4 * cw, tape.y - 3, cw * Math.min(8, n - 4), tape.h + 6, 6, alpha(c.teal, 0.08 + 0.1 * focus.plan));
  text(ctx, "planner look-ahead →", tape.x + 4 * cw + 6, tape.y + tape.h + 14, { size: 10, family: "mono", color: c.teal, a: 0.4 + 0.6 * focus.plan });
  text(ctx, "↑ cell 5,681,078", tape.x + 3 * cw + 2, tape.y + tape.h + 14, { size: 10, family: "mono", weight: 700, color: c.accent, a: 1 - ph("plan") });

  // Modules.
  module(ctx, c, planner, "PLANNER", "finds reuse within the budget", focus.plan, c.teal);
  module(ctx, c, ring, "PLAN RING", "entries ahead of workers", focus.plan, c.teal);
  module(ctx, c, frames, "CACHE FRAMES", "planned pages · decoded rows", focus.frames, c.teal);
  workers.forEach((r, i) => module(ctx, c, r, `WORKER ${i}`, "", i === 1 ? focus.worker : 0.15, c.accent));
  module(ctx, c, ssd, "NVMe · io_uring", "", focus.io, c.accent);
  const inflight = 9 + Math.round(4 * Math.sin(t * 1.3));
  for (let k = 0; k < 16; k++) fillRound(ctx, ssd.x + 170 + k * 12, ssd.y + 16, 9, 14, 2, alpha(c.accent, k < inflight ? 0.7 : 0.15));
  text(ctx, `${inflight} reads in flight`, ssd.x + 170 + 16 * 12 + 8, ssd.y + 27, { size: 10, family: "mono", color: c.accent });
  text(ctx, "OUTPUT RING · requested order", box.x, outY - 8, { size: 11, family: "mono", color: c.text2, a: 0.4 + 0.6 * focus.out });
  fillRound(ctx, model.x, model.y, model.w, model.h, 8, c.text);
  text(ctx, "MODEL", model.x + model.w / 2, model.y + model.h / 2 + 4, { size: 11, family: "mono", weight: 700, color: c.bg, align: "center" });

  // Frames grid: frame 17 is reserved for our page.
  const fc = 8;
  const fr = 4;
  const fw = (frames.w - 24) / fc;
  const fh = (frames.h - 52) / fr;
  const frame = (i) => ({ x: frames.x + 12 + (i % fc) * fw, y: frames.y + 44 + Math.floor(i / fc) * fh });
  const OUR = 17;
  for (let i = 0; i < fc * fr; i++) {
    const { x, y } = frame(i);
    const used = hash(i, 8) < 0.55;
    let color = used ? alpha(c.teal, 0.45) : alpha(c.divider, 0.8);
    if (i === OUR) color = ph("decode") >= 1 ? c.teal : alpha(c.accent, 0.25 + 0.5 * ph("plan"));
    fillRound(ctx, x + 1, y + 1, fw - 3, fh - 3, 2, color);
  }
  if (ph("plan") > 0) strokeRound(ctx, frame(OUR).x, frame(OUR).y, fw - 1, fh - 1, 2, alpha(c.accent, ph("plan")), 1.5);
  text(ctx, "f17", frame(OUR).x + fw / 2 - 1, frame(OUR).y + fh / 2 + 3, { size: 9, family: "mono", weight: 700, color: c.bg, align: "center", a: ph("plan") });

  // Planner: decisions for upcoming positions, scrolling as it looks ahead.
  const DECISIONS = [
    ["reused in 37 → keep f9", c.teal],
    ["no reuse → plain read", c.text3],
    ["row reused → keep row f4", c.teal],
    ["already in f12 → hit", c.teal],
    ["no reuse → plain read", c.text3],
    ["reused in 90 → keep f21", c.teal],
  ];
  const step = Math.floor(t * 0.8);
  const lines = Math.max(3, Math.floor((planner.h - 50) / 16));
  for (let k = 0; k < lines; k++) {
    const ours = focus.plan > 0.3 && k === lines - 1;
    const [msg, col] = ours ? ["reused in 412 → admit f17", c.accent] : DECISIONS[(step + k) % DECISIONS.length];
    const pos = 25178039 + step + k;
    ctx.save();
    ctx.beginPath();
    ctx.rect(planner.x + 6, planner.y, planner.w - 12, planner.h);
    ctx.clip();
    text(ctx, `${ours ? "▶" : " "} ${String(pos).slice(-4)} ${msg}`, planner.x + 10, planner.y + 52 + k * 16, { size: wide ? 10 : 8, family: "mono", weight: ours ? 700 : 400, color: col, a: 0.35 + 0.65 * Math.max(focus.plan, 0.4) });
    ctx.restore();
  }
  // Plan-ring entries, published ahead of the workers.
  const ENTRIES = ["hit f9", "read", "row f4", "admit f17", "hit f12", "read"];
  const ew = (ring.w - 24) / 3;
  ENTRIES.forEach((e, k) => {
    const ours = e === "admit f17";
    const x = ring.x + 12 + (k % 3) * ew;
    const y = ring.y + 48 + Math.floor(k / 3) * 30;
    const col = ours ? c.accent : e.startsWith("read") ? c.text3 : c.teal;
    fillRound(ctx, x, y, ew - 6, 24, 4, alpha(col, ours ? 0.25 + 0.6 * focus.plan : 0.25));
    text(ctx, e, x + (ew - 6) / 2, y + 16, { size: wide ? 10 : 8, family: "mono", weight: 600, color: ours ? c.text : c.text2, align: "center", maxWidth: ew - 8 });
  });

  // Workers' current stage pills (others cycle on their own).
  workers.forEach((r, i) => {
    let label;
    let color = c.text3;
    if (i !== 1) {
      label = ["claim", "decode", "map genes", "process fn", "wait CQ"][Math.floor(t * 0.8 + i * 1.7) % 5];
    } else if (p < PH.claim[0]) {
      label = "idle";
    } else if (p < PH.submit[0]) {
      label = "claim · plan says: admit";
      color = c.accent;
    } else if (p < PH.decode[0]) {
      label = "read in flight · next cell";
      color = c.teal;
    } else if (p < PH.out[0]) {
      const k = Math.min(STAGES.length - 1, Math.floor(seg(p, ...PH.decode) * STAGES.length));
      label = STAGES[k];
      color = c.teal;
    } else label = "write slot 3";
    const parked = 2 + Math.floor(2.5 + 2.5 * Math.sin(t * 0.9 + i * 2));
    text(ctx, `${parked} coroutines parked on I/O`, r.x + r.w - 12, r.y + 36, { size: wide ? 10 : 8, family: "mono", color: c.text3, align: "right" });
    fillRound(ctx, r.x + 12, r.y + r.h - 34, r.w - 24, 24, 12, alpha(color, i === 1 ? 0.2 + 0.6 * focus.worker : 0.12));
    // Shrink the label until it fits inside the pill.
    let size = wide ? 11 : 9;
    ctx.font = `600 ${size}px 'IBM Plex Mono', monospace`;
    while (size > 7 && ctx.measureText(label).width > r.w - 40) {
      size -= 0.5;
      ctx.font = `600 ${size}px 'IBM Plex Mono', monospace`;
    }
    text(ctx, label, r.x + r.w / 2, r.y + r.h - 18, { size, family: "mono", weight: 600, color: i === 1 ? c.text : c.text3, align: "center" });
  });

  // Ambient traffic, faint.
  for (let k = 0; k < 10; k++) {
    const u = (t * (0.2 + hash(k, 2) * 0.2) + hash(k, 5)) % 1;
    const w = workers[k % 3];
    const pts = k % 2 ? [[ring.x + ring.w / 2, ring.y + ring.h], [w.x + w.w / 2, w.y]] : [[w.x + w.w * 0.3, w.y + w.h], [w.x + w.w * 0.3, ssd.y]];
    fillRound(ctx, lerp(pts[0][0], pts[1][0], u) - 3, lerp(pts[0][1], pts[1][1], u) - 3, 6, 6, 2, alpha(k % 2 ? c.teal : c.accent, 0.25));
  }

  // Output ring as a conveyor: slot 0 is next to the model. Each cell of a
  // slot is written by whichever worker owns that position.
  const out0 = ph("out");
  const shift = ease.inOut(seg(out0, 0.55, 1)) * 4;
  const WCOL = [mix(c.teal, c.text3, 0.45), c.accent, mix(c.teal, "#4f97c2", 0.6)];
  const cellW = (slotW - 18) / 4;
  const slotX = (sIdx) => model.x - 14 - (sIdx + 1) * slotW + shift * slotW;
  const ourCell = [3, 1];
  for (let sIdx = 0; sIdx < slots + 4; sIdx++) {
    const x = slotX(sIdx);
    if (x + slotW < box.x) continue;
    const into = clamp((x + slotW - model.x) / slotW);
    const edge = clamp((x - box.x + slotW * 0.6) / (slotW * 0.6));
    const a = (1 - into) * edge;
    if (a <= 0) continue;
    strokeRound(ctx, x, outY, slotW - 6, outH, 6, alpha(c.border, a), 1);
    for (let k = 0; k < 4; k++) {
      const ours = sIdx === ourCell[0] && k === ourCell[1];
      const w = (sIdx * 4 + k) % 3;
      let filled = sIdx < 3 || (sIdx === 3 && !ours) || hash(sIdx, k) < 0.25 + 0.6 * p;
      if (ours) filled = p > PH.out[0] + 0.035;
      fillRound(ctx, x + 6 + k * cellW, outY + 8, cellW - 3, outH - 16, 3, alpha(filled ? WCOL[w] : c.divider, a * (filled ? 0.9 : 0.6)));
    }
    if (sIdx === 3) {
      strokeRound(ctx, x - 2, outY - 2, slotW - 2, outH + 4, 7, alpha(c.accent, a * (0.4 + 0.6 * focus.out)), 2);
      text(ctx, "slot 3", x, outY + outH + 14, { size: 10, family: "mono", color: c.accent, a: a * (0.4 + 0.6 * focus.out) });
    }
  }
  text(ctx, "■ worker 0  ■ worker 1  ■ worker 2", model.x - 14, outY - 8, { size: 10, family: "mono", color: c.text3, align: "right" });

  // The highlighted access, moving through the modules.
  const pathPts = [
    [tape.x + 3.5 * cw, tape.y + tape.h + 26],
    [planner.x + planner.w / 2, planner.y + planner.h / 2],
    [ring.x + ring.w / 2, ring.y + ring.h / 2],
    [me.x + me.w / 2, me.y + me.h / 2],
  ];
  const plan = ph("plan");
  const claim = ph("claim");
  let pos = pathPts[0];
  let label = "cell 5,681,078";
  if (plan > 0) {
    const u = plan * 2;
    pos = u < 1 ? [lerp(pathPts[0][0], pathPts[1][0], u), lerp(pathPts[0][1], pathPts[1][1], u)] : [lerp(pathPts[1][0], pathPts[2][0], u - 1), lerp(pathPts[1][1], pathPts[2][1], u - 1)];
    if (u > 1) label = "miss · admit → f17";
  }
  if (claim > 0) {
    pos = [lerp(pathPts[2][0], pathPts[3][0], claim), lerp(pathPts[2][1], pathPts[3][1], claim)];
    label = "miss · admit → f17";
  }
  const submit = ph("submit");
  const read = ph("read");
  const decode = ph("decode");
  const out = out0;
  const dim = p > PH.submit[0] && p < PH.decode[0] ? 0.4 : 1;
  if (out <= 0 && plan > 0) card(ctx, c, pos[0], pos[1] - (claim > 0 ? me.h * 0.12 : 0), 150, 22, label, c.accent, dim * clamp(plan * 6));
  // The read request: SQ → SSD → CQ → back to the worker, and into frame 17.
  if (submit > 0 && decode < 1) {
    const sqx = me.x + me.w * 0.3;
    const cqx = me.x + me.w * 0.7;
    let rx;
    let ry;
    if (read <= 0) {
      rx = sqx;
      ry = lerp(me.y + me.h, ssd.y + ssd.h / 2, submit);
    } else {
      rx = lerp(sqx, cqx, read);
      ry = ssd.y + ssd.h / 2;
    }
    if (decode > 0) {
      rx = lerp(cqx, frame(OUR).x + fw / 2, decode);
      ry = lerp(ssd.y + ssd.h / 2, frame(OUR).y + fh / 2, decode);
    }
    // The page itself, frame-shaped: red while it is being read, green
    // once it lands in frame 17 and replaces the reservation.
    const landed = decode > 0;
    const pw = landed ? lerp(36, fw - 3, decode) : 36;
    const phh = landed ? lerp(26, fh - 3, decode) : 26;
    fillRound(ctx, rx - pw / 2, ry - phh / 2, pw, phh, 3, landed ? c.teal : c.accent);
    text(ctx, "f17", rx, ry + 4, { size: 10, family: "mono", weight: 700, color: c.bg, align: "center" });
    text(ctx, "SQ", sqx, me.y + me.h + 16, { size: 10, family: "mono", color: c.accent, align: "center", a: focus.io });
    text(ctx, "CQ", cqx, me.y + me.h + 16, { size: 10, family: "mono", color: c.accent, align: "center", a: focus.io });
    if (read > 0 && decode <= 0) glow(ctx, rx, ry, 50, c.accent, 0.3);
  }
  // The decoded row lands in one cell of slot 3; complete slots then turn
  // into the model.
  if (out > 0) {
    const tx = slotX(3) + 6 + ourCell[1] * cellW;
    const u = ease.inOut(clamp(out * 2.2));
    const w = lerp(110, cellW - 3, u);
    const h = lerp(22, outH - 16, u);
    const x = lerp(me.x + me.w / 2 - 55, tx, u);
    const y = lerp(me.y + me.h / 2 - 11, outY + 8, u);
    // Fade out with the slot as it turns into the model.
    const gone = clamp((tx + cellW - model.x) / cellW);
    fillRound(ctx, x, y, w, h, 4, alpha(c.accent, 1 - gone));
    if (u < 0.6) text(ctx, "decoded row", x + w / 2, y + h / 2 + 4, { size: 11, family: "mono", weight: 700, color: c.bg, align: "center", a: 1 - u / 0.6 });
    const v = seg(out, 0.55, 1);
    if (v > 0) text(ctx, "slots 0–3 complete, in order → leased by the model, no copy", model.x - 14, outY + outH + 32, { size: 10, family: "mono", color: c.teal, align: "right", a: v });
  }
  // The model sits on top of the ring: slots slide into it.
  fillRound(ctx, model.x, model.y, model.w, model.h, 8, c.text);
  text(ctx, "MODEL", model.x + model.w / 2, model.y + model.h / 2 + 4, { size: 11, family: "mono", weight: 700, color: c.bg, align: "center" });
}

function compact(ctx, { p, t, box: b, c }) {
  const gap = 12;
  const col = (b.w - gap) / 2;
  const phase = (name) => ease.inOut(seg(p, ...PH[name]));
  const panel = (x, y, w, h, title, detail, color, active) => {
    fillRound(ctx, x, y, w, h, 8, alpha(color, active ? 0.13 : 0.04));
    strokeRound(ctx, x, y, w, h, 8, alpha(color, active ? 0.9 : 0.35), active ? 2 : 1);
    text(ctx, title, x + 10, y + 18, { size: 11, family: "mono", weight: 700, color });
    if (detail) text(ctx, detail, x + 10, y + 35, { size: 10, family: "mono", color: c.text2, maxWidth: w - 20 });
  };
  text(ctx, "ACCESS SEQUENCE", b.x, b.y + 10, { size: 10, family: "mono", color: c.text2 });
  const cw = Math.min(46, b.w / 7);
  for (let k = 0; k < 7; k++) {
    fillRound(ctx, b.x + k * cw + 1, b.y + 18, cw - 4, 23, 3, alpha(k === 3 ? c.accent : c.text3, k === 3 ? 0.8 : 0.15));
    text(ctx, k === 3 ? "cell" : String(8040 + k), b.x + (k + 0.5) * cw, b.y + 33, { size: 9, family: "mono", color: k === 3 ? c.bg : c.text2, align: "center" });
  }
  const py = b.y + b.h * 0.14, ph = Math.min(76, b.h * 0.18);
  panel(b.x, py, col, ph, "PLANNER", "reuse → admit f17", c.teal, p >= 0.18 && p < 0.36);
  panel(b.x + col + gap, py, col, ph, "PLAN RING", "admit f17 · use · read", c.teal, p >= 0.18 && p < 0.46);
  const my = b.y + b.h * 0.38, mh = Math.min(62, b.h * 0.14);
  panel(b.x, my, col, mh, "CACHE FRAMES", phase("decode") >= 1 ? "f17 · ready" : "f17 · reserved", c.teal, p >= 0.66 && p < 0.82);
  panel(b.x + col + gap, my, col, mh, "NVMe / io_uring", `${9 + Math.round(Math.sin(t) * 3)} reads in flight`, c.accent, p >= 0.52 && p < 0.68);
  const wy = b.y + b.h * 0.6, wh = Math.min(76, b.h * 0.17), ww = (b.w - gap * 2) / 3;
  const labels = p < 0.36 ? "claim" : p < 0.52 ? "plan entry" : p < 0.68 ? "submit / CQ" : p < 0.82 ? "decode / map" : "write";
  for (let w = 0; w < 3; w++) panel(b.x + w * (ww + gap), wy, ww, wh, `WORKER ${w}`, w === 1 ? labels : "other cells", c.accent, w === 1 && p >= 0.36);
  // The same access advances through the planner, worker, read and output stages.
  const points = [[b.x + 3.5 * cw, b.y + 48], [b.x + col / 2, py + ph], [b.x + b.w / 2, wy - 8], [b.x + col + gap + col / 2, my + mh], [b.x + b.w / 2, wy + wh], [b.x + b.w * 0.7, b.y + b.h * 0.86]];
  const u = p < 0.18 ? 0 : p < 0.36 ? phase("plan") : p < 0.52 ? 1 + phase("claim") : p < 0.68 ? 2 + phase("read") : p < 0.82 ? 3 + phase("decode") : 4 + phase("out");
  const k = Math.min(4, Math.floor(u)), f = u - k;
  const x = lerp(points[k][0], points[k + 1][0], f), y = lerp(points[k][1], points[k + 1][1], f);
  fillRound(ctx, x - 5, y - 4, 10, 8, 2, p < 0.68 ? c.accent : c.teal);
  const oy = b.y + b.h * 0.86, oh = Math.min(34, b.h * 0.08), mw = 66;
  text(ctx, "OUTPUT · requested order", b.x, oy - 8, { size: 10, family: "mono", color: c.text2 });
  const sw = (b.w - mw - 10) / 6;
  for (let i = 0; i < 6; i++) {
    strokeRound(ctx, b.x + i * sw, oy, sw - 5, oh, 4, c.border);
    fillRound(ctx, b.x + i * sw + 4, oy + 5, sw - 13, oh - 10, 2, alpha(i === 3 ? c.accent : c.teal, p > 0.82 || i < 3 ? 0.75 : 0.15));
  }
  fillRound(ctx, b.x + b.w - mw, oy, mw, oh, 6, c.text);
  text(ctx, "MODEL", b.x + b.w - mw / 2, oy + oh / 2 + 4, { size: 10, family: "mono", color: c.bg, align: "center" });
  text(ctx, "plan → read → decode → ordered output", b.x, b.y + b.h - 2, { size: 10, family: "mono", color: c.teal, maxWidth: b.w });
}

export default { draw };
