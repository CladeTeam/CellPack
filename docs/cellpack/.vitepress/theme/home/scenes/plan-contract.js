import data from "../scenes.json";
import { alpha, clamp, compact, ease, fillRound, glow, hash, lerp, line, mix, seg, strokeRound, text } from "../engine.js";
import { bracket, fit, hatch, lock } from "./plan-draw.js";

// Why the loader caches by plan. A runtime cache puts a lock on every access:
// workers queue for it, the LRU list is reordered for accesses that miss
// anyway, and the SSD idles. A plan moves every decision ahead, so workers run
// without asking anyone; the contract for one frame shows what they rely on.
const BUDGET = data.cache.budgets.indexOf(0.1);
const LRU = data.cache.methods.find((m) => m.key === "lru");
const PH = { numbers: [0.18, 0.28], plan: [0.4, 0.5], contract: [0.58, 0.66] };
const WORKERS = 4;

// Event simulation of WORKERS workers. With a lock, each access waits for it
// (FIFO), holds it to look up and reorder, then decodes. Without one, each
// access reads its plan entry and decodes.
function simulate(locked, horizon = 400) {
  const segs = [...Array(WORKERS)].map(() => []);
  const holds = [];
  const next = [...Array(WORKERS)].map((_, w) => w * 0.23);
  let lockFree = 0;
  let k = 0;
  for (;;) {
    let w = 0;
    for (let i = 1; i < WORKERS; i++) if (next[i] < next[w]) w = i;
    const req = next[w];
    if (req > horizon) break;
    const work = 0.55 + 0.35 * hash(k, 3);
    if (locked) {
      const start = Math.max(req, lockFree);
      const hold = 0.5 + 0.2 * hash(k, 5);
      if (start > req) segs[w].push([req, start, "wait"]);
      segs[w].push([start, start + hold, "lock"]);
      holds.push({ start, end: start + hold, worker: w, k });
      lockFree = start + hold;
      segs[w].push([start + hold, start + hold + work, "work"]);
      next[w] = start + hold + work;
    } else {
      segs[w].push([req, req + 0.06, "entry"]);
      segs[w].push([req + 0.06, req + 0.06 + work, "work"]);
      next[w] = req + 0.06 + work;
    }
    k++;
  }
  const done = segs.flat().filter((s) => s[2] === "work").length;
  return { segs, holds, rate: done / horizon };
}
const RUNTIME = simulate(true);
const PLANNED = simulate(false);

// The LRU list after each locked access: random order rarely finds its page,
// so almost every access inserts at the front and evicts from the back.
const LIST_LEN = 10;
const LISTS = (() => {
  const lists = [[]];
  let fresh = 0;
  for (const h of RUNTIME.holds) {
    const prev = lists[lists.length - 1];
    const hit = prev.length === LIST_LEN && hash(h.k, 9) < LRU.miss[BUDGET];
    const page = hit ? prev[Math.floor(hash(h.k, 10) * LIST_LEN)] : `p${(fresh++ * 37) % 997}`;
    lists.push([page, ...prev.filter((x) => x !== page)].slice(0, LIST_LEN));
    h.hit = hit;
  }
  return lists;
})();

const STATE_COLOR = (c) => ({ wait: c.danger, lock: c.accent, work: c.teal, entry: c.teal });
const STATE_LABEL = { wait: "blocked on the lock", lock: "holds the lock", work: "decode · map", entry: "read entry" };

function at(sim, w, now) {
  return sim.segs[w].find((s) => s[0] <= now && now < s[1]) || [now, now, "work"];
}

function system(ctx, { p, t, box, c, wide }, a) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  const W = box.w;
  const H = box.h;
  const mode = ease.inOut(seg(p, ...PH.plan)); // 0 runtime cache, 1 plan
  const runtime = 1 - mode;
  const fs = wide ? 11 : 9;
  const sim = mode < 0.5 ? RUNTIME : PLANNED;
  const now = 30 + ((t * 0.9) % 300);
  const col = STATE_COLOR(c);

  // Access sequence: the runtime cache sees only the past; the planner sees ahead.
  const cw = clamp(W / 15, 26, 46);
  const n = Math.floor(W / cw);
  const nowCell = 4;
  const tape = { x: box.x, y: box.y + 20, h: 26 };
  text(ctx, "ACCESS SEQUENCE", tape.x, tape.y - 7, { size: fs, family: "mono", color: c.text2 });
  const scroll = (t * 1.2) % 1;
  for (let i = -1; i < n; i++) {
    const x = tape.x + (i - scroll) * cw;
    if (x < box.x - 1 || x + cw > box.x + W + 1) continue;
    const id = Math.floor(t * 1.2) + i;
    const past = i < nowCell;
    fillRound(ctx, x + 2, tape.y, cw - 4, tape.h, 4, alpha(past ? c.divider : c.surface, 1));
    text(ctx, (0x100 + Math.floor(hash(id, 91) * 0xeff)).toString(16).toUpperCase(), x + cw / 2, tape.y + 17, { size: wide ? 9 : 8, family: "mono", color: past ? c.text3 : c.text2, align: "center" });
  }
  const nowX = tape.x + (nowCell + 1 - scroll) * cw;
  line(ctx, nowX, tape.y - 4, nowX, tape.y + tape.h + 4, c.accent, 2);
  fit(ctx, "history: all a runtime cache knows", box.x, tape.y + tape.h + 14, nowX - box.x - 6, { size: fs - 1, family: "mono", color: c.accent, a: runtime });
  fillRound(ctx, nowX + 4, tape.y - 4, box.x + W - nowX - 4, tape.h + 8, 6, alpha(c.teal, 0.12 * mode));
  text(ctx, wide ? "known future: the planner reads ahead" : "planner look-ahead", nowX + 8, tape.y + tape.h + 14, { size: fs - 1, family: "mono", color: c.teal, a: mode });

  // Layout: cache and card, the lock gate, workers, NVMe.
  const top = tape.y + tape.h + 30;
  const ch = wide ? clamp(H * 0.24, 120, 170) : clamp(H * 0.2, 84, 150);
  const cache = { x: box.x, y: top, w: wide ? W * 0.58 : W, h: ch };
  const card = { x: box.x + W * 0.62, y: top, w: W * 0.38, h: ch };
  const gateY = top + ch;
  const wy = gateY + clamp(H * 0.12, 64, 92);
  const wh = wide ? clamp(H * 0.17, 90, 128) : clamp(H * 0.15, 66, 96);
  const gap = 8;
  const cols = wide ? 4 : 2;
  const ww = (W - gap * (cols - 1)) / cols;
  const rows = Math.ceil(WORKERS / cols);
  const workers = [...Array(WORKERS)].map((_, i) => ({ x: box.x + (i % cols) * (ww + gap), y: wy + Math.floor(i / cols) * (wh + gap), w: ww, h: wh }));
  const ssdY = wy + rows * (wh + gap) + clamp(H * 0.03, 14, 30);
  const ssd = { x: box.x, y: ssdY, w: W, h: clamp(H * 0.1, 54, 72) };

  // The cache box. Runtime: an LRU list reordered under the lock. Plan: frames.
  const tint = mix(c.accent, c.teal, mode);
  fillRound(ctx, cache.x, cache.y, cache.w, cache.h, 10, alpha(tint, 0.06));
  strokeRound(ctx, cache.x, cache.y, cache.w, cache.h, 10, alpha(tint, 0.8), 1.2);
  text(ctx, mode < 0.5 ? "SHARED CACHE · LRU" : "CACHE FRAMES", cache.x + 12, cache.y + 20, { size: fs + 1, family: "mono", weight: 700, color: tint });
  fit(ctx, mode < 0.5 ? "one lock: look up, move to front, evict the tail" : "which page lives in which frame is in the plan", cache.x + 12, cache.y + 36, cache.w - 24, { size: fs - 1, family: "mono", color: c.text3 });
  const slotW = (cache.w - 24) / LIST_LEN;
  const listY = cache.y + 52;
  const slotH = clamp(Math.min(slotW * 0.8, cache.h - 80), 14, 40);
  const holdIdx = RUNTIME.holds.findIndex((h) => h.start <= now && now < h.end);
  const hold = RUNTIME.holds[holdIdx];
  if (runtime > 0 && hold) {
    const prev = LISTS[holdIdx];
    const cur = LISTS[holdIdx + 1];
    const f = ease.inOut(clamp((now - hold.start) / (hold.end - hold.start)));
    ctx.globalAlpha = a * runtime;
    cur.forEach((page, i) => {
      const old = prev.indexOf(page);
      const x0 = cache.x + 12 + (old < 0 ? 0 : old) * slotW;
      const x1 = cache.x + 12 + i * slotW;
      const y = old < 0 ? lerp(listY - 14, listY, f) : listY;
      const touched = i === 0;
      fillRound(ctx, lerp(x0, x1, f) + 1, y, slotW - 4, slotH, 3, touched ? alpha(hold.hit ? c.teal : c.accent, 0.8) : alpha(c.text3, 0.3));
      if (wide) text(ctx, page, lerp(x0, x1, f) + slotW / 2 - 1, y + slotH / 2 + 3, { size: 8, family: "mono", color: touched ? c.bg : c.text2, align: "center" });
    });
    const evicted = prev.length === LIST_LEN && !cur.includes(prev[LIST_LEN - 1]) ? prev[LIST_LEN - 1] : null;
    if (evicted) {
      const x = cache.x + 12 + (LIST_LEN - 1) * slotW + f * slotW * 0.6;
      fillRound(ctx, x + 1, listY + f * 18, slotW - 4, slotH, 3, alpha(c.text3, 0.3 * (1 - f)));
    }
    ctx.globalAlpha = a;
    const verdict = hold.hit ? "hit: page found" : "miss: insert at front, evict the tail, read from SSD";
    fit(ctx, `W${hold.worker} · ${verdict}`, cache.x + 12, listY + slotH + 20, cache.w - 24, { size: fs, family: "mono", weight: 600, color: hold.hit ? c.teal : c.accent, a: runtime });
  }
  if (mode > 0) {
    const fc = wide ? 10 : 7;
    const fw = (cache.w - 24) / fc;
    for (let k = 0; k < fc * 2; k++) {
      const x = cache.x + 12 + (k % fc) * fw;
      const y = listY + Math.floor(k / fc) * (slotH * 0.6 + 6);
      fillRound(ctx, x + 1, y, fw - 4, slotH * 0.6, 3, alpha(c.teal, (hash(k, 5) < 0.6 ? 0.55 : 0.12) * mode));
    }
  }

  // The lock: one gate in front of the cache. The holder is inside; the others
  // queue in arrival order, well clear of the gate.
  const gx = cache.x + cache.w / 2;
  if (runtime > 0) {
    const waiting = [...Array(WORKERS).keys()]
      .map((w) => ({ w, s: at(RUNTIME, w, now) }))
      .filter((x) => x.s[2] === "wait")
      .sort((x, y) => x.s[0] - y.s[0]);
    const contended = waiting.length > 0;
    line(ctx, cache.x + 12, gateY, cache.x + cache.w - 12, gateY, alpha(c.danger, 0.5 * runtime), 3);
    glow(ctx, gx, gateY + 4, 46, c.danger, (contended ? 0.35 : 0.12) * runtime * (0.8 + 0.2 * Math.sin(t * 6)));
    lock(ctx, gx, gateY + 2, 30, contended ? c.danger : c.accent, runtime);
    if (hold) text(ctx, `held by W${hold.worker}`, gx + 26, gateY + 8, { size: fs, family: "mono", weight: 700, color: c.accent, a: runtime });
    const bw = wide ? 34 : 28;
    waiting.forEach((x, i) => {
      const bx = gx - 44 - (i + 1) * (bw + 6);
      fillRound(ctx, bx, gateY + 18, bw, 20, 5, alpha(c.danger, 0.85 * runtime));
      text(ctx, `W${x.w}`, bx + bw / 2, gateY + 32, { size: fs, family: "mono", weight: 700, color: c.bg, align: "center", a: runtime });
    });
    if (waiting.length) {
      line(ctx, gx - 44 - 4, gateY + 28, gx - 22, gateY + 28, alpha(c.danger, runtime), 1.5, [3, 3]);
      text(ctx, `${waiting.length} blocked`, wide ? gx - 44 - waiting.length * (bw + 6) - 8 : box.x + 12, gateY + (wide ? 32 : 56), { size: fs, family: "mono", weight: 700, color: c.danger, align: wide ? "right" : "left", a: runtime });
    }
  }

  // Right card: LRU against no cache (paper), then the planner.
  const nums = wide ? ease.out(seg(p, ...PH.numbers)) * runtime : 0;
  if (nums > 0) {
    const x = card.x + 12;
    const bw = card.w - 24;
    fit(ctx, "LRU · 10% BUDGET · REPORT FIG. 6", x, card.y + 14, bw, { size: fs - 1, family: "mono", color: c.text2, a: nums });
    const big = clamp(card.w / 7, 18, 30);
    text(ctx, `${(LRU.miss[BUDGET] * 100).toFixed(1)}%`, x, card.y + 18 + big, { size: big, family: "head", weight: 600, color: c.accent, a: nums });
    fit(ctx, "of page misses removed", x + big * 2.6, card.y + 14 + big, bw - big * 2.6, { size: fs - 1, family: "mono", color: c.text2, a: nums });
    const bars = [
      ["cells/s", data.cache.no_cache, LRU.cells_per_s[BUDGET], compact],
      ["SSD reads", data.cache.no_cache_read_GBps, LRU.read_GBps[BUDGET], (v) => `${v.toFixed(1)} GB/s`],
    ];
    const bh = 9;
    bars.forEach(([label, none, lru, fmt], k) => {
      const y = card.y + 30 + big + k * 46;
      text(ctx, label, x, y + 4, { size: fs - 1, family: "mono", weight: 700, color: c.text2, a: nums });
      fillRound(ctx, x, y + 9, bw * 0.75, bh, 3, alpha(c.text3, 0.6 * nums));
      fit(ctx, `no cache ${fmt(none)}`, x + bw * 0.75 + 4, y + 17, bw * 0.25, { size: fs - 2, family: "mono", color: c.text3, a: nums });
      fillRound(ctx, x, y + 22, bw * 0.75 * (lru / none), bh, 3, alpha(c.danger, 0.85 * nums));
      text(ctx, `LRU ${fmt(lru)}`, x + bw * 0.75 * (lru / none) + 4, y + 30, { size: fs - 2, family: "mono", weight: 700, color: c.danger, a: nums });
    });
  }
  if (mode > 0 && wide) {
    fillRound(ctx, card.x, card.y, card.w, card.h, 10, alpha(c.teal, 0.06 * mode));
    strokeRound(ctx, card.x, card.y, card.w, card.h, 10, alpha(c.teal, 0.8 * mode), 1.2);
    text(ctx, "PLANNER", card.x + 12, card.y + 20, { size: fs + 1, family: "mono", weight: 700, color: c.teal, a: mode });
    fit(ctx, "own cores · off the hot path", card.x + 12, card.y + 36, card.w - 24, { size: fs - 1, family: "mono", color: c.text3, a: mode });
    const KINDS = [["admit f7", c.accent], ["use f3", c.teal], ["read", c.text3], ["use f7", c.teal], ["admit f2", c.accent], ["use f2", c.teal]];
    const step = Math.floor(t * 1.4);
    const lines = Math.max(2, Math.floor((card.h - 52) / 15));
    ctx.save();
    ctx.beginPath();
    ctx.rect(card.x + 4, card.y + 42, card.w - 8, card.h - 46);
    ctx.clip();
    for (let k = 0; k < lines; k++) {
      const [kind, kc] = KINDS[(step + k) % KINDS.length];
      fit(ctx, `pos ${4140 + step + k}  →  ${kind}`, card.x + 12, card.y + 56 + k * 15, card.w - 24, { size: fs - 1, family: "mono", color: kc, a: mode * (0.5 + 0.5 * (k / lines)) });
    }
    ctx.restore();
  }
  if (mode > 0) {
    const KC = [c.accent, c.teal, c.text3];
    const ox = wide ? card.x + card.w / 2 : box.x + W * 0.7;
    const oy = wide ? card.y + card.h : tape.y + tape.h;
    for (let k = 0; k < 8; k++) {
      const u = (t * 0.55 + k / 8) % 1;
      const w = workers[k % WORKERS];
      fillRound(ctx, lerp(ox, w.x + w.w * 0.75, u) - 9, lerp(oy, w.y, u) - 4, 18, 8, 3, alpha(KC[k % 3], 0.7 * mode * Math.sin(Math.PI * u)));
    }
  }

  // Workers: current state, and the last few moments as a strip, newest right.
  const span = 7;
  workers.forEach((r, w) => {
    fillRound(ctx, r.x, r.y, r.w, r.h, 8, c.bg);
    strokeRound(ctx, r.x, r.y, r.w, r.h, 8, alpha(c.border, 1), 1);
    const s = at(sim, w, now);
    text(ctx, `WORKER ${w}`, r.x + 10, r.y + 18, { size: fs, family: "mono", weight: 700, color: c.text2 });
    const pillY = r.y + 26;
    const ph = clamp(r.h * 0.24, 20, 28);
    fillRound(ctx, r.x + 8, pillY, r.w - 16, ph, 6, alpha(col[s[2]], 0.18));
    strokeRound(ctx, r.x + 8, pillY, r.w - 16, ph, 6, alpha(col[s[2]], 0.8), 1);
    fit(ctx, STATE_LABEL[s[2]], r.x + r.w / 2, pillY + ph / 2 + 4, r.w - 28, { size: fs, family: "mono", weight: 700, color: col[s[2]], align: "center" });
    const sy = pillY + ph + 10;
    const sh = Math.max(10, r.y + r.h - sy - (wide ? 22 : 8));
    const sx = (time) => r.x + 8 + ((time - (now - span)) / span) * (r.w - 16);
    let blocked = 0;
    for (const g of sim.segs[w]) {
      if (g[1] < now - span || g[0] > now) continue;
      const x0 = Math.max(r.x + 8, sx(g[0]));
      const x1 = Math.min(r.x + r.w - 8, sx(Math.min(g[1], now)));
      fillRound(ctx, x0, sy, Math.max(0, x1 - x0 - 1), sh, 2, alpha(col[g[2]], g[2] === "wait" ? 0.85 : 0.6));
      if (g[2] === "wait") blocked += Math.min(g[1], now) - Math.max(g[0], now - span);
    }
    if (wide) fit(ctx, `${Math.round((blocked / span) * 100)}% of the last moments blocked`, r.x + 10, r.y + r.h - 7, r.w - 20, { size: fs - 2, family: "mono", color: blocked > 0 ? c.danger : c.teal });
  });
  text(ctx, "■ blocked  ■ holds lock  ■ decode", box.x + W, wy - 8, { size: fs - 1, family: "mono", color: c.text3, align: "right", a: runtime });

  // NVMe: reads in flight follow how many accesses the workers get through.
  fillRound(ctx, ssd.x, ssd.y, ssd.w, ssd.h, 8, alpha(c.accent, 0.06));
  strokeRound(ctx, ssd.x, ssd.y, ssd.w, ssd.h, 8, alpha(c.accent, 0.6), 1);
  text(ctx, "NVMe · page reads in flight", ssd.x + 12, ssd.y + (wide ? 20 : 14), { size: fs, family: "mono", weight: 700, color: c.accent });
  const slots = 24;
  const target = lerp(RUNTIME.rate, PLANNED.rate, mode) * 2.1;
  const inflight = Math.round(clamp(target + Math.sin(t * 1.7) * 0.8, 0, slots));
  const sw = (ssd.w - 24) / slots;
  for (let k = 0; k < slots; k++) fillRound(ctx, ssd.x + 12 + k * sw, ssd.y + (wide ? 30 : 24), sw - 3, wide ? ssd.h - 40 : 12, 2, alpha(c.accent, k < inflight ? 0.75 : 0.12));
  const idle = runtime > 0.5 ? "mostly idle: workers are queued at the lock" : "kept busy: no worker waits for another";
  if (wide) fit(ctx, `${inflight} / ${slots} · ${idle}`, ssd.x + ssd.w - 12, ssd.y + 20, ssd.w * 0.55, { size: fs - 1, family: "mono", weight: 600, color: runtime > 0.5 ? c.danger : c.teal, align: "right" });
  else fit(ctx, `${inflight} / ${slots} · ${idle}`, ssd.x, ssd.y + ssd.h - 4, ssd.w, { size: fs - 1, family: "mono", weight: 600, color: runtime > 0.5 ? c.danger : c.teal });
  ctx.restore();
}

// The contract for one frame: who admits which page when, which generation
// each use waits for, and when the next tenant may move in. Later accesses of
// the page are served from the frame.
const SEQ = [12, 5, 31, 7, 18, 40, 22, 7, 9, 7, 3, 15, 26, 44, 26];
const U = 3;
const USES = [7, 9];
const END = 9;
const NEXT = 12;
const D = 2;
const STEPS = [0.64, 0.72, 0.8, 0.88, 0.95];

function contract(ctx, { p, t, box, c, wide }, a) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  const W = box.w;
  const H = box.h;
  const fs = wide ? 11 : 9;
  const s = STEPS.map((x) => ease.out(seg(p, x - 0.04, x + 0.03)));
  const n = SEQ.length;
  const left = box.x + (wide ? 40 : 30);
  const cw = (box.x + W - left) / n;
  const X = (i) => left + i * cw;
  const blue = mix(c.teal, "#4f97c2", 0.6);
  const lh = clamp(H * 0.07, 30, 44);
  const content = 78 + 28 + 40 + lh + (wide ? 44 + 70 : 4 * 40 + 24) + 90;
  const top = box.y + Math.max(0, (H - content) * 0.4);
  const tapeY = top + 78;
  const th = 28;

  text(ctx, wide ? "ONE FRAME, AS THE PLAN SCHEDULES IT" : "ONE PLANNED FRAME", box.x, top + 16, { size: fs, family: "mono", weight: 700, color: c.text2 });
  text(ctx, "position →", box.x + W, tapeY - 10, { size: fs - 1, family: "mono", color: c.text3, align: "right" });
  for (let i = 0; i < n; i++) {
    const ours = SEQ[i] === 7;
    const next = SEQ[i] === 26;
    const fill = ours ? alpha(c.teal, 0.3) : next ? alpha(blue, 0.3) : alpha(c.surface, 1);
    fillRound(ctx, X(i) + 2, tapeY, cw - 4, th, 4, fill);
    text(ctx, String(SEQ[i]), X(i) + cw / 2, tapeY + 18, { size: wide ? 11 : 8, family: "mono", weight: ours || next ? 700 : 400, color: ours || next ? c.text : c.text3, align: "center" });
    text(ctx, String(i), X(i) + cw / 2, tapeY + th + 12, { size: 8, family: "mono", color: c.text3, align: "center" });
  }
  text(ctx, "page", box.x, tapeY + 18, { size: 8, family: "mono", color: c.text3 });

  // The frame's lane.
  const laneY = tapeY + th + 40;
  text(ctx, "f7", box.x, laneY + lh / 2 + 4, { size: fs + 1, family: "mono", weight: 700, color: c.teal });
  strokeRound(ctx, left, laneY, X(n) - left, lh, 6, alpha(c.border, 1), 1);
  hatch(ctx, X(U - D), laneY + 2, (X(U) - X(U - D)) * s[0], lh - 4, alpha(c.teal, 0.6));
  fillRound(ctx, X(U), laneY + 2, (X(END + 1) - X(U)) * ease.inOut(seg(p, STEPS[0], STEPS[2] + 0.04)), lh - 4, 4, alpha(c.teal, 0.25 + 0.4 * s[1]));
  if (s[0] > 0) text(ctx, "page 7", X(U) + 6, laneY + lh / 2 + 4, { size: fs, family: "mono", weight: 700, color: c.text, a: s[0] });
  hatch(ctx, X(NEXT - D), laneY + 2, (X(NEXT) - X(NEXT - D)) * s[3], lh - 4, alpha(blue, 0.7));
  fillRound(ctx, X(NEXT), laneY + 2, (X(n) - X(NEXT)) * s[3], lh - 4, 4, alpha(blue, 0.4 * s[4] + 0.2 * s[3]));
  if (s[3] > 0) text(ctx, "page 26", X(NEXT) + 6, laneY + lh / 2 + 4, { size: fs, family: "mono", weight: 700, color: c.text, a: s[3] });

  // Generation counter: the only state workers share.
  const gen = s[3] >= 1 && s[4] > 0.5 ? 2 : s[1] > 0.5 ? 1 : 0;
  const gw = wide ? 150 : 110;
  const gx = box.x + W - gw;
  fillRound(ctx, gx, top, gw, 46, 8, alpha(c.teal, 0.08));
  strokeRound(ctx, gx, top, gw, 46, 8, alpha(c.teal, 0.7), 1);
  text(ctx, "f7.generation", gx + 10, top + 16, { size: fs - 1, family: "mono", color: c.text2 });
  text(ctx, String(gen), gx + 10, top + 39, { size: 20, family: "head", weight: 600, color: c.teal });

  // Events, each under its position.
  const ev = [
    { i: U, k: 0, col: c.accent, head: "worker 2 · admit", lines: ["read page 7 into f7", "then publish gen 1"] },
    { i: USES[0], k: 2, col: c.teal, head: "worker 0 · use", lines: ["wait for gen 1,", "decode from f7"] },
    { i: USES[1], k: 2, col: c.teal, head: "worker 1 · use", lines: ["last use of page 7"] },
    { i: NEXT, k: 3, col: c.teal, head: "worker 2 · admit", lines: ["page 26 → f7", "publish gen 2"] },
  ];
  const evY = laneY + lh + (wide ? 30 : 26);
  ev.forEach((e, j) => {
    const v = s[e.k];
    if (v <= 0) return;
    const x = X(e.i) + cw / 2;
    const row = wide ? j % 2 : j;
    const y = evY + row * (wide ? 44 : 40);
    line(ctx, x, laneY + lh, x, y - 2, alpha(e.col, v), 1, [2, 3]);
    const align = e.i > n * 0.65 ? "right" : "left";
    const tx = align === "right" ? x + 4 : x - 4;
    text(ctx, e.head, tx, y + 10, { size: fs, family: "mono", weight: 700, color: e.col, align, a: v });
    e.lines.forEach((l, k) => text(ctx, l, tx, y + 24 + k * 13, { size: fs - 1, family: "mono", color: c.text2, align, a: v }));
  });
  bracket(ctx, X(END) + cw, X(NEXT), laneY - 10, c.teal, wide ? `free; next tenant reserves from ${NEXT - D}` : `reserve ${NEXT - D}`, { a: s[3], size: fs - 1, dir: -1 });
  bracket(ctx, X(U - D), X(U), laneY - 10, c.teal, "reserved from u − D", { a: s[0], size: fs - 1, dir: -1 });

  // Reuse in motion: one SSD read fills f7, and the later cells of page 7
  // take it straight from the frame.
  const live = s[2];
  if (live > 0) {
    const cyc = (t * 0.45) % 1;
    const read = seg(cyc, 0, 0.3);
    if (read > 0 && read < 1) {
      const x = X(U) + cw / 2;
      const y = lerp(tapeY + th / 2, laneY + lh / 2, ease.inOut(read));
      fillRound(ctx, x - 7, y - 5, 14, 10, 3, alpha(c.accent, live));
      text(ctx, "SSD read", x + 12, top + 58, { size: fs - 1, family: "mono", weight: 700, color: c.accent, a: live * (1 - read) });
    }
    USES.forEach((v, k) => {
      const u = seg(cyc, 0.35 + k * 0.22, 0.6 + k * 0.22);
      if (u <= 0 || u >= 1) return;
      const x0 = X(v) + cw / 2;
      fillRound(ctx, x0 - 6, lerp(laneY + lh / 2, tapeY + th / 2, ease.inOut(u)) - 5, 12, 10, 3, alpha(c.teal, live));
      glow(ctx, x0, tapeY + th / 2, 28, c.teal, 0.45 * live * Math.sin(Math.PI * u));
      text(ctx, "hit · no read", x0, tapeY - 8, { size: fs - 1, family: "mono", weight: 700, color: c.teal, align: "center", a: live * Math.sin(Math.PI * u) });
    });
    fit(ctx, `page 7: 1 read, ${USES.length} later cells served from f7`, box.x, top + 36, gx - box.x - 12, { size: fs, family: "mono", weight: 700, color: c.teal, a: live });
  }

  // The rules, spelled out.
  const ry = evY + (wide ? 44 + 70 : 4 * 40 + 24);
  const rules = [
    "Each position's entry fixes: frame · admit or use · generation.",
    "A use may run before the admitting read lands: it waits for the generation.",
    "A frame changes tenant only after D quiet positions: no reader is left.",
    "Nothing else is shared: no lock, no lookup, no reordering.",
  ];
  rules.forEach((r, k) => fit(ctx, `${k + 1}  ${r}`, box.x, ry + k * 20, W, { size: fs, family: "mono", color: k === 0 ? c.text : c.text2, weight: k === 0 ? 600 : 400, a: s[Math.min(4, k + 1)] }));
  ctx.restore();
}

function draw(ctx, env) {
  const switchTo = ease.inOut(seg(env.p, ...PH.contract));
  system(ctx, env, 1 - switchTo);
  contract(ctx, env, switchTo);
}

export default { draw };
