import { alpha, clamp, ease, fillRound, glow, hash, lerp, line, seg, strokeRound, text, paragraph } from "../engine.js";

// One worker, first person. Every position is a lightweight coroutine. A
// cache hit runs straight through; a miss submits its read to io_uring,
// parks, and the worker moves on. At fixed check points — after claiming and
// between stages — the worker polls its completion queue and resumes parked
// coroutines whose reads have landed. Illustrative schedule with typical
// latencies; not a measured trace.
// New reads stop at T_STOP; the worker then keeps claiming cached positions
// and polling until every read it parked has landed and been processed.
const T_STOP = 330;
const STAGES = ["claim", "plan", "decode", "map", "process fn", "write"];
const DUR = { claim: 1.2, plan: 0.8, map: 3, "process fn": 4, write: 1.5, submit: 1.6 };

const SIM = (() => {
  const segs = [];
  const reads = [];
  const checks = [];
  const done = [];
  const claims = [];
  const parked = [];
  const resume = [];
  let t = 0;
  let next = 0;
  const run = (pos, stage, d) => {
    segs.push({ pos, stage, t0: t, t1: t + d });
    t += d;
  };
  const check = () => {
    const took = [];
    for (let i = parked.length - 1; i >= 0; i--)
      if (parked[i].t1 <= t) {
        resume.push(parked[i].pos);
        parked[i].picked = t;
        took.push(parked[i].pos);
        parked.splice(i, 1);
      }
    checks.push({ t, took });
    t += 0.3;
  };
  const finish = (pos) => {
    for (const st of ["decode", "map", "process fn", "write"]) {
      run(pos, st, st === "decode" ? 11 + hash(pos, 3) * 9 : DUR[st]);
      if (st !== "write") check();
    }
    done.push({ pos, t });
  };
  while (t < T_STOP || parked.length || resume.length) {
    check();
    if (resume.length) {
      finish(resume.shift());
      continue;
    }
    const pos = next++;
    claims.push({ pos, t });
    run(pos, "claim", DUR.claim);
    run(pos, "plan", DUR.plan);
    if (hash(pos, 17) < 0.65 || t >= T_STOP) finish(pos);
    else {
      run(pos, "submit", DUR.submit);
      const r = { pos, t0: t, t1: t + 150 + hash(pos, 5) * 60 };
      reads.push(r);
      parked.push(r);
    }
  }
  return { segs, reads, checks, done, claims, end: t };
})();
const T_END = SIM.end;
const id = (pos) => `#${300 + pos * 3}`;
// Global positions: mine are every third one; the other two workers finish
// theirs at similar times. A slot of four publishes once all are done and
// every earlier slot has published.
const MY_DONE = Object.fromEntries(SIM.done.map((d) => [d.pos, d.t]));
const DONE_AT = Array.from({ length: SIM.claims.length * 3 }, (_, g) => {
  const mine = MY_DONE[Math.floor(g / 3)];
  if (g % 3 === 0) return mine ?? Infinity;
  return (mine ?? T_END * 2) + (hash(g, 41) - 0.5) * 50;
});
const PUBLISH = (() => {
  const out = [];
  let prev = 0;
  for (let s = 0; s * 4 < DONE_AT.length; s++) {
    prev = Math.max(prev, ...DONE_AT.slice(s * 4, s * 4 + 4));
    out.push(prev);
  }
  return out;
})();
const SAY = {
  claim: "claim the next position",
  plan: "look up my plan entry",
  submit: "page not cached → submit read to io_uring, park, yield",
  decode: "decode the record — this cell only",
  map: "map genes to the model's vocabulary",
  "process fn": "run the native process function",
  write: "write to my slot in the output ring",
};

function draw(ctx, { p, box, c, wide }) {
  const now = lerp(0, T_END, seg(p, 0.04, 0.94));
  const intro = ease.out(seg(p, 0, 0.05));
  const W = box.w;
  const H = box.h;
  const big = Math.min(34, W / 17);
  const ringR = Math.min(W * 0.15, H * 0.16, W < 620 ? 54 : Infinity);
  const compact = W < 620;
  const handsH = compact ? Math.max(110, Math.min(160, H - 330)) : Math.max(H * 0.3, ringR * 2 + 30);
  if (!compact) ctx.translate(0, Math.max(0, (H - (big + 36 + handsH + 190)) / 2));

  const cur = SIM.segs.find((s) => now >= s.t0 && now < s.t1);
  const lastCheck = [...SIM.checks].reverse().find((k) => k.t <= now);
  const checking = lastCheck && now - lastCheck.t < 3;

  text(ctx, `t = ${now.toFixed(1)} µs`, box.x, box.y + big * 0.85 + (compact ? 24 : 0), { size: big, family: "head", color: c.text, a: intro });
  text(ctx, compact ? "WORKER · one coroutine / position" : "I AM A WORKER · one coroutine per position", box.x + W, box.y + 14, { size: compact ? 10 : 11, family: "mono", weight: 700, color: c.accent, align: "right", a: intro });

  const top = box.y + big + (compact ? 54 : 36);
  const qW = compact ? 64 : W * 0.17;
  const ring = { x: box.x + W - ringR - 20, y: compact ? top + handsH + ringR + 28 : top + ringR + 14 };
  const hands = { x: box.x + qW + 22, y: top, w: compact ? W - qW - 22 : ring.x - ringR - 60 - (box.x + qW + 22), h: handsH };

  // Left: resumed coroutines first, then new positions.
  text(ctx, "UP NEXT", box.x, top - 6, { size: 11, family: "mono", weight: 600, color: c.text2, a: intro });
  const started = (pos) => SIM.segs.some((s) => s.pos === pos && s.stage === "decode" && s.t0 <= now);
  const resumed = SIM.reads.filter((r) => r.picked !== undefined && r.picked <= now && !started(r.pos));
  const upcoming = SIM.claims.filter((k) => k.t > now);
  const queue = [...resumed.map((r) => [r.pos, "resume"]), ...upcoming.map((k) => [k.pos, "new"])].slice(0, 5);
  queue.forEach(([pos, kind], k) => {
    const y = top + 6 + k * (compact ? 40 : 32);
    const col = kind === "resume" ? c.teal : c.text3;
    fillRound(ctx, box.x, y, qW, compact ? 36 : 26, 6, alpha(kind === "resume" ? c.tealSoft : c.surface, intro));
    strokeRound(ctx, box.x, y, qW, compact ? 36 : 26, 6, alpha(col, 0.7 * intro), 1);
    text(ctx, id(pos), box.x + 10, y + 17, { size: 12, family: "mono", weight: 700, color: c.text, a: intro });
    text(ctx, kind === "resume" ? "resume" : "new", box.x + qW - 8, y + (compact ? 30 : 17), { size: 10, family: "mono", color: col, align: "right", a: intro });
  });

  // Centre: the running coroutine, its stages and the check points.
  fillRound(ctx, hands.x, hands.y, hands.w, hands.h, 14, alpha(c.surface, intro));
  strokeRound(ctx, hands.x, hands.y, hands.w, hands.h, 14, alpha(c.teal, intro), 2);
  text(ctx, "RUNNING", hands.x + 14, hands.y + 22, { size: 11, family: "mono", weight: 600, color: c.text2, a: intro });
  if (!cur && now >= T_END - 0.5) {
    text(ctx, "all caught up", hands.x + 14, hands.y + 24 + big, { size: big * 0.8, family: "head", weight: 600, color: c.teal, a: intro });
    paragraph(ctx, "every parked read has landed and been processed", hands.x + 14, hands.y + 48 + big, hands.w - 28, { size: 11, family: "mono", color: c.text2, a: intro });
  }
  if (cur) {
    text(ctx, id(cur.pos), hands.x + 14, hands.y + 24 + big, { size: big, family: "head", weight: 600, color: c.text, a: intro });
    const isSubmit = cur.stage === "submit";
    const si = isSubmit ? 1 : STAGES.indexOf(cur.stage);
    const sw = (hands.w - 28) / STAGES.length;
    const sy = hands.y + hands.h - 66;
    STAGES.forEach((name, k) => {
      const x = hands.x + 14 + k * sw;
      const on = k === si;
      fillRound(ctx, x, sy, sw - 10, 26, 6, on ? (isSubmit ? c.accent : c.teal) : alpha(k < si ? c.teal : c.divider, k < si ? 0.4 : 0.8));
      if (sw > 54) text(ctx, name, x + (sw - 10) / 2, sy + 17, { size: 10, family: "mono", weight: 600, color: on ? c.bg : c.text3, align: "center" });
      if (k >= 1 && k < STAGES.length - 1) {
        ctx.save();
        ctx.translate(x + sw - 5, sy + 13);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = checking ? c.accent : alpha(c.text3, 0.7);
        ctx.fillRect(-3.5, -3.5, 7, 7);
        ctx.restore();
      }
    });
    text(ctx, compact ? "◆ poll CQ" : "◆ check point: poll the completion queue", hands.x + hands.w - 14, sy - 8, { size: 10, family: "mono", color: c.text3, align: "right", a: intro });
    const f = (now - cur.t0) / (cur.t1 - cur.t0);
    fillRound(ctx, hands.x + 14, sy + 32, (hands.w - 28) * f, 3, 1, isSubmit ? c.accent : c.teal);
    paragraph(ctx, compact ? cur.stage === "submit" ? "submit → park → yield" : cur.stage : SAY[cur.stage], hands.x + 14, hands.y + hands.h - 12, hands.w - 28, { size: wide ? 12 : 10, family: "mono", weight: 600, color: isSubmit ? c.accent : c.text2, a: intro });
    if (cur.stage === "decode") {
      const n = 10;
      const vx = hands.x + 14 + big * 2.8;
      const vw = (hands.x + hands.w - 14 - vx) / n;
      for (let k = 0; k < n; k++) fillRound(ctx, vx + k * vw, hands.y + 36, vw - 4, big * 0.8, 4, alpha(c.teal, f * n > k ? 0.85 : 0.12));
    }
  }

  // Right: my io_uring. Reads travel around from SQ to CQ.
  ctx.beginPath();
  ctx.arc(ring.x, ring.y, ringR, 0, Math.PI * 2);
  ctx.strokeStyle = alpha(c.accent, 0.45 * intro);
  ctx.lineWidth = 10;
  ctx.stroke();
  ctx.lineWidth = 1;
  text(ctx, "io_uring", ring.x, ring.y - 2, { size: 12, family: "mono", weight: 700, color: c.accent, align: "center", a: intro });
  const flying = SIM.reads.filter((r) => r.t0 <= now && now < r.t1).length;
  text(ctx, `${flying} in flight`, ring.x, ring.y + 14, { size: 10, family: "mono", color: c.text3, align: "center", a: intro });
  text(ctx, "SQ", ring.x - ringR - 18, ring.y - ringR * 0.6, { size: 11, family: "mono", weight: 700, color: c.accent, align: "center", a: intro });
  text(ctx, "CQ", ring.x - ringR - 18, ring.y + ringR * 0.75, { size: 11, family: "mono", weight: 700, color: c.teal, align: "center", a: intro });
  const handPt = [hands.x + hands.w, hands.y + hands.h * 0.45];
  const cq = [ring.x - ringR * 0.72, ring.y + ringR * 0.7];
  const waitingCQ = SIM.reads.filter((q) => q.t1 <= now && (q.picked === undefined || q.picked > now));
  SIM.reads.forEach((r) => {
    if (now < r.t0 - DUR.submit || (r.picked !== undefined && now > r.picked + 6)) return;
    let x;
    let y;
    let col = c.accent;
    if (now < r.t0) {
      const u = ease.inOut(clamp((now - (r.t0 - DUR.submit)) / DUR.submit));
      x = lerp(handPt[0], ring.x - ringR * 0.75, u);
      y = lerp(handPt[1], ring.y - ringR * 0.65, u);
    } else if (now < r.t1) {
      const angle = Math.PI * 1.2 + ((now - r.t0) / (r.t1 - r.t0)) * Math.PI * 1.6;
      x = ring.x + Math.cos(angle) * ringR;
      y = ring.y + Math.sin(angle) * ringR;
    } else if (r.picked === undefined || now < r.picked) {
      const k = waitingCQ.indexOf(r);
      x = cq[0] - k * 10;
      y = cq[1] + k * 6;
      col = c.teal;
    } else {
      const u = ease.inOut(clamp((now - r.picked) / 6));
      x = lerp(cq[0], box.x + qW / 2, u);
      y = lerp(cq[1], top + 19, u);
      col = c.teal;
    }
    fillRound(ctx, x - 22, y - 10, 44, 20, 5, col);
    text(ctx, id(r.pos), x, y + 4, { size: 10, family: "mono", weight: 700, color: c.bg, align: "center" });
  });
  if (checking) {
    // A check point: the CQ lights up briefly; landed reads are taken.
    const a = 1 - (now - lastCheck.t) / 3;
    // A soft haze at the completion queue, denser than a plain glow.
    glow(ctx, cq[0], cq[1], 34, c.teal, 0.55 * a);
    glow(ctx, cq[0], cq[1], 18, c.teal, 0.6 * a);
    text(ctx, lastCheck.took.length ? "poll ✓" : "poll", compact ? ring.x : cq[0] - 34, compact ? ring.y - 28 : cq[1] + 26, { size: 10, family: "mono", weight: 700, color: c.teal, align: "center", a });
  }

  // The output ring: my cells and other workers' cells land in their slots;
  // full slots turn into the model, in requested order.
  const ringY = top + hands.h + (compact ? ringR * 2 + 54 : 40);
  const ringH = 40;
  const modelW = wide ? 84 : 60;
  fillRound(ctx, box.x, ringY, modelW, ringH, 8, c.text);
  text(ctx, "MODEL", box.x + modelW / 2, ringY + ringH / 2 + 4, { size: 11, family: "mono", weight: 700, color: c.bg, align: "center" });
  text(ctx, compact ? "OUTPUT · ■ mine  ■ others" : "OUTPUT RING · ■ mine  ■ other workers", box.x, ringY - 8, { size: 11, family: "mono", weight: 600, color: c.text2, a: intro });
  const slotW = Math.min(150, (W - modelW - 16) / 6);
  const headF = PUBLISH.reduce((sum, at) => sum + ease.inOut(clamp((now - at - 4) / 12)), 0);
  const first = Math.max(0, Math.floor(headF) - 1);
  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x + modelW + 6, ringY - 4, W - modelW - 6, ringH + 8);
  ctx.clip();
  for (let sIdx = first; sIdx < first + 9 && sIdx < PUBLISH.length; sIdx++) {
    const x = box.x + modelW + 12 + (sIdx - headF) * slotW;
    strokeRound(ctx, x, ringY, slotW - 8, ringH, 6, alpha(c.border, intro), 1);
    const cw = (slotW - 20) / 4;
    for (let k = 0; k < 4; k++) {
      const g = sIdx * 4 + k;
      const at = DONE_AT[g];
      if (!(at <= now)) continue;
      const mine = g % 3 === 0;
      const pop = ease.out(clamp((now - at) / 6));
      fillRound(ctx, x + 6 + k * cw, ringY + 6 + (ringH - 12) * (1 - pop) * 0.5, cw - 3, (ringH - 12) * pop, 3, mine ? c.teal : alpha(c.text3, 0.45));
      if (mine && cw > 26) text(ctx, id(g / 3), x + 6 + k * cw + (cw - 3) / 2, ringY + ringH / 2 + 4, { size: 9, family: "mono", weight: 700, color: c.bg, align: "center", a: pop });
    }
  }
  ctx.restore();

  // Bottom: a continuous CPU timeline — never idle.
  const ty = ringY + ringH + (compact ? 42 : 44);
  const X = (us) => box.x + (clamp(us, 0, T_END) / T_END) * W;
  text(ctx, "MY CPU TIME", box.x, ty - (compact ? 24 : 8), { size: 11, family: "mono", weight: 600, color: c.text2, a: intro });
  fillRound(ctx, box.x, ty, W, 16, 4, alpha(c.surface, intro));
  const COL = { claim: c.text3, plan: c.text3, decode: c.teal, map: c.teal, "process fn": c.teal, write: c.teal, submit: c.accent };
  SIM.segs.forEach((sg) => {
    if (sg.t0 > now) return;
    ctx.fillStyle = alpha(COL[sg.stage], sg.stage === "decode" ? 0.95 : 0.7);
    ctx.fillRect(X(sg.t0), ty + 2, Math.max(1, X(Math.min(now, sg.t1)) - X(sg.t0)), 12);
  });
  SIM.checks.forEach((k) => {
    if (k.t > now) return;
    ctx.fillStyle = k.took.length ? c.accent : alpha(c.text, 0.45);
    ctx.fillRect(X(k.t), ty + 18, 1.5, k.took.length ? 8 : 4);
  });
  SIM.reads.forEach((r) => {
    if (r.t0 > now) return;
    fillRound(ctx, X(r.t0), ty + 30, Math.max(2, X(Math.min(now, r.t1)) - X(r.t0)), 4, 2, alpha(c.accent, 0.5));
  });
  line(ctx, X(now), ty - 4, X(now), ty + 38, c.text, 1.5);
  paragraph(ctx, "■ decode / map / process   ■ claim / plan   ■ submit   | poll   ▬ read", box.x, ty + 52, W, { size: 10, family: "mono", color: c.text3, a: intro });
  const finished = SIM.done.filter((d) => d.t <= now).length;
  text(ctx, compact ? `${finished} done · ${SIM.claims.filter((k) => k.t <= now).length} claimed · 0 µs idle` : `${finished} cells finished · ${SIM.claims.filter((k) => k.t <= now).length} claimed · 0 µs idle`, box.x + W, ty - 8, { size: 11, family: "mono", weight: 600, color: c.teal, align: "right", a: intro });
}

export default { draw, continuous: false };
