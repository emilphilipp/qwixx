/* RL bot: afterstate features, MLP inference and a game loop shared by gen.js and eval.js. */
const Q = require('../qwixx-engine.js');
const path = require('path');
Q.loadTableFromDir(path.join(__dirname, '..', 'table'));

const NF = 23;
const FEATURE_NAMES = [
  'active', 'npl', 'myTurn', 'myEV', 'myScore', 'myK', 'myEligible', 'closedAfter', 'ends',
  'oppEVmax', 'oppEV2', 'oppEVmean', 'oppScoreMax', 'oppScoreMean', 'oppKmax', 'oppEligible', 'lamSum',
  'evLead', 'scoreLead', 'endsSign', 'evGap', 'myPotential', 'oppPotential',
];

/* features that do not depend on my option: computed once per decision */
function context(G, me) {
  const n = G.players.length, ev = [], sc = [];
  let kmax = 0;
  G.players.forEach((pl, p) => {
    if (p === me) return;
    ev.push(Q.stateValue(pl.rows, pl.k, Q.threats(G, p)));
    sc.push(Q.rowsScore(pl.rows, pl.k));
    if (pl.k > kmax) kmax = pl.k;
  });
  ev.sort((a, b) => b - a);
  let elig = 0;
  for (let i = 0; i < 4; i++) if (!G.closed[i] && G.players.some((pl, p) => p !== me && Q.RC[pl.rows[i]] >= 5)) elig++;
  const lam = Q.threats(G, me);
  return {
    n, lam, lamSum: lam.reduce((a, b) => a + b, 0), kmax, elig,
    evMax: ev[0], ev2: ev.length > 1 ? ev[1] : ev[0], evMean: ev.reduce((a, b) => a + b, 0) / ev.length,
    scMax: Math.max(...sc), scMean: sc.reduce((a, b) => a + b, 0) / sc.length,
    closed: G.closed.filter(Boolean).length,
  };
}

/* options for `me` on the current dice, each with its afterstate value (the Expert's criterion) */
function options(G, me, ctx) {
  const pl = G.players[me], active = me === G.active, d = G.dice;
  const opts = Q.enumOptions(pl.rows, G.closed, d.w[0], d.w[1], d.c, active);
  for (const o of opts) {
    o.k = pl.k + (active && o.nm === 0 ? 1 : 0);
    o.value = Q.stateValue(o.rows, o.k, ctx.lam);
  }
  return opts;
}

function features(G, me, ctx, o, bestValue, out, off) {
  const pl = G.players[me];
  let newLocks = 0, myElig = 0;
  for (let i = 0; i < 4; i++) {
    if (G.closed[i]) continue;
    if (Q.RL[o.rows[i]] && !Q.RL[pl.rows[i]]) newLocks++;
    else if (!Q.RL[o.rows[i]] && Q.RC[o.rows[i]] >= 5) myElig++;
  }
  const score = Q.rowsScore(o.rows, o.k), closedAfter = ctx.closed + newLocks;
  const ends = closedAfter >= 2 || o.k >= 4 ? 1 : 0;
  const lead = score - ctx.scMax;
  const f = [
    me === G.active ? 1 : 0, ctx.n, G.turn / ctx.n, o.value, score, o.k, myElig, closedAfter, ends,
    ctx.evMax, ctx.ev2, ctx.evMean, ctx.scMax, ctx.scMean, ctx.kmax, ctx.elig, ctx.lamSum,
    o.value - ctx.evMax, lead, ends * Math.sign(lead), o.value - bestValue, o.value - score, ctx.evMax - ctx.scMax,
  ];
  for (let j = 0; j < NF; j++) out[off + j] = f[j];
}

/* ---------- MLP (weights exported by train.py) ---------- */
function loadModel(file) {
  const m = JSON.parse(require('fs').readFileSync(file, 'utf8'));
  m.layers = m.layers.map(L => ({ W: Float64Array.from(L.W.flat()), b: Float64Array.from(L.b), nin: L.W[0].length, nout: L.W.length }));
  return m;
}
/* returns the logit of P(win) */
function forward(m, x, off) {
  let h = new Float64Array(NF);
  for (let j = 0; j < NF; j++) h[j] = (x[off + j] - m.mean[j]) / m.std[j];
  for (let l = 0; l < m.layers.length; l++) {
    const L = m.layers[l], o = new Float64Array(L.nout), last = l === m.layers.length - 1;
    for (let r = 0; r < L.nout; r++) {
      let s = L.b[r]; const base = r * L.nin;
      for (let c = 0; c < L.nin; c++) s += L.W[base + c] * h[c];
      o[r] = last || s > 0 ? s : 0;
    }
    h = o;
  }
  return h[0];
}

/* ---------- policies: (G, me) -> chosen option; may push the chosen afterstate's features to rec ---------- */
function expertPolicy(G, me) {
  const ctx = context(G, me), opts = options(G, me, ctx);
  let best = opts[0];
  for (const o of opts) if (o.value > best.value + 1e-9) best = o;
  return { ctx, opts, best };
}
function makeRL(model, eps) {
  const buf = new Float64Array(NF);
  return function (G, me) {
    const { ctx, opts, best } = expertPolicy(G, me);
    if (!model) return { ctx, opts, best, chosen: best };
    let chosen = null, bl = -Infinity;
    for (const o of opts) {
      features(G, me, ctx, o, best.value, buf, 0);
      o.logit = forward(model, buf, 0);
      if (o.logit > bl) { bl = o.logit; chosen = o; }
    }
    return { ctx, opts, best, chosen };
  };
}

/* play one game. specs: [{kind}] with kind in BOT_KINDS or 'policy:<name>'; policies: {name: fn}.
   rec (optional): {seat, eps, rng, rows: []} records features of the seat's chosen afterstates. */
function playGame(specs, seed, policies, rec) {
  const G = Q.newGame(specs.map((s, p) => ({ name: 'p' + p, kind: s.kind.startsWith('policy:') ? 'expert' : s.kind })), seed);
  const pol = specs.map(s => (s.kind.startsWith('policy:') ? policies[s.kind.slice(7)] : null));
  let guard = 0;
  while (!G.over) {
    Q.rollDice(G);
    for (let p = 0; p < specs.length; p++) {
      if (!pol[p]) continue;
      const r = pol[p](G, p);
      let ch = r.chosen || r.best;
      if (rec && rec.seat === p) {
        if (rec.eps && rec.rng() < rec.eps) ch = r.opts[Math.floor(rec.rng() * r.opts.length)];
        const x = new Float64Array(NF);
        features(G, p, r.ctx, ch, r.best.value, x, 0);
        rec.rows.push(x);
      }
      G.plans[p] = ch;
    }
    Q.botsWhite(G); Q.updateClosed(G); Q.botColour(G); Q.finishTurn(G);
    if (++guard > 500) throw new Error('game does not end');
  }
  const sc = G.players.map((_, p) => Q.scoreOf(G, p)), mx = Math.max(...sc);
  const share = sc.map(s => (s < mx ? 0 : 1 / sc.filter(x => x === mx).length));
  return { G, sc, share };
}

/* ---------- rollouts: win share of `me` after taking option `o` now and everybody playing their kind later
   (policy seats play the Expert). Same seed => same future dice for every option. ---------- */
function optionRollout(G, me, o, seed) {
  const S = { players: G.players.map(p => ({ kind: p.kind, rows: p.rows.slice(), k: p.k, tm: 0 })), closed: G.closed.slice() };
  const a = G.active, M = S.players[me];
  S.players.forEach((pl, p) => { const w = p === me ? o.w : G.plans[p] && G.plans[p].w; if (w) Q.markL(pl, w.row, w.n); });
  Q.updateClosed(S);
  const c = a === me ? o.c : G.plans[a] && G.plans[a].c;
  if (c && !S.closed[c.row]) Q.markL(S.players[a], c.row, c.n);
  if (S.players[a].tm === 0) S.players[a].k++;
  Q.updateClosed(S);
  const rng = Q.mulberry32(seed);
  let t = G.turn + 1, guard = 0;
  while (!Q.isOver(S) && guard++ < 400) { Q.simTurn(S, t % S.players.length, rng); t++; }
  const sc = S.players.map(pl => Q.rowsScore(pl.rows, pl.k)), mx = Math.max(...sc);
  return sc[me] < mx ? 0 : 1 / sc.filter(x => x === mx).length;
}
/* rollout value of every option within `prune` points of the Expert's best (others get -1) */
function rolloutValues(G, me, opts, best, M, prune, seed) {
  for (const o of opts) {
    o.ro = -1;
    if (o.value < best.value - prune) continue;
    let s = 0;
    for (let j = 0; j < M; j++) s += optionRollout(G, me, o, seed + j);
    o.ro = s / M;
  }
}
/* switch away from the Expert's move only if the rollout advantage exceeds `margin` (noise guard) */
function makeRollout(M, prune, margin = 0) {
  let calls = 0;
  return function (G, me) {
    const r = expertPolicy(G, me);
    if (r.opts.length > 1) {
      rolloutValues(G, me, r.opts, r.best, M, prune, 0x9e3779b1 ^ (G.turn * 7919 + (calls++) * 104729));
      r.chosen = r.best;
      for (const o of r.opts) if (o.ro > r.chosen.ro + 1e-12) r.chosen = o;
      if (r.chosen.ro < r.best.ro + margin) r.chosen = r.best;
    }
    return r;
  };
}

const POOL = ['patient', 'careful', 'balanced', 'greedy', 'expert'];
module.exports = { Q, NF, FEATURE_NAMES, context, options, features, loadModel, forward, expertPolicy, makeRL, playGame, POOL, optionRollout, rolloutValues, makeRollout };
