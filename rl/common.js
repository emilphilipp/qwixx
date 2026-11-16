/* RL bot: afterstate features, MLP inference and a game loop shared by gen.js and eval.js. */
const Q = require('../qwixx-engine.js');
const path = require('path');
Q.loadTableFromDir(path.join(__dirname, '..', 'table'));

const NF = Q.RL_NF;
const FEATURE_NAMES = [
  'active', 'npl', 'myTurn', 'myEV', 'myScore', 'myK', 'myEligible', 'closedAfter', 'ends',
  'oppEVmax', 'oppEV2', 'oppEVmean', 'oppScoreMax', 'oppScoreMean', 'oppKmax', 'oppEligible', 'lamSum',
  'evLead', 'scoreLead', 'endsSign', 'evGap', 'myPotential', 'oppPotential',
];
/* features, MLP and option scoring live in the engine (shared with the browser) */
const features = (G, me, ctx, o, bestValue, out, off) => Q.rlFeatures(G, me, me === G.active, ctx, o, bestValue, out, off);
const loadModel = file => Q.prepareModel(JSON.parse(require('fs').readFileSync(file, 'utf8')));

/* ---------- policies: (G, me) -> {ctx, opts, best (Expert), chosen} ---------- */
const decide = (G, me, model) => Q.rlDecide(G, me, me === G.active, G.dice.w[0], G.dice.w[1], G.dice.c, model);
const expertPolicy = (G, me) => decide(G, me, null);
const makeRL = model => (G, me) => decide(G, me, model);

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
module.exports = { Q, NF, FEATURE_NAMES, features, loadModel, expertPolicy, makeRL, playGame, POOL, optionRollout, rolloutValues, makeRollout };
