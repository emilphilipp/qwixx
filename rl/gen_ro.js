// Rollout-labelled data: node rl/gen_ro.js MODEL|none GAMES SEED0 M PRUNE OUT.bin
// The hero seat plays MODEL (or the Expert). At every hero decision with a choice, each option within PRUNE points
// of the Expert's best gets its rollout win share (M rollouts, common random numbers).
// Rows: float32 [features..., rollout share, game index, decision index].
const C = require('./common.js');
const fs = require('fs');
const [modelFile, gamesS, seedS, MS, pruneS, out] = process.argv.slice(2);
const N = +gamesS, seed0 = +seedS, M = +MS, prune = +pruneS, W = C.NF + 3;
const play = C.makeRL(modelFile === 'none' ? null : C.loadModel(modelFile), 0), rng = C.Q.mulberry32(seed0 ^ 0x2545f491);
let g = 0, dec = 0, rows = [], ndec = 0;
const x = new Float64Array(C.NF);
function hero(G, me) {
  const r = play(G, me);
  const cand = r.opts.filter(o => o.value >= r.best.value - prune);
  if (cand.length > 1) {
    C.rolloutValues(G, me, cand, r.best, M, prune, (seed0 + g) * 131 + dec);
    for (const o of cand) {
      C.features(G, me, r.ctx, o, r.best.value, x, 0);
      const a = new Float32Array(W); a.set(x); a[C.NF] = o.ro; a[C.NF + 1] = g; a[C.NF + 2] = dec;
      rows.push(a);
    }
    ndec++;
  }
  dec++;
  return r;
}
let chunks = [], nrows = 0;
fs.writeFileSync(out, '');
for (g = 0; g < N; g++) {
  dec = 0;
  const nb = 1 + Math.floor(rng() * 5), seat = Math.floor(rng() * (nb + 1)), same = rng() < 0.5;
  const k0 = C.POOL[Math.floor(rng() * 5)], specs = [];
  for (let p = 0; p <= nb; p++) specs.push({ kind: p === seat ? 'policy:hero' : same ? k0 : C.POOL[Math.floor(rng() * 5)] });
  C.playGame(specs, seed0 + g, { hero });
  for (const a of rows) chunks.push(Buffer.from(a.buffer));
  rows = [];
  if (g % 50 === 49 || g === N - 1) { fs.appendFileSync(out, Buffer.concat(chunks)); nrows += chunks.length; chunks = []; }   // survive interruptions
}
console.log(JSON.stringify({ out, games: N, decisions: ndec, rows: nrows }));
