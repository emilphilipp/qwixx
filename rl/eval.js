// Paired evaluation of the RL bot vs the Expert: node rl/eval.js MODEL.json|rollout:M:PRUNE [games per pool] [seed0] [pools]
// Same seed => same dice, so each game is played once with the Expert and once with the RL bot in the hero seat.
const C = require('./common.js');
const [modelFile, nS, seedS, only] = process.argv.slice(2);
const N = +(nS || 20000), seed0 = +(seedS || 9000000);
const hero = modelFile.startsWith('rollout:') ? C.makeRollout(...modelFile.split(':').slice(1).map(Number)) : C.makeRL(C.loadModel(modelFile), 0);
const pol = { rl: hero, ex: (G, p) => C.expertPolicy(G, p) };

const pools = {
  'patient x4': () => Array(4).fill('patient'),
  'careful x4': () => Array(4).fill('careful'),
  'balanced x4': () => Array(4).fill('balanced'),
  'greedy x4': () => Array(4).fill('greedy'),
  'expert x4': () => Array(4).fill('expert'),
  'mixed 4': () => ['patient', 'careful', 'balanced', 'greedy'],
  'random 1-5': r => Array.from({ length: 1 + Math.floor(r() * 5) }, () => C.POOL[Math.floor(r() * 5)]),
};

function stats(d) {
  const n = d.length, m = d.reduce((a, b) => a + b, 0) / n;
  const v = d.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1);
  return [m, Math.sqrt(v / n)];
}
const pct = x => (100 * x).toFixed(1).padStart(5);

console.log('pool          Expert win%   RL win%    diff (pp) ± SE');
for (const [name, mk] of Object.entries(pools)) {
  if (only && !only.split(',').includes(name)) continue;
  const r = C.Q.mulberry32(seed0 + name.length), ex = [], rl = [], diff = [];
  for (let g = 0; g < N; g++) {
    const opp = mk(r), seat = g % (opp.length + 1);
    const specs = h => { const s = opp.map(k => ({ kind: k })); s.splice(seat, 0, { kind: 'policy:' + h }); return s; };
    const a = C.playGame(specs('ex'), seed0 + g, pol).share[seat];
    const b = C.playGame(specs('rl'), seed0 + g, pol).share[seat];
    ex.push(a); rl.push(b); diff.push(b - a);
  }
  const [d, se] = stats(diff);
  console.log(name.padEnd(13), pct(stats(ex)[0]), '      ', pct(stats(rl)[0]), '   ', (100 * d).toFixed(2).padStart(6), '±', (100 * se).toFixed(2));
}

// head-to-head: RL and Expert at the same table with 0-3 other bots; seats rotate
if (!only || only.split(',').includes('h2h')) {
  const r = C.Q.mulberry32(seed0 + 77), sr = [], se_ = [];
  for (let g = 0; g < N; g++) {
    const others = Array.from({ length: Math.floor(r() * 4) }, () => C.POOL[Math.floor(r() * 5)]);
    const specs = [{ kind: 'policy:rl' }, { kind: 'policy:ex' }, ...others.map(k => ({ kind: k }))];
    const rot = g % specs.length, s = specs.slice(rot).concat(specs.slice(0, rot));
    const res = C.playGame(s, seed0 + 500000 + g, pol);
    const iRL = (specs.length - rot) % specs.length, iEX = (iRL + 1) % specs.length;
    sr.push(res.share[iRL]); se_.push(res.share[iEX]);
  }
  const [d, se] = stats(sr.map((x, i) => x - se_[i]));
  console.log('head-to-head ', pct(stats(se_)[0]), '      ', pct(stats(sr)[0]), '   ', (100 * d).toFixed(2).padStart(6), '±', (100 * se).toFixed(2));
}
