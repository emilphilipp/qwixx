// Where does the RL bot deviate from the Expert? node rl/analyze.js MODEL.json [games]
const C = require('./common.js');
const Q = C.Q;
const [modelFile, nS] = process.argv.slice(2);
const N = +(nS || 4000);
const rlp = C.makeRL(C.loadModel(modelFile), 0);

const marks = o => o.nm, locks = (G, me, o) => o.rows.filter((r, i) => !G.closed[i] && Q.RL[r] && !Q.RL[G.players[me].rows[i]]).length;
const cat = {}, add = (k, gap) => { const c = cat[k] || (cat[k] = { n: 0, gap: 0 }); c.n++; c.gap += gap; };
let dec = 0, diff = 0;
const byLead = {}, byTurn = {};
const bucket = (o, k, v) => { const b = o[k] || (o[k] = { n: 0, d: 0 }); b.n++; b.d += v; };

function rec(G, me) {
  const r = rlp(G, me), e = r.best, c = r.chosen;
  if (r.opts.length < 2) return r;
  dec++;
  const lead = Q.rowsScore(G.players[me].rows, G.players[me].k) - r.ctx.scMax;
  const lb = lead <= -15 ? 'behind >15' : lead < -5 ? 'behind 5-15' : lead <= 5 ? 'level ±5' : lead <= 15 ? 'ahead 5-15' : 'ahead >15';
  const tb = 'closed rows ' + r.ctx.closed;
  const d = c !== e ? 1 : 0;
  bucket(byLead, lb, d); bucket(byTurn, tb, d);
  if (!d) return r;
  diff++;
  const gap = e.value - c.value, active = me === G.active;
  if (active && c.nm === 0) add('RL takes a penalty (Expert marks)', gap);
  else if (active && e.nm === 0) add('RL marks instead of a penalty', gap);
  else if (marks(c) > marks(e)) add('RL marks more cells', gap);
  else if (marks(c) < marks(e)) add('RL marks fewer cells (skips)', gap);
  else add('same number of marks, other cells', gap);
  if (locks(G, me, c) > locks(G, me, e)) add('  … of which RL closes a row', gap);
  if (locks(G, me, c) < locks(G, me, e)) add('  … of which RL avoids closing a row', gap);
  const endsC = r.ctx.closed + locks(G, me, c) >= 2 || c.k >= 4, endsE = r.ctx.closed + locks(G, me, e) >= 2 || e.k >= 4;
  if (endsC && !endsE) add('  … of which RL ends the game now', gap);
  if (!endsC && endsE) add('  … of which RL keeps the game going', gap);
  return r;
}

const pools = ['patient', 'careful', 'balanced', 'greedy', 'expert'], rng = Q.mulberry32(42);
const game = { ex: { sc: 0, win: 0, len: 0, pen: 0, locks: 0 }, rl: { sc: 0, win: 0, len: 0, pen: 0, locks: 0 } };
for (let g = 0; g < N; g++) {
  const opp = Array.from({ length: 1 + Math.floor(rng() * 5) }, () => pools[Math.floor(rng() * 5)]), seat = g % (opp.length + 1);
  for (const h of ['ex', 'rl']) {
    const specs = opp.map(k => ({ kind: k })); specs.splice(seat, 0, { kind: 'policy:' + h });
    const res = C.playGame(specs, 777 + g, { ex: (G, p) => C.expertPolicy(G, p), rl: h === 'rl' ? rec : rlp });
    const s = game[h], pl = res.G.players[seat];
    s.sc += res.sc[seat]; s.win += res.share[seat]; s.len += res.G.turn / specs.length; s.pen += pl.k;
    s.locks += pl.rows.filter(x => Q.RL[x]).length;
  }
}
console.log(`${N} paired games vs random tables of 1-5 bots (hero seat):`);
console.log('           win%   score  rounds  penalties  rows I closed');
for (const h of ['ex', 'rl']) { const s = game[h]; console.log((h === 'ex' ? 'Expert' : 'RL bot').padEnd(9), (100 * s.win / N).toFixed(1).padStart(6), (s.sc / N).toFixed(1).padStart(7), (s.len / N).toFixed(1).padStart(7), (s.pen / N).toFixed(2).padStart(10), (s.locks / N).toFixed(2).padStart(10)); }
console.log(`\nRL bot decisions with a choice: ${dec}; differs from the Expert in ${(100 * diff / dec).toFixed(1)}%`);
console.log('\nkind of deviation                          share of deviations   mean expected score given up');
for (const [k, c] of Object.entries(cat)) console.log(k.padEnd(44), (100 * c.n / diff).toFixed(1).padStart(6) + '%', (c.gap / c.n).toFixed(2).padStart(18));
console.log('\ndeviation rate by score position before the move:');
for (const k of ['behind >15', 'behind 5-15', 'level ±5', 'ahead 5-15', 'ahead >15']) if (byLead[k]) console.log('  ' + k.padEnd(13), (100 * byLead[k].d / byLead[k].n).toFixed(1).padStart(5) + '%', ` (n=${byLead[k].n})`);
console.log('deviation rate by game stage:');
for (const k of Object.keys(byTurn).sort()) console.log('  ' + k.padEnd(13), (100 * byTurn[k].d / byTurn[k].n).toFixed(1).padStart(5) + '%', ` (n=${byTurn[k].n})`);
