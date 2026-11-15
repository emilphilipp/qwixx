// node test_engine.js TABLEDIR [games]
const Q = require('./qwixx-engine.js');
const dir = process.argv[2] || './table', N = +(process.argv[3] || 2000);
Q.loadTableFromDir(dir);

function playBots(specs, seed) {
  const G = Q.newGame(specs, seed);
  let guard = 0;
  while (!G.over) {
    Q.rollDice(G); Q.botsWhite(G); Q.botColour(G); Q.finishTurn(G);
    if (++guard > 500) throw new Error('game does not end');
  }
  return G;
}
// sanity: invariants of every finished game
function check(G) {
  G.players.forEach((pl, p) => {
    for (let i = 0; i < 4; i++) {
      const m = pl.marks[i];
      for (let j = 1; j < m.length; j++) if (m[j] <= m[j - 1]) throw new Error('marks not increasing');
      const c = m.length + (Q.RL[pl.rows[i]] ? 1 : 0);
      if (c !== Q.RC[pl.rows[i]]) throw new Error('count mismatch ' + c + ' vs ' + Q.RC[pl.rows[i]]);
      if (m.includes(11) && m.length < 5 + 1) throw new Error('lock with <5 marks');
    }
  });
}
const kinds = ['patient', 'careful', 'balanced', 'greedy'];
function scenario(name, oppKinds, heroKind) {
  let hs = 0, os = 0, win = 0, turns = 0;
  for (let g = 0; g < N; g++) {
    const seat = g % 5, specs = []; let q = 0;
    for (let p = 0; p < 5; p++) specs.push(p === seat ? { name: 'hero', kind: heroKind } : { name: 'b' + p, kind: oppKinds[q++] });
    const G = playBots(specs, 1000 + g); check(G);
    const sc = G.players.map((_, p) => Q.scoreOf(G, p));
    const h = sc[seat]; let o = 0, tie = 0, beat = true;
    sc.forEach((s, p) => { if (p === seat) return; o += s; if (s > h) beat = false; else if (s === h) tie++; });
    hs += h; os += o / 4; if (beat) win += 1 / (1 + tie); turns += G.turn;
  }
  console.log(name.padEnd(14), heroKind.padEnd(9), 'hero', (hs / N).toFixed(2), ' opp', (os / N).toFixed(2), ' win', (win / N).toFixed(3), ' turns', (turns / N).toFixed(1));
}
console.log('single-row f(p=0,c=0):', [0, 0.1, 0.3, 1].map(l => Q.Fint(l, 0).toFixed(2)).join(' / '));
const t0 = Date.now();
scenario('balanced x4', Array(4).fill('balanced'), 'expert');
scenario('balanced x4', Array(4).fill('balanced'), 'balanced');
scenario('patient x4', Array(4).fill('patient'), 'expert');
scenario('mixed', kinds, 'expert');
console.log('ok', ((Date.now() - t0) / 1000).toFixed(1) + 's');
