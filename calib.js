// Calibration: at turn 8 predict the Expert seat's win chance by rollouts, then play the real game out.
const Q = require('./qwixx-engine.js'); Q.loadTableFromDir(process.argv[2] || './table');
const NG = +(process.argv[3] || 600), NR = 200, kinds = ['balanced', 'greedy', 'careful', 'patient', 'expert'];
const buckets = Array.from({ length: 5 }, () => ({ p: 0, w: 0, n: 0 })); let brier = 0, cnt = 0, t0 = Date.now();
for (let g = 0; g < NG; g++) {
  const nb = 1 + (g % 5), seat = g % (nb + 1), specs = []; let q = 0;
  for (let p = 0; p <= nb; p++) specs.push(p === seat ? { name: 'me', kind: 'expert' } : { name: 'b', kind: kinds[(g + q++) % 4] });
  const G = Q.newGame(specs, 5000 + g); let pred = null;
  while (!G.over) {
    Q.rollDice(G);
    if (G.turn === 8 && pred === null) {
      const myPlan = G.plans[seat];
      G.players.forEach((pl, p) => { const pn = G.plans[p]; if (p !== seat && pn && pn.w) Q.doMark(G, p, pn.w.row, pn.w.n); });
      let s = 0; for (let j = 0; j < NR; j++) s += Q.rollout(G, seat, 'white', myPlan.w, 777000 + j); pred = s / NR;
      if (myPlan.w) Q.doMark(G, seat, myPlan.w.row, myPlan.w.n);
      Q.updateClosed(G); Q.botColour(G); Q.finishTurn(G); continue;
    }
    Q.botsWhite(G); Q.updateClosed(G); Q.botColour(G); Q.finishTurn(G);
  }
  if (pred === null) continue;
  const sc = G.players.map((_, p) => Q.scoreOf(G, p)), mx = Math.max(...sc);
  const won = sc[seat] < mx ? 0 : 1 / sc.filter(x => x === mx).length;
  const b = buckets[Math.min(4, Math.floor(pred * 5))]; b.p += pred; b.w += won; b.n++;
  brier += (pred - won) ** 2; cnt++;
}
console.log('predicted vs actual win rate (games with 1-5 bots, prediction at turn 8):');
buckets.forEach((b, i) => b.n && console.log(`  pred ${(i * 20)}-${i * 20 + 20}%: n=${b.n}  mean pred ${(100 * b.p / b.n).toFixed(1)}%  actual ${(100 * b.w / b.n).toFixed(1)}%`));
console.log('Brier', (brier / cnt).toFixed(4), ' games', cnt, ((Date.now() - t0) / 1000).toFixed(0) + 's');
