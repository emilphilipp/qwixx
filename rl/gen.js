// Self-play data: node rl/gen.js MODEL|none EPS GAMES SEED0 OUT.bin
// Writes float32 rows [features..., win share, game id] for every decision of the RL seat.
const C = require('./common.js');
const fs = require('fs');
const [modelFile, epsS, gamesS, seedS, out] = process.argv.slice(2);
const model = modelFile === 'none' ? null : C.loadModel(modelFile);
const eps = +epsS, N = +gamesS, seed0 = +seedS;
const rl = C.makeRL(model, eps), rng = C.Q.mulberry32(seed0 ^ 0x5bd1e995);

/* table setup: 1-5 opponents; half the games use one kind for all of them, the rest a random mix */
function setup(r) {
  const nb = 1 + Math.floor(r() * 5), seat = Math.floor(r() * (nb + 1)), same = r() < 0.5;
  const k0 = C.POOL[Math.floor(r() * C.POOL.length)], specs = [];
  for (let p = 0; p <= nb; p++) specs.push({ kind: p === seat ? 'policy:rl' : same ? k0 : C.POOL[Math.floor(r() * C.POOL.length)] });
  return { specs, seat };
}

const chunks = [];
let total = 0, wins = 0;
for (let g = 0; g < N; g++) {
  const { specs, seat } = setup(rng);
  const rec = { seat, eps, rng, rows: [] };
  const { share } = C.playGame(specs, seed0 + g, { rl }, rec);
  const a = new Float32Array(rec.rows.length * (C.NF + 2));
  rec.rows.forEach((x, i) => { a.set(x, i * (C.NF + 2)); a[i * (C.NF + 2) + C.NF] = share[seat]; a[i * (C.NF + 2) + C.NF + 1] = seed0 + g; });
  chunks.push(Buffer.from(a.buffer)); total += rec.rows.length; wins += share[seat];
}
fs.writeFileSync(out, Buffer.concat(chunks));
console.log(JSON.stringify({ out, games: N, rows: total, winShare: wins / N }));
