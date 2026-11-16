/* Qwixx engine: rules, bots, DP-table value function with row-closure threat correction.
 * Works in the browser (window.Qwixx) and in node (require). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Qwixx = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const NRS = 62, NR = 4;
  const DIR = [1, 1, -1, -1];                    // +1: 2→12 (red, yellow), -1: 12→2 (green, blue)
  const COLORS = ['red', 'yellow', 'green', 'blue'];

  /* ---------- row states: (p = last marked position 0..11, c = marks incl. lock bonus, locked) ---------- */
  const RP = new Int8Array(NRS), RC = new Int8Array(NRS), RL = new Int8Array(NRS);
  (function () {
    let id = 0; RP[id] = 0; RC[id] = 0; RL[id] = 0; id++;
    for (let p = 1; p <= 10; p++) for (let c = 1; c <= p; c++) { RP[id] = p; RC[id] = c; RL[id] = 0; id++; }
    for (let c = 7; c <= 12; c++) { RP[id] = 11; RC[id] = c; RL[id] = 1; id++; }
  })();
  const posOf = (d, n) => (d > 0 ? n - 1 : 13 - n);
  const numAt = (d, pos) => (d > 0 ? pos + 1 : 13 - pos);
  const TRt = [new Int8Array(NRS * 14).fill(-1), new Int8Array(NRS * 14).fill(-1)];
  (function () {
    for (let di = 0; di < 2; di++) {
      const d = di === 0 ? 1 : -1;
      for (let r = 0; r < NRS; r++) for (let n = 2; n <= 12; n++) {
        if (RL[r]) continue;
        const q = posOf(d, n);
        if (q <= RP[r]) continue;
        let np, nc, nl;
        if (q === 11) { if (RC[r] < 5) continue; np = 11; nc = RC[r] + 2; nl = 1; }
        else { np = q; nc = RC[r] + 1; nl = 0; }
        for (let s = 0; s < NRS; s++) if (RP[s] === np && RC[s] === nc && RL[s] === nl) { TRt[di][r * 14 + n] = s; break; }
      }
    }
  })();
  /* new row state after marking number n in row i (or -1) */
  const tr = (i, st, n) => (n < 2 || n > 12 ? -1 : TRt[DIR[i] > 0 ? 0 : 1][st * 14 + n]);

  const T = c => (c * (c + 1)) / 2;
  const nLocks = rows => RL[rows[0]] + RL[rows[1]] + RL[rows[2]] + RL[rows[3]];
  const isTerminal = (rows, k) => k >= 4 || nLocks(rows) >= 2;
  const rowsScore = (rows, k) => T(RC[rows[0]]) + T(RC[rows[1]]) + T(RC[rows[2]]) + T(RC[rows[3]]) - 5 * k;

  const PRB = n => (6 - Math.abs(n - 7)) / 36;
  function skipCost(i, st, n) {                    // probability mass of the cells jumped over
    const q = posOf(DIR[i], n); let c = 0;
    for (let x = RP[st] + 1; x < q; x++) c += PRB(numAt(DIR[i], x));
    return c;
  }

  /* ---------- value table (stationary DP, loaded from disk/network) ---------- */
  const Tbl = { ready: false, H: 0.035, ACT: 0.2, min: 0, scale: 0, np2: 0, tb: null };
  function setTable(metaText, bufs) {
    const m = metaText.trim().split(/\s+/).map(Number);
    Tbl.min = m[0]; Tbl.scale = m[1]; Tbl.H = m[2]; Tbl.ACT = 1 / m[3]; Tbl.np2 = m[4];
    Tbl.tb = bufs.map(b => new Uint16Array(b.buffer ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) : b));
    Tbl.ready = true;
    solveAllSingle();
  }
  async function loadTable(base, onProgress) {
    const meta = await (await fetch(base + 'meta.txt')).text();
    const bufs = [];
    for (let k = 0; k < 4; k++) {
      let r = await fetch(base + 'table_k' + k + '.bin');
      if (r.ok) bufs.push(await r.arrayBuffer());
      else {                                         // hosted copy: base64 inside JSON
        r = await fetch(base + 'table_k' + k + '.json');
        if (!r.ok) throw new Error('table_k' + k + ' not found');
        const bin = atob((await r.json()).b64), u = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
        bufs.push(u.buffer);
      }
      if (onProgress) onProgress((k + 1) / 4);
    }
    setTable(meta, bufs);
  }
  function loadTableFromDir(dir) {
    const fs = require('fs');
    setTable(fs.readFileSync(dir + '/meta.txt', 'utf8'), [0, 1, 2, 3].map(k => fs.readFileSync(dir + '/table_k' + k + '.bin')));
  }
  function vtab(r, k) {
    let a = r[0], b = r[1], c = r[2], d = r[3], t;
    if (a > b) { t = a; a = b; b = t; }
    if (c > d) { t = c; c = d; d = t; }
    return Tbl.min + Tbl.scale * Tbl.tb[k][(b * (b + 1) / 2 + a) * Tbl.np2 + (d * (d + 1) / 2 + c)];
  }

  /* ---------- single-row DP with closure hazard (threat correction) ---------- */
  const LG = [0, .01, .02, .035, .05, .075, .1, .15, .2, .3, .45, .6, 1.0];
  const NG = LG.length;
  const F = LG.map(() => new Float64Array(NRS));
  function solveSingle(g) {
    const H = Tbl.H, ACT = Tbl.ACT, lam = LG[g], rho = 1 - (1 - H) * (1 - lam), Fg = F[g];
    for (let s = 0; s < NRS; s++) if (RL[s]) Fg[s] = T(RC[s]);
    const VAL = s2 => rho * T(RC[s2]) + (1 - rho) * Fg[s2];
    for (let p = 10; p >= 0; p--) for (let s = 0; s < NRS; s++) {
      if (RL[s] || RP[s] !== p) continue;
      const Ts = T(RC[s]), mpas = [], ppas = [], mact = [], pact = [];
      for (let w1 = 1; w1 <= 6; w1++) for (let w2 = w1; w2 <= 6; w2++) {
        const pw = (w1 === w2 ? 1 : 2) / 36;
        const s1 = tr(0, s, w1 + w2), mw = s1 >= 0 ? VAL(s1) : -1e18;
        mpas.push(mw); ppas.push(pw);
        for (let cd = 1; cd <= 6; cd++) {
          let best = mw;
          for (let j = 0; j < 2; j++) {
            const w = j ? w2 : w1; if (j && w2 === w1) break;
            const a = tr(0, s, w + cd); if (a >= 0 && VAL(a) > best) best = VAL(a);
            if (s1 >= 0) { const b = tr(0, s1, w + cd); if (b >= 0 && VAL(b) > best) best = VAL(b); }
          }
          mact.push(best); pact.push(pw / 6);
        }
      }
      let x = Ts;
      for (let it = 0; it < 2000; it++) {
        const Sx = rho * Ts + (1 - rho) * x; let a = 0, b = 0;
        for (let q = 0; q < mact.length; q++) a += pact[q] * (mact[q] > Sx ? mact[q] : Sx);
        for (let q = 0; q < mpas.length; q++) b += ppas[q] * (mpas[q] > Sx ? mpas[q] : Sx);
        const xn = ACT * a + (1 - ACT) * b;
        if (Math.abs(xn - x) < 1e-9) { x = xn; break; }
        x = xn;
      }
      Fg[s] = x;
    }
  }
  function solveAllSingle() { for (let g = 0; g < NG; g++) solveSingle(g); }
  function Fint(lam, s) {
    if (RL[s]) return T(RC[s]);
    let g = 0; while (g < NG - 2 && lam > LG[g + 1]) g++;
    let t = (lam - LG[g]) / (LG[g + 1] - LG[g]); t = t < 0 ? 0 : t > 1 ? 1 : t;
    return F[g][s] * (1 - t) + F[g + 1][s] * t;
  }

  const KAPPA = 1.0;                               // strength of the threat correction (tuned in simulation)
  const PL = 0.034;                                // chance per turn that one qualified opponent locks a row
  /* expected final score after a move: table value with end hazard + closure correction */
  function stateValue(rows, k, lam, kappa) {
    if (isTerminal(rows, k)) return rowsScore(rows, k);
    const sc = rowsScore(rows, k);
    let v = (1 - Tbl.H) * vtab(rows, k) + Tbl.H * sc;
    kappa = kappa === undefined ? KAPPA : kappa;
    if (kappa && lam) for (let i = 0; i < NR; i++) if (lam[i] > 0) v += kappa * (Fint(lam[i], rows[i]) - F[0][rows[i]]);
    return v;
  }
  /* closure risk per row from everybody else's boards: lock needs >=5 marks and the last cell still open */
  function threats(G, me) {
    const lam = [0, 0, 0, 0];
    for (let i = 0; i < NR; i++) {
      if (G.closed[i]) { lam[i] = 1; continue; }
      let prod = 1;
      G.players.forEach((pl, p) => {
        if (p === me) return;
        const st = pl.rows[i], c = RC[st];
        const reach = c >= 5 ? 1 : c === 4 ? .55 : c === 3 ? .25 : c === 2 ? .1 : .03;
        const cost = skipCost(i, st, DIR[i] > 0 ? 12 : 2);
        prod *= 1 - PL * reach * Math.exp(-cost / 0.12);
      });
      lam[i] = Math.min(0.6, 1 - prod);
    }
    return lam;
  }

  /* ---------- option enumeration for one player's turn ---------- */
  function applyRow(rows, i, st) { const r = rows.slice(); r[i] = st; return r; }
  /* all (white mark, colour mark) combinations; each option: {w, c, rows, nm, cw, cc} */
  function enumOptions(rows, closed, w1, w2, cdice, active) {
    const S = w1 + w2, out = [];
    const wl = [{ i: -1, st: -1 }];
    for (let i = 0; i < NR; i++) if (!closed[i]) { const st = tr(i, rows[i], S); if (st >= 0) wl.push({ i, st }); }
    for (const wc of wl) {
      const r1 = wc.i >= 0 ? applyRow(rows, wc.i, wc.st) : rows;
      const cw = wc.i >= 0 ? skipCost(wc.i, rows[wc.i], S) : 0;
      out.push({ w: wc.i >= 0 ? { row: wc.i, n: S } : null, c: null, rows: r1, nm: wc.i >= 0 ? 1 : 0, cw, cc: 0 });
      if (!active || nLocks(r1) >= 2) continue;
      for (let i = 0; i < NR; i++) {
        if (closed[i]) continue;
        for (let j = 0; j < 2; j++) {
          if (j && w1 === w2) break;
          const n = (j ? w2 : w1) + cdice[i], st = tr(i, r1[i], n);
          if (st < 0) continue;
          out.push({ w: wc.i >= 0 ? { row: wc.i, n: S } : null, c: { row: i, n }, rows: applyRow(r1, i, st),
                     nm: (wc.i >= 0 ? 1 : 0) + 1, cw, cc: skipCost(i, r1[i], n) });
        }
      }
    }
    return out;
  }

  /* heuristic bot: only accepts marks that skip little probability mass */
  function heurPlan(rows, active, w1, w2, cdice, closed, th, pm) {
    const opts = enumOptions(rows, closed, w1, w2, cdice, active);
    let best = null, bestN = 0, bestTot = Infinity, cheap = null, cheapV = Infinity;
    for (const o of opts) {
      if (o.nm === 0) continue;
      const tot = o.cw + o.cc;
      const ok = (o.w === null || o.cw <= th) && (o.c === null || o.cc <= th);
      if (ok && (o.nm > bestN || (o.nm === bestN && tot < bestTot))) { best = o; bestN = o.nm; bestTot = tot; }
      if (o.nm === 1 && tot < cheapV) { cheap = o; cheapV = tot; }
    }
    if (best) return best;
    if (active && cheap && cheapV <= pm) return cheap;
    return opts[0];                                // no mark (penalty if active)
  }
  /* expert bot: maximise the DP value of the resulting state */
  function expertPlan(G, me, active, w1, w2, cdice) {
    const pl = G.players[me], lam = threats(G, me);
    const opts = enumOptions(pl.rows, G.closed, w1, w2, cdice, active);
    let best = null, bv = -Infinity;
    for (const o of opts) {
      const v = stateValue(o.rows, pl.k + (active && o.nm === 0 ? 1 : 0), lam);
      o.value = v;
      if (v > bv + 1e-9) { bv = v; best = o; }
    }
    return best;
  }

  /* ---------- RL bot: MLP estimate of P(win | afterstate), distilled from rollout labels (see rl/) ---------- */
  const RL_NF = 23;
  let RLM = null;                                  // default model for the 'rl' bot kind
  function prepareModel(m) {
    return { mean: m.mean, std: m.std, layers: m.layers.map(L => ({ W: Float64Array.from(L.W.flat()), b: Float64Array.from(L.b), nin: L.W[0].length, nout: L.W.length })) };
  }
  function setRLModel(json) { RLM = prepareModel(json); }
  async function loadRLModel(url) { setRLModel(await (await fetch(url)).json()); }
  /* features that do not depend on my option: computed once per decision */
  function rlContext(G, me) {
    const ev = [], sc = [];
    let kmax = 0, elig = 0;
    G.players.forEach((pl, p) => {
      if (p === me) return;
      ev.push(stateValue(pl.rows, pl.k, threats(G, p)));
      sc.push(rowsScore(pl.rows, pl.k));
      if (pl.k > kmax) kmax = pl.k;
    });
    ev.sort((a, b) => b - a);
    for (let i = 0; i < NR; i++) if (!G.closed[i] && G.players.some((pl, p) => p !== me && RC[pl.rows[i]] >= 5)) elig++;
    const lam = threats(G, me);
    return {
      n: G.players.length, lam, lamSum: lam[0] + lam[1] + lam[2] + lam[3], kmax, elig,
      evMax: ev[0], ev2: ev.length > 1 ? ev[1] : ev[0], evMean: ev.reduce((a, b) => a + b, 0) / ev.length,
      scMax: Math.max(...sc), scMean: sc.reduce((a, b) => a + b, 0) / sc.length,
      closed: G.closed.filter(Boolean).length,
    };
  }
  function rlFeatures(G, me, active, ctx, o, bestValue, out, off) {
    const pl = G.players[me];
    let newLocks = 0, myElig = 0;
    for (let i = 0; i < NR; i++) {
      if (G.closed[i]) continue;
      if (RL[o.rows[i]] && !RL[pl.rows[i]]) newLocks++;
      else if (!RL[o.rows[i]] && RC[o.rows[i]] >= 5) myElig++;
    }
    const score = rowsScore(o.rows, o.k), closedAfter = ctx.closed + newLocks;
    const ends = closedAfter >= 2 || o.k >= 4 ? 1 : 0, lead = score - ctx.scMax;
    const f = [
      active ? 1 : 0, ctx.n, (G.turn || 0) / ctx.n, o.value, score, o.k, myElig, closedAfter, ends,
      ctx.evMax, ctx.ev2, ctx.evMean, ctx.scMax, ctx.scMean, ctx.kmax, ctx.elig, ctx.lamSum,
      o.value - ctx.evMax, lead, ends * Math.sign(lead), o.value - bestValue, o.value - score, ctx.evMax - ctx.scMax,
    ];
    for (let j = 0; j < RL_NF; j++) out[off + j] = f[j];
  }
  /* logit of P(win) */
  function rlForward(m, x, off) {
    let h = new Float64Array(RL_NF);
    for (let j = 0; j < RL_NF; j++) h[j] = (x[off + j] - m.mean[j]) / m.std[j];
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
  /* all options with their Expert value; `best` = the Expert's choice, `chosen` = the model's (Expert if no model) */
  function rlDecide(G, me, active, w1, w2, cdice, model) {
    const pl = G.players[me], ctx = rlContext(G, me);
    const opts = enumOptions(pl.rows, G.closed, w1, w2, cdice, active);
    let best = opts[0];
    for (const o of opts) {
      o.k = pl.k + (active && o.nm === 0 ? 1 : 0);
      o.value = stateValue(o.rows, o.k, ctx.lam);
      if (o.value > best.value + 1e-9) best = o;
    }
    if (!model) return { ctx, opts, best, chosen: best };
    const x = new Float64Array(RL_NF);
    let chosen = best, bl = -Infinity;
    for (const o of opts) {
      rlFeatures(G, me, active, ctx, o, best.value, x, 0);
      o.logit = rlForward(model, x, 0);
      if (o.logit > bl) { bl = o.logit; chosen = o; }
    }
    return { ctx, opts, best, chosen };
  }
  const rlPlan = (G, me, active, w1, w2, cdice) => rlDecide(G, me, active, w1, w2, cdice, RLM).chosen;
  /* the RL network's P(win) for each choice a player faces in the UI ({key: p}, key 'row-n' or 'skip').
     stage 'white': a white mark is worth its best colour follow-up (if active); 'colour': rows already hold the white mark. */
  function rlWinProbs(G, me, stage) {
    if (!RLM || !Tbl.ready) return null;
    const pl = G.players[me], active = me === G.active, d = G.dice, ctx = rlContext(G, me);
    let opts;
    if (stage === 'white') opts = enumOptions(pl.rows, G.closed, d.w[0], d.w[1], d.c, active);
    else {
      opts = [{ w: null, c: null, rows: pl.rows, nm: pl.turnMarks }];
      for (const o of colourOptions(G, me)) opts.push({ w: null, c: o, rows: applyRow(pl.rows, o.row, tr(o.row, pl.rows[o.row], o.n)), nm: 1 });
    }
    let best = opts[0];
    for (const o of opts) {
      o.k = pl.k + (active && o.nm === 0 ? 1 : 0);
      o.value = stateValue(o.rows, o.k, ctx.lam);
      if (o.value > best.value + 1e-9) best = o;
    }
    const x = new Float64Array(RL_NF), out = {};
    for (const o of opts) {
      rlFeatures(G, me, active, ctx, o, best.value, x, 0);
      const p = 1 / (1 + Math.exp(-rlForward(RLM, x, 0)));
      const m = stage === 'white' ? o.w : o.c, key = m ? m.row + '-' + m.n : 'skip';
      if (!(key in out) || p > out[key]) out[key] = p;
    }
    return out;
  }

  /* ---------- game ---------- */
  const BOT_KINDS = {
    patient:  { label: 'Patient',  th: 0,   pm: 0 },
    careful:  { label: 'Careful',  th: .04, pm: .12 },
    balanced: { label: 'Balanced', th: .09, pm: .25 },
    greedy:   { label: 'Greedy',   th: 1,   pm: 1 },
    expert:   { label: 'Expert',   expert: true },
    rl:       { label: 'RL bot',   expert: true, rl: true },   // falls back to the Expert until the model is loaded
  };
  function mulberry32(a) {
    const f = function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    f.getState = () => a; f.setState = v => { a = v; };   // lets the UI snapshot and replay a game (undo)
    return f;
  }

  /* specs: [{name, kind:'human'|botKind}] */
  function newGame(specs, seed) {
    const rng = mulberry32(seed === undefined ? (Math.random() * 2 ** 31) | 0 : seed);
    return {
      rng,
      players: specs.map(s => ({ name: s.name, kind: s.kind, rows: [0, 0, 0, 0], marks: [[], [], [], []], k: 0, turnMarks: 0 })),
      closed: [false, false, false, false], turn: 0, over: false, dice: null, plans: [], phase: 'idle', log: [],
    };
  }
  const die = G => 1 + Math.floor(G.rng() * 6);
  function rollDice(G) {
    G.active = G.turn % G.players.length;
    G.dice = { w: [die(G), die(G)], c: [die(G), die(G), die(G), die(G)] };
    G.players.forEach(p => { p.turnMarks = 0; });
    G.plans = G.players.map((pl, p) => {
      if (pl.kind === 'human') return null;
      const act = p === G.active, d = G.dice;
      if (BOT_KINDS[pl.kind].rl && RLM && Tbl.ready) return rlPlan(G, p, act, d.w[0], d.w[1], d.c);
      if (BOT_KINDS[pl.kind].expert && Tbl.ready) return expertPlan(G, p, act, d.w[0], d.w[1], d.c);
      const kd = BOT_KINDS[pl.kind].expert ? BOT_KINDS.balanced : BOT_KINDS[pl.kind];
      return heurPlan(pl.rows, act, d.w[0], d.w[1], d.c, G.closed, kd.th, kd.pm);
    });
    G.phase = 'white';
    return G.dice;
  }
  function doMark(G, p, row, n) {
    const pl = G.players[p], st = tr(row, pl.rows[row], n);
    if (st < 0 || G.closed[row]) throw new Error('illegal mark');
    pl.rows[row] = st; pl.marks[row].push(posOf(DIR[row], n)); pl.turnMarks++;
    return { row, n, lock: !!RL[st] };
  }
  const whiteSum = G => G.dice.w[0] + G.dice.w[1];
  function whiteOptions(G, p) {
    const pl = G.players[p], S = whiteSum(G), out = [];
    for (let i = 0; i < NR; i++) if (!G.closed[i] && tr(i, pl.rows[i], S) >= 0) out.push({ row: i, n: S });
    return out;
  }
  function colourOptions(G, p) {
    const pl = G.players[p], out = [], w = G.dice.w;
    for (let i = 0; i < NR; i++) {
      if (G.closed[i]) continue;
      const seen = new Set();
      for (const wd of w) { const n = wd + G.dice.c[i]; if (!seen.has(n) && tr(i, pl.rows[i], n) >= 0) { seen.add(n); out.push({ row: i, n }); } }
    }
    return out;
  }
  /* white phase for all bots (human decides separately via doMark) */
  function botsWhite(G) {
    const ev = [];
    G.players.forEach((pl, p) => { const pn = G.plans[p]; if (pn && pn.w) ev.push({ p, ...doMark(G, p, pn.w.row, pn.w.n) }); });
    return ev;
  }
  function botColour(G) {
    const p = G.active, pn = G.plans[p];
    if (pn && pn.c && !G.closed[pn.c.row]) return [{ p, ...doMark(G, p, pn.c.row, pn.c.n) }];
    return [];
  }
  /* official rule: a row locked during the white phase closes at once (its colour die is gone) */
  function updateClosed(S) {
    const now = [];
    for (let i = 0; i < NR; i++) if (!S.closed[i] && S.players.some(pl => RL[pl.rows[i]])) { S.closed[i] = true; now.push(i); }
    return now;
  }
  /* end of turn: penalty, row closures, game end, advance */
  function finishTurn(G) {
    const ev = { penalty: false, closedNow: [], over: false };
    const a = G.players[G.active];
    if (a.turnMarks === 0) { a.k++; ev.penalty = true; }
    ev.closedNow = updateClosed(G);
    if (G.closed.filter(Boolean).length >= 2 || G.players.some(pl => pl.k >= 4)) { G.over = true; ev.over = true; }
    G.turn++; G.phase = G.over ? 'over' : 'idle';
    return ev;
  }
  const scoreOf = (G, p) => rowsScore(G.players[p].rows, G.players[p].k);
  const rowScores = (G, p) => G.players[p].rows.map(r => T(RC[r]));

  /* ---------- value labels for the human ---------- */
  /* stage 'white': value of each white option (incl. best colour follow-up) and of skipping.
     stage 'colour': value of each colour option given the white mark already made, and of skipping. */
  function optionValues(G, p, stage) {
    const pl = G.players[p], d = G.dice, lam = threats(G, p), active = p === G.active;
    if (stage === 'white') {
      const opts = enumOptions(pl.rows, G.closed, d.w[0], d.w[1], d.c, active);
      const byW = new Map(); let skip = -Infinity;
      for (const o of opts) {
        const v = stateValue(o.rows, pl.k + (active && o.nm === 0 ? 1 : 0), lam);
        if (o.w) { const key = o.w.row; byW.set(key, Math.max(byW.has(key) ? byW.get(key) : -Infinity, v)); }
        else skip = Math.max(skip, v);
      }
      return { cells: [...byW].map(([row, v]) => ({ row, n: whiteSum(G), v })), skip };
    }
    /* colour stage: current rows already include the white mark */
    const cells = [];
    for (const o of colourOptions(G, p)) {
      const st = tr(o.row, pl.rows[o.row], o.n);
      cells.push({ row: o.row, n: o.n, v: stateValue(applyRow(pl.rows, o.row, st), pl.k, lam) });
    }
    const skip = stateValue(pl.rows, pl.k + (pl.turnMarks === 0 ? 1 : 0), lam);
    return { cells, skip };
  }

  /* ---------- win probability by simulating the rest of the game ----------
     Everyone plays their real policy (bots: their kind; the human seat: the Expert policy).
     RL bots play their network (about 17x slower than the Expert, so the UI shows labels progressively). */
  function planFor(S, p, active, w1, w2, c) {
    const pl = S.players[p], kd = BOT_KINDS[pl.kind];
    if (kd && kd.rl && RLM) return rlPlan(S, p, active, w1, w2, c);
    if (pl.kind === 'human' || (kd && kd.expert)) return expertPlan(S, p, active, w1, w2, c);
    return heurPlan(pl.rows, active, w1, w2, c, S.closed, kd.th, kd.pm);
  }
  function markL(pl, row, n) { const st = tr(row, pl.rows[row], n); if (st >= 0) { pl.rows[row] = st; pl.tm++; } }
  const isOver = S => S.closed.filter(Boolean).length >= 2 || S.players.some(pl => pl.k >= 4);
  function simTurn(S, a, rng) {
    const d = () => 1 + Math.floor(rng() * 6), w1 = d(), w2 = d(), c = [d(), d(), d(), d()];
    const plans = S.players.map((pl, p) => planFor(S, p, p === a, w1, w2, c));
    S.players.forEach((pl, p) => { pl.tm = 0; if (plans[p].w) markL(pl, plans[p].w.row, plans[p].w.n); });
    updateClosed(S);
    const A = S.players[a], oc = plans[a].c;
    if (oc && !S.closed[oc.row]) markL(A, oc.row, oc.n);
    if (A.tm === 0) A.k++;
    updateClosed(S);
  }
  function bestColour(S, p, d) {          /* Expert colour choice after the white phase */
    const pl = S.players[p], lam = threats(S, p);
    let best = null, bv = stateValue(pl.rows, pl.k + (pl.tm === 0 ? 1 : 0), lam);
    for (let i = 0; i < NR; i++) {
      if (S.closed[i]) continue;
      for (let j = 0; j < 2; j++) {
        if (j && d.w[0] === d.w[1]) break;
        const n = d.w[j] + d.c[i], st = tr(i, pl.rows[i], n); if (st < 0) continue;
        const r = pl.rows.slice(); r[i] = st;
        const v = stateValue(r, pl.k, lam); if (v > bv) { bv = v; best = { row: i, n }; }
      }
    }
    return best;
  }
  /* one simulated future after player `me` takes `cand` ({row,n} or null = skip) at `stage`.
     Returns me's share of the win (1, 1/ties or 0). Same seed => same future dice (common random numbers). */
  function rollout(G, me, stage, cand, seed) {
    const S = { players: G.players.map(p => ({ kind: p.kind, rows: p.rows.slice(), k: p.k, tm: p.turnMarks })), closed: G.closed.slice() };
    const rng = mulberry32(seed), a = G.active, M = S.players[me];
    if (cand) markL(M, cand.row, cand.n);
    if (stage === 'white') {
      updateClosed(S);
      if (a === me) { const o = bestColour(S, me, G.dice); if (o) markL(M, o.row, o.n); }
      else { const pn = G.plans[a]; if (pn && pn.c && !S.closed[pn.c.row]) markL(S.players[a], pn.c.row, pn.c.n); }
    }
    if (S.players[a].tm === 0) S.players[a].k++;
    updateClosed(S);
    let t = G.turn + 1, guard = 0;
    while (!isOver(S) && guard++ < 400) { S.turn = t; simTurn(S, t % S.players.length, rng); t++; }
    const sc = S.players.map(pl => rowsScore(pl.rows, pl.k)), mx = Math.max(...sc);
    return sc[me] < mx ? 0 : 1 / sc.filter(x => x === mx).length;
  }

  return {
    NR, DIR, COLORS, BOT_KINDS, RP, RC, RL, Tbl, T, posOf, numAt, tr, skipCost, rowsScore, isTerminal,
    loadTable, loadTableFromDir, setTable, stateValue, threats, enumOptions, heurPlan, expertPlan,
    newGame, rollDice, doMark, whiteOptions, colourOptions, botsWhite, botColour, finishTurn,
    scoreOf, rowScores, optionValues, whiteSum, Fint, F, KAPPA, updateClosed, rollout, mulberry32,
    simTurn, isOver, markL, RL_NF, prepareModel, setRLModel, loadRLModel, rlContext, rlFeatures, rlForward, rlDecide, rlPlan, rlWinProbs,
  };
});
