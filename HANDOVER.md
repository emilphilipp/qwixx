# Handover: Qwixx solver, bots and next step (RL)

Written at the end of a Claude chat session so the work can continue in Claude Code.
Repo: https://github.com/emilphilipp/qwixx. Live game (once Pages is on): https://emilphilipp.github.io/qwixx/

## Where things stand

| Piece | File | Status |
|---|---|---|
| Exact solitaire DP | `solvers/qwixx_dp.c` (+ readable `solvers/qwixx_dp.py`) | Done. Optimal E[score] = **115.48** (heuristic ≈ 102) |
| 5-player sim, hazard DP hero | `solvers/qwixx_mp.c` | Done. Exact-phase DP with fixed per-turn end hazard H |
| Stationary DP table (used by the bots/UI) | `solvers/qwixx_stat.c` → `table/` | Done. H = 0.035, 5 players, random turn phase; V(start) = 52.19; 14.5 M states; ~8 min to solve |
| Row-closure threat correction | `solvers/qwixx_thr.c`, `qwixx-engine.js` | Done. Raises win share vs every opponent pool (kappa = 1) |
| Web game vs 1–5 bots | `index.html`, `qwixx-engine.js` | Done. Labels = **win probability** per option (1000 rollouts each, common random numbers) |
| Calibration of win probabilities | `calib.js` | Done. Predicted vs actual agree within noise (Brier 0.135) |
| **RL bot to beat the Expert** | – | **Not started** (plan below) |

## Model, in short

- Row state = (p = last marked position 0..11, c = marks incl. lock bonus, locked). 62 row states; rows with the same
  direction are interchangeable (sorted) → 4-row state space ≈ 62⁴·4 / 4.
- Turn: white sum (all players) → colour mark (active only) → penalty if active marked nothing.
  Lock needs ≥ 5 marks; lock cell counts double. Game ends at 2 closed rows or 4 penalties.
- Fast expectation trick: given whites, the 4 colour dice act independently, so
  E[max] uses P(max ≤ t) = Π_i P(M_i ≤ t).
- Multiplayer is approximated: opponents = per-turn end hazard H (+ per-row closure hazard λ_i estimated from
  opponents' boards, single-row DP f(p,c;λ), correction = κ·(f(λ) − f(0))).
- Table file format (`table/`): `meta.txt` = `min scale H npl NP2 4`; `table_k{0..3}.bin` = uint16,
  index `(pRY * NP2 + pGB)` with `pRY = b(b+1)/2 + a` for sorted red/yellow states a ≤ b (same for green/blue);
  value = min + scale · u16. (The Claude-hosted copy uses base64 JSON instead of .bin.)

## Engine API (`qwixx-engine.js`, works in node and browser)

- `loadTableFromDir(dir)` (node) / `loadTable(url)` (browser)
- `newGame(specs, seed)`, `rollDice`, `botsWhite`, `updateClosed`, `botColour`, `finishTurn`, `scoreOf`
- Bot kinds: `patient, careful, balanced, greedy` (weighted-skip heuristic with different thresholds), `expert`
  (maximises `stateValue` = DP table + threat correction, i.e. **expected score**, not win chance)
- `rollout(G, me, stage, cand, seed)` → my win share for one simulated future (used for the UI labels)
- Headless test: `node test_engine.js table 2000`; calibration: `node calib.js table 800`
  (JS sim ≈ 6000 full games/s, ≈ 14 000 rollouts/s on one core)

## Key results (5-player, 30 000 games/row, opponents = 4 identical bots)

| Opponents | Heuristic hero | Table only | Table + threat (Expert) | Expert win share |
|---|---|---|---|---|
| Too patient | 29.4 | 40.5 | 40.6 | 99 % |
| Very cautious | 53.8 | 65.6 | 65.8 | 98 % |
| Cautious | 107.4 | 105.6 | 105.3 | 78 % |
| Balanced | 90.6 | 95.1 | 96.1 | 50 % |
| Greedy | 48.2 | 61.9 | 66.1 | 66 % |
| Mixed | 39.9 | 52.0 | 54.1 | 61 % |

Fair share = 20 %. Solitaire-optimal patience is badly wrong in multiplayer (scored 51 vs 91 against balanced bots).

## Next step: RL bot that maximises win probability

Idea: the Expert maximises expected score; winning is what matters (short games are won with low scores).
Approximate policy iteration with Monte-Carlo returns on afterstates:

1. Afterstate features (after my move): my DP expected final score (`stateValue`), my current score,
   my penalties; for opponents: max / second max / mean of their DP expected score and current score,
   max opponent penalties; # closed rows; turn index; number of players; my lock-eligible rows; differences
   (me − best opponent).
2. Model: P(win | afterstate) via logistic regression on those features + interactions (scikit-learn is available;
   no torch in the cloud sandbox, so a small MLP via sklearn or numpy). Use `stateValue` as a feature so the
   model learns a residual on top of the Expert.
3. Loop: play games where the RL seat acts ε-greedy on the current model (other seats: mixed pool of the five kinds,
   1–5 bots) → record afterstates of every decision + final win share → refit → repeat 5–10 rounds.
   Start round 0 from the Expert's policy to get data.
4. Evaluate paired (same seeds/dice) vs the Expert across the opponent pools above, plus head-to-head games
   containing both an Expert and the RL bot. Report win share with standard errors.
5. Cheap strong baseline to beat: "rollout bot" = pick the option with the highest `rollout` win share
   (one-step policy improvement over the Expert). It is slow per move but shows the achievable ceiling.

## Caveats / open items

- C simulators close rows at end of turn; the web engine follows the official rule (a row locked in the white
  phase closes before the colour phase). Small effect on the C results.
- The win-chance labels assume you play the Expert line after the chosen move.
- Stray branch `main` in `emilphilipp.github.io` (default branch is `master`) can be deleted.
- Portfolio Projects page already links to the game and the repo.
- Testing against playqwixx.onrender.com did not work (free Render host kept restarting, tab unresponsive);
  only use Claude in Chrome with Emil's explicit approval.

## Rebuild the table locally (macOS)

```bash
clang -O3 -o qwixx_stat solvers/qwixx_stat.c -lm
mkdir -p table && ./qwixx_stat 0.035 5 table      # ~8 min, ~1.2 GB RAM
python3 -m http.server 8000                        # play at http://localhost:8000
```
