# Qwixx: dynamic programming, simulation and a playable bot game

**Play it: https://emilphilipp.github.io/qwixx/** (first load downloads ~30 MB of value tables)

Play Qwixx against 1-5 bots (Patient, Careful, Balanced, Greedy, Expert, **RL bot**) and optionally see, for every
legal move, **your chance to win** if you pick it, according to the RL bot. Misclicked? **↶ Undo** takes back your last move (same dice).

## How the win chances are computed

The labels are the **RL bot's** estimate of your chance to win for each option: its neural network scores the
position after the move (see *RL bot* below), so the labels are instant. The network judges the position, not who
your opponents are, so its absolute numbers are rougher than its ranking: compared with 1000 simulated games per
option it is off by about 8 percentage points on average, but its favourite option costs only about 0.25 pp of win
chance versus the best one. (`Q.rollout` / `calib.js` still provide the simulation-based estimate.)

## What is in here

- **Exact DP for solitaire Qwixx** (`solvers/qwixx_dp.c`, readable version `solvers/qwixx_dp.py`):
  expected score of optimal play is **115.5**, versus about 102 for a simple "weighted skip cost" heuristic.
- **Multiplayer model** (`solvers/qwixx_stat.c`): opponents are modelled as a per-turn chance that the game ends (3.5 %).
  One stationary DP table (14.5 M states) is the value function behind the Expert bot and the option labels.
- **Row-closure threat correction** (`solvers/qwixx_thr.c`, `qwixx-engine.js`): a single-row DP with a closure hazard,
  driven by the opponents' boards. Simulation shows it raises win share against every opponent type.
- **Simulators** (`solvers/qwixx_mp.c`, `solvers/qwixx_thr.c`): 5-player games with heuristic opponents of different patience.

Mean score of the table-based player (with threat correction) in a 5-player game, four identical opponents,
30 000 games per row:

| Opponents | Player | Win share |
|---|---|---|
| Too patient | 40.6 | 99 % |
| Very cautious | 65.8 | 98 % |
| Cautious | 105.3 | 78 % |
| Balanced | 96.1 | 50 % |
| Greedy | 66.1 | 66 % |
| Mixed | 54.1 | 61 % |

(A fair share in a 5-player game is 20 %.)

## RL bot (`rl/`)

The Expert maximises expected score; the RL bot maximises its **chance to win**. A small MLP scores the position
after each option (23 features: the Expert's DP value, scores, penalties, opponents' values, closed rows,
"does this end the game while I'm ahead", …). It was trained by distilling rollouts: for 30 000 games, every
reasonable option at every decision was labelled with its win share over 128 simulated futures (common random
numbers), and the MLP was fitted with a pointwise loss plus a within-decision ranking loss.
Self-play with plain Monte-Carlo returns did not work: one win/loss per game is too noisy to rank single moves.

Paired games (same dice) against the Expert, 24 000 per pool, win share:

| Opponents | Expert | RL bot | diff (pp) |
|---|---|---|---|
| careful ×4 | 77.7 % | 79.5 % | +1.8 ± 0.3 |
| balanced ×4 | 50.3 % | 56.6 % | +6.3 ± 0.3 |
| greedy ×4 | 66.1 % | 67.8 % | +1.7 ± 0.2 |
| expert ×4 | 19.8 % | 28.6 % | +8.8 ± 0.3 |
| mixed | 61.2 % | 73.2 % | +11.9 ± 0.3 |
| random 1–5 bots | 58.6 % | 66.6 % | +8.0 ± 0.3 |
| RL bot and Expert at the same table | 33.4 % | 53.0 % | +19.6 ± 0.6 |

Why it wins (`node rl/analyze.js rl/models/d1.json`): it deviates from the Expert in 15 % of decisions, mostly
by marking more cells now (accepting skipped cells) and closing rows sooner. Its games are shorter and its mean
score is *lower* (65.6 vs 67.6), yet it wins more: it plays for tempo when ahead and, when a 4th penalty ends a
game it leads, it takes the penalty on purpose.

```bash
uv sync                                                    # Python venv (numpy, torch) for training
node rl/gen_ro.js none 2500 1 128 6 rl/data/ro1_w0.bin     # rollout-labelled data (run one per core)
uv run python rl/train.py --ro rl/models/d1.json rl/data/ro1_w*.bin
rl/peval.sh rl/models/d1.json 2000 12                      # paired evaluation vs the Expert on 12 cores
```

## Run locally

```bash
python3 -m http.server 8000      # then open http://localhost:8000
```
or with Docker:
```bash
docker build -t qwixx . && docker run -p 8080:80 qwixx    # http://localhost:8080
```
Headless bot-vs-bot test: `node test_engine.js table 2000`

## Caveats

The "optimal" player is optimal against a simple model of the opponents (game ends with a fixed chance after every turn,
plus a closure-risk estimate per row), not a true multiplayer equilibrium; the Expert bot maximises expected score, not win chance.
The win-chance labels use the real bots and table size. The C simulators close rows at the end of a turn;
the web game follows the official rule (a row locked in the white phase closes before the colour phase).
