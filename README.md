# Qwixx: dynamic programming, simulation and a playable bot game

**Play it: https://emilphilipp.github.io/qwixx/** (first load downloads ~30 MB of value tables)

Play Qwixx against 1-5 bots (Patient, Careful, Balanced, Greedy, Expert) and optionally see, for every
legal move, how many expected final points it gains or loses versus the best option.

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
plus a closure-risk estimate per row), not a true multiplayer equilibrium. The value labels assume a 5-player game
and are approximate for other table sizes. Rules simplification: row closures and the end-of-game check happen at the end of a turn.
