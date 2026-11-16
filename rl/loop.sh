#!/bin/bash
# Approximate policy iteration: rl/loop.sh ROUNDS [games per round] [workers] [eps]
# Round r plays with model r-1 (round 0: the Expert), trains model r on the data of the last 3 rounds.
set -e
cd "$(dirname "$0")/.."
R=${1:-6}; GAMES=${2:-240000}; W=${3:-12}; EPS=${4:-0.05}
mkdir -p rl/data rl/models
for r in $(seq 0 $((R - 1))); do
  prev=$([ "$r" = 0 ] && echo none || echo "rl/models/m$((r - 1)).json")
  per=$((GAMES / W))
  for w in $(seq 0 $((W - 1))); do
    node rl/gen.js "$prev" "$EPS" "$per" $((r * 10000000 + w * per)) "rl/data/r${r}_w${w}.bin" &
  done
  wait
  files=$(ls rl/data/r$((r > 1 ? r - 2 : 0))_*.bin rl/data/r$((r > 0 ? r - 1 : 0))_*.bin rl/data/r${r}_*.bin | sort -u)
  echo "== round $r: train on $(echo $files | wc -w) shards"
  uv run python rl/train.py "rl/models/m${r}.json" $files
  node rl/eval.js "rl/models/m${r}.json" 4000 | tee "rl/models/m${r}.eval.txt"
done
