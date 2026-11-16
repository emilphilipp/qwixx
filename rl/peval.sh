#!/bin/bash
# Parallel eval.js: rl/peval.sh HERO GAMES_PER_WORKER [workers] [pools]; combines equal-size worker results.
cd "$(dirname "$0")/.."
HERO=$1; N=$2; W=${3:-12}; POOLS=${4:-}
tmp=$(mktemp -d)
for w in $(seq 0 $((W - 1))); do node rl/eval.js "$HERO" "$N" $((20000000 + w * 1000000)) "$POOLS" > "$tmp/$w.txt" & done
wait
echo "hero $HERO, $((N * W)) paired games per pool"
cat "$tmp"/*.txt | LC_ALL=C awk '
  /^pool/ {next}
  { n=NF; name=$1; for(i=2;i<=n-5;i++) name=name" "$i
    k[name]++; ex[name]+=$(n-4); rl[name]+=$(n-3); d[name]+=$(n-2); se[name]+=$n*$n; if(!(name in ord)){ord[name]=++c; nm[c]=name} }
  END { printf "%-13s %8s %8s %10s\n","pool","Expert%","hero%","diff pp";
        for(i=1;i<=c;i++){x=nm[i]; printf "%-13s %8.1f %8.1f %+7.2f ± %.2f\n", x, ex[x]/k[x], rl[x]/k[x], d[x]/k[x], sqrt(se[x])/k[x]} }'
rm -r "$tmp"
