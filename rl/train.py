"""Fit P(win | afterstate) with a small MLP and export it as JSON for rl/common.js.

uv run python rl/train.py OUT.json DATA.bin [DATA.bin ...]          (self-play MC data, gen.js)
uv run python rl/train.py --ro OUT.json DATA.bin [DATA.bin ...]     (rollout-labelled options, gen_ro.js)
"""
import json
import sys

import numpy as np
import torch

NF = 23
HIDDEN = (64, 64)
RANK_W = 20.0      # weight of the within-decision loss (rollout data)


def load(files, ro):
    if not ro:
        a = np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, NF + 2) for f in files])
        gid = a[:, NF + 1].astype(np.int64)
        return a[:, :NF], a[:, NF], gid, gid
    parts = [np.fromfile(f, dtype=np.float32).reshape(-1, NF + 3) for f in files]
    gid = np.concatenate([i * 1_000_000 + a[:, NF + 1].astype(np.int64) for i, a in enumerate(parts)])
    a = np.concatenate(parts)
    return a[:, :NF], a[:, NF], gid, gid * 1000 + a[:, NF + 2].astype(np.int64)


def regret(score, y, dec, gap):
    """mean rollout win share lost per decision: model pick vs best option, and the Expert's pick (evGap == 0)"""
    order = np.lexsort((-score, dec))
    first = np.r_[True, dec[order][1:] != dec[order][:-1]]
    best = np.maximum.reduceat(y[order], np.flatnonzero(first))
    pick = y[order][first]
    ex_order = np.lexsort((-(gap == 0).astype(np.int8), dec))
    ex = y[ex_order][np.r_[True, dec[ex_order][1:] != dec[ex_order][:-1]]]
    return (best - pick).mean(), (best - ex).mean(), (pick - ex).mean()


def main(out, files, ro=False, epochs=6, seed=0):
    torch.manual_seed(seed)
    x, y, gid, dec = load(files, ro)
    val = (gid * 2654435761 % 10) == 0          # hold out 10 % of games (grouped split)
    mean, std = x[~val].mean(0), x[~val].std(0) + 1e-6
    xs = torch.tensor((x - mean) / std)
    yt = torch.tensor(y)
    dec_t = torch.from_numpy(dec)
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)

    dims = (NF, *HIDDEN, 1)
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = torch.nn.BCEWithLogitsLoss()

    def evaluate(idx):
        with torch.no_grad():
            p = torch.sigmoid(net(xs[idx]).squeeze(1))
            t = yt[idx]
            ll = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), t).item()
            return ll, ((p - t) ** 2).mean().item()

    base = y[tr_idx].mean()
    base_ll = -np.mean(y[va_idx] * np.log(base) + (1 - y[va_idx]) * np.log(1 - base))
    print(f"rows {len(y)}  train {len(tr_idx)}  val {len(va_idx)}  base logloss {base_ll:.4f}", flush=True)
    bs = 8192
    for ep in range(epochs):
        net.train()
        rng = np.random.default_rng(seed + ep)
        if ro:   # shuffle whole decisions so a batch holds all options of each decision
            _, inv = np.unique(dec[tr_idx], return_inverse=True)
            perm = torch.from_numpy(tr_idx[np.argsort(rng.random(inv.max() + 1)[inv], kind="stable")])
        else:
            perm = torch.from_numpy(rng.permutation(tr_idx))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logit = net(xs[b]).squeeze(1)
            loss = lossf(logit, yt[b])
            if ro:   # within-decision term: centred win prob vs centred rollout share
                _, g = torch.unique(dec_t[b], return_inverse=True)
                cnt = torch.zeros(int(g.max()) + 1).index_add_(0, g, torch.ones(len(b)))
                p = torch.sigmoid(logit)
                dp = p - (torch.zeros_like(cnt).index_add_(0, g, p) / cnt)[g]
                dy = yt[b] - (torch.zeros_like(cnt).index_add_(0, g, yt[b]) / cnt)[g]
                loss = loss + RANK_W * ((dp - dy) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()
        ll, br = evaluate(va_idx)
        msg = f"epoch {ep}  val logloss {ll:.4f}  brier {br:.4f}"
        if ro:
            with torch.no_grad():
                sc = net(xs[va_idx]).squeeze(1).numpy()
            rm, rx, gain = regret(sc, y[va_idx], dec[va_idx], x[va_idx, 20])
            msg += f"  regret model {100*rm:.2f}pp  expert {100*rx:.2f}pp  model-expert {100*gain:+.2f}pp/decision"
        print(msg, flush=True)

    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    model = {
        "mean": mean.tolist(), "std": std.tolist(),
        "layers": [{"W": m.weight.detach().double().tolist(), "b": m.bias.detach().double().tolist()} for m in lin],
        "val_logloss": ll, "val_brier": br,
    }
    with open(out, "w") as f:
        json.dump(model, f)


if __name__ == "__main__":
    args = sys.argv[1:]
    ro = args[0] == "--ro"
    if ro:
        args = args[1:]
    main(args[0], args[1:], ro=ro, epochs=12 if ro else 6)
