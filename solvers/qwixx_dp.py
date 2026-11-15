"""
Exact DP for solitaire Qwixx (you are the active player every turn).

V(s) = expected final score from state s under optimal play.
State s = (rows, k):
    rows[i] = (p, c, locked)   p = last marked position 0..11 (in the row's
                               direction), c = #marks (lock bonus included)
    k       = penalties 0..4
Pure Python is fine for a reduced game (e.g. 2 rows); the full 4-row game
needs the C port (qwixx_dp.c), which implements exactly the same recursion.
"""
import sys
from functools import lru_cache

sys.setrecursionlimit(100_000)
UP, DOWN = 0, 1                      # red/yellow: 2->12, green/blue: 12->2
DIRS = (UP, UP, DOWN, DOWN)          # change to (UP, DOWN) for a quick test
NEG_INF = float("-inf")


def pos(n, d):                       # index 1..11 of number n in a row
    return n - 1 if d == UP else 13 - n


def mark(row, n, d):
    """Row state after marking number n, or None if illegal."""
    p, c, locked = row
    q = pos(n, d)
    if locked or q <= p:
        return None
    if q == 11:                      # last cell = lock, needs >= 5 marks
        return (11, c + 2, True) if c >= 5 else None   # +1 cell, +1 lock bonus
    return (q, c + 1, False)


def is_terminal(rows, k):
    return k >= 4 or sum(r[2] for r in rows) >= 2


def score(rows, k):
    return sum(c * (c + 1) // 2 for _, c, _ in rows) - 5 * k


def canon(rows):
    """Rows with the same direction are interchangeable -> sort them."""
    groups = {}
    for r, d in zip(rows, DIRS):
        groups.setdefault(d, []).append(r)
    out, used = [], {d: 0 for d in groups}
    for d in DIRS:
        g = sorted(groups[d])
        out.append(g[used[d]])
        used[d] += 1
    return tuple(out)


def V(rows, k):
    return _V(canon(rows), k)


@lru_cache(maxsize=None)
def _V(rows, k):
    if is_terminal(rows, k):
        return score(rows, k)
    total = 0.0
    for w1 in range(1, 7):
        for w2 in range(w1, 7):
            p = (1 if w1 == w2 else 2) / 36
            total += p * value_given_whites(rows, k, w1, w2)
    return total


def value_given_whites(rows, k, w1, w2):
    # Stage 1: white sum (optional) -> candidate intermediate states
    S = w1 + w2
    opts = [(rows, False)]
    for i, d in enumerate(DIRS):
        r = mark(rows[i], S, d)
        if r:
            opts.append((rows[:i] + (r,) + rows[i + 1:], True))

    # m0: best result without a colour mark (penalty if nothing marked)
    m0 = max(V(r, k) if marked else V(r, k + 1) for r, marked in opts)

    # M[i][d]: best result that uses colour die i showing d.
    # Depends only on that die -> the four M_i are independent.
    M = []
    for i, d_i in enumerate(DIRS):
        vals = []
        for die in range(1, 7):
            best = NEG_INF
            for r, _ in opts:
                if is_terminal(r, k):          # white mark ended the game
                    continue
                for w in {w1, w2}:
                    r2 = mark(r[i], w + die, d_i)
                    if r2:
                        best = max(best, V(r[:i] + (r2,) + r[i + 1:], k))
            vals.append(best)
        M.append(vals)
    return expected_max(m0, M)


def expected_max(m0, M):
    """E[max(m0, M_1..M_n)] for independent M_i, each uniform over 6 values.
    Uses P(max <= t) = 1[m0 <= t] * prod_i P(M_i <= t)."""
    def cdf(t):
        prod = 1.0
        for vals in M:
            prod *= sum(v <= t for v in vals) / 6
        return prod

    prev = cdf(m0)
    e = m0 * prev
    for t in sorted({v for vals in M for v in vals if v > m0}):
        f = cdf(t)
        e += t * (f - prev)
        prev = f
    return e


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        DIRS = (UP, DOWN)
    start = tuple((0, 0, False) for _ in DIRS)
    print(f"rows={len(DIRS)}  E[optimal score] = {V(start, 0):.6f}")
