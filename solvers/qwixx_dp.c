/* Fast port of qwixx_dp.py + Monte Carlo comparison of policies.
 * usage: ./qwixx_dp NROWS(2|4) P(1=solitaire, 2=every 2nd turn passive) NSIM
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define NRS 62
#define MAXR 4
static int NR, P;
static int DIR[MAXR];
static int RP[NRS], RC[NRS], RL[NRS];
static int TR[2][NRS][13];
static double *memo;

static int pos_of(int n, int d) { return d == 0 ? n - 1 : 13 - n; }

static void build_rows(void) {
    int id = 0;
    RP[id] = 0; RC[id] = 0; RL[id] = 0; id++;
    for (int p = 1; p <= 10; p++) for (int c = 1; c <= p; c++) { RP[id] = p; RC[id] = c; RL[id] = 0; id++; }
    for (int c = 7; c <= 12; c++) { RP[id] = 11; RC[id] = c; RL[id] = 1; id++; }
    if (id != NRS) { fprintf(stderr, "bad row count %d\n", id); exit(1); }
    for (int d = 0; d < 2; d++) for (int r = 0; r < NRS; r++) for (int n = 0; n < 13; n++) {
        TR[d][r][n] = -1;
        if (n < 2 || RL[r]) continue;
        int q = pos_of(n, d);
        if (q <= RP[r]) continue;
        int np, nc, nl;
        if (q == 11) { if (RC[r] < 5) continue; np = 11; nc = RC[r] + 2; nl = 1; }
        else { np = q; nc = RC[r] + 1; nl = 0; }
        for (int s = 0; s < NRS; s++) if (RP[s] == np && RC[s] == nc && RL[s] == nl) { TR[d][r][n] = s; break; }
    }
}

static void canon(int *r) {
    for (int i = 0; i < NR; i++) for (int j = i + 1; j < NR; j++)
        if (DIR[i] == DIR[j] && r[i] > r[j]) { int t = r[i]; r[i] = r[j]; r[j] = t; }
}
static int nlocks(const int *r) { int n = 0; for (int i = 0; i < NR; i++) n += RL[r[i]]; return n; }
static int terminal(const int *r, int k) { return k >= 4 || nlocks(r) >= 2; }
static double score(const int *r, int k) {
    double s = -5.0 * k;
    for (int i = 0; i < NR; i++) s += RC[r[i]] * (RC[r[i]] + 1) / 2;
    return s;
}
static long idx(const int *r, int k, int ph) {
    long x = 0; for (int i = 0; i < NR; i++) x = x * NRS + r[i];
    return (x * 4 + k) * P + ph;
}

static double V(const int *r0, int k, int ph);

static double cdf(double M[][6], double t) {
    double p = 1;
    for (int i = 0; i < NR; i++) { int c = 0; for (int d = 0; d < 6; d++) if (M[i][d] <= t) c++; p *= c / 6.0; }
    return p;
}
static int cmpd(const void *a, const void *b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
static double expmax(double m0, double M[][6]) {
    double ts[24]; int nt = 0;
    for (int i = 0; i < NR; i++) for (int d = 0; d < 6; d++) if (M[i][d] > m0) ts[nt++] = M[i][d];
    qsort(ts, nt, sizeof(double), cmpd);
    double prev = cdf(M, m0), e = m0 * prev;
    for (int q = 0; q < nt; q++) {
        if (q > 0 && ts[q] == ts[q - 1]) continue;
        double f = cdf(M, ts[q]); e += ts[q] * (f - prev); prev = f;
    }
    return e;
}

static int white_opts(const int *r, int S, int opt[5][MAXR], int *mk) {
    int no = 1; memcpy(opt[0], r, sizeof(int) * NR); mk[0] = 0;
    for (int i = 0; i < NR; i++) {
        int n = TR[DIR[i]][r[i]][S]; if (n < 0) continue;
        memcpy(opt[no], r, sizeof(int) * NR); opt[no][i] = n; mk[no] = 1; no++;
    }
    return no;
}

static double active_w(const int *r, int k, int w1, int w2, int nph) {
    int opt[5][MAXR], mk[5], t[MAXR];
    int no = white_opts(r, w1 + w2, opt, mk);
    double m0 = -INFINITY;
    for (int a = 0; a < no; a++) { double v = mk[a] ? V(opt[a], k, nph) : V(opt[a], k + 1, nph); if (v > m0) m0 = v; }
    double M[MAXR][6];
    for (int i = 0; i < NR; i++) for (int d = 1; d <= 6; d++) {
        double best = -INFINITY;
        for (int a = 0; a < no; a++) {
            if (nlocks(opt[a]) >= 2) continue;
            for (int j = 0; j < 2; j++) {
                int w = j ? w2 : w1; if (j && w2 == w1) break;
                int n = TR[DIR[i]][opt[a][i]][w + d]; if (n < 0) continue;
                memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
                double v = V(t, k, nph); if (v > best) best = v;
            }
        }
        M[i][d - 1] = best;
    }
    return expmax(m0, M);
}

static double passive_w(const int *r, int k, int S, int nph) {
    int opt[5][MAXR], mk[5];
    int no = white_opts(r, S, opt, mk);
    double best = -INFINITY;
    for (int a = 0; a < no; a++) { double v = V(opt[a], k, nph); if (v > best) best = v; }
    return best;   /* passive players never take a penalty */
}

static double V(const int *r0, int k, int ph) {
    int r[MAXR]; memcpy(r, r0, sizeof(int) * NR); canon(r);
    if (terminal(r, k)) return score(r, k);
    long id = idx(r, k, ph);
    if (!isnan(memo[id])) return memo[id];
    int nph = (ph + 1) % P;
    double tot = 0;
    for (int w1 = 1; w1 <= 6; w1++) for (int w2 = w1; w2 <= 6; w2++) {
        double pw = (w1 == w2 ? 1.0 : 2.0) / 36.0;
        tot += pw * (ph == 0 ? active_w(r, k, w1, w2, nph) : passive_w(r, k, w1 + w2, nph));
    }
    memo[id] = tot;
    return tot;
}

/* ---------------- simulation ---------------- */
static uint64_t rs = 88172645463325252ULL;
static int die(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return 1 + (int)(rs % 6); }
static double PR(int n) { return (6 - abs(n - 7)) / 36.0; }
static double skipcost(int row, int d, int n) {   /* prob-weighted skipped cells */
    double c = 0; int q = pos_of(n, d);
    for (int x = RP[row] + 1; x < q; x++) c += PR(d == 0 ? x + 1 : 13 - x);
    return c;
}
static double THETA = 0.09, PEN_MAX = 0.25;

/* plays one turn; policy 0 = optimal (via V), 1 = weighted-skip heuristic */
static void turn(int *r, int *k, int ph, int policy) {
    int w1 = die(), w2 = die(), c[MAXR]; for (int i = 0; i < NR; i++) c[i] = die();
    int nph = (ph + 1) % P, active = (ph == 0);
    int opt[5][MAXR], mk[5], t[MAXR], best[MAXR], bestk = *k;
    int no = white_opts(r, w1 + w2, opt, mk);
    double bv = -INFINITY;
    /* heuristic bookkeeping */
    double hbest = INFINITY; int hmarks = 0; double cheap = INFINITY; int cheapst[MAXR];
    int wrow[5]; wrow[0] = -1; { int q = 1; for (int i = 0; i < NR; i++) if (TR[DIR[i]][r[i]][w1 + w2] >= 0) wrow[q++] = i; }

    for (int a = 0; a < no; a++) {
        double cw = mk[a] ? skipcost(r[wrow[a]], DIR[wrow[a]], w1 + w2) : 0;
        /* option: no colour mark */
        int kk = (!mk[a] && active) ? *k + 1 : *k;
        if (policy == 0) {
            double v = V(opt[a], kk, nph);
            if (v > bv) { bv = v; memcpy(best, opt[a], sizeof(int) * NR); bestk = kk; }
        } else if (mk[a]) {
            if (cw <= THETA && (1 > hmarks || (1 == hmarks && cw < hbest))) { hmarks = 1; hbest = cw; memcpy(best, opt[a], sizeof(int) * NR); bestk = *k; }
            if (cw < cheap) { cheap = cw; memcpy(cheapst, opt[a], sizeof(int) * NR); }
        }
        if (!active || nlocks(opt[a]) >= 2) continue;
        for (int i = 0; i < NR; i++) for (int j = 0; j < 2; j++) {
            int w = j ? w2 : w1;
            int n = TR[DIR[i]][opt[a][i]][w + c[i]]; if (n < 0) continue;
            memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
            if (policy == 0) {
                double v = V(t, *k, nph);
                if (v > bv) { bv = v; memcpy(best, t, sizeof(int) * NR); bestk = *k; }
            } else {
                double cc = skipcost(opt[a][i], DIR[i], w + c[i]);
                int nm = mk[a] + 1; double tot = cw + cc;
                int ok = cc <= THETA && (!mk[a] || cw <= THETA);
                if (ok && (nm > hmarks || (nm == hmarks && tot < hbest))) { hmarks = nm; hbest = tot; memcpy(best, t, sizeof(int) * NR); bestk = *k; }
                if (nm == 1 && cc < cheap) { cheap = cc; memcpy(cheapst, t, sizeof(int) * NR); }
            }
        }
    }
    if (policy == 1 && hmarks == 0) {
        if (active && cheap <= PEN_MAX) { memcpy(best, cheapst, sizeof(int) * NR); bestk = *k; }
        else { memcpy(best, r, sizeof(int) * NR); bestk = active ? *k + 1 : *k; }
    }
    memcpy(r, best, sizeof(int) * NR); *k = bestk;
}

static void simulate(int policy, int nsim) {
    double s = 0, s2 = 0, turns = 0, pens = 0, locks = 0;
    for (int g = 0; g < nsim; g++) {
        int r[MAXR] = {0}, k = 0, ph = 0, nt = 0;
        while (!terminal(r, k)) { turn(r, &k, ph, policy); ph = (ph + 1) % P; nt++; }
        double sc = score(r, k); s += sc; s2 += sc * sc; turns += nt; pens += k; locks += nlocks(r);
    }
    double m = s / nsim;
    printf("  %-10s mean %.2f  (sd %.1f)  turns %.1f  penalties %.2f  locks %.2f\n",
           policy ? "heuristic" : "optimal", m, sqrt(s2 / nsim - m * m), turns / nsim, pens / nsim, locks / nsim);
}

int main(int argc, char **argv) {
    NR = argc > 1 ? atoi(argv[1]) : 4; P = argc > 2 ? atoi(argv[2]) : 1;
    int nsim = argc > 3 ? atoi(argv[3]) : 0;
    if (NR == 4) { DIR[0] = 0; DIR[1] = 0; DIR[2] = 1; DIR[3] = 1; } else { DIR[0] = 0; DIR[1] = 1; }
    build_rows();
    long n = 1; for (int i = 0; i < NR; i++) n *= NRS; n *= 4 * P;
    memo = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) memo[i] = NAN;
    clock_t t0 = clock();
    int r0[MAXR] = {0};
    double v = V(r0, 0, 0);
    long filled = 0; for (long i = 0; i < n; i++) filled += !isnan(memo[i]);
    printf("rows=%d P=%d  E[optimal score] = %.6f   (%ld states, %.1fs)\n", NR, P, v, filled, (double)(clock() - t0) / CLOCKS_PER_SEC);
    if (nsim) { simulate(0, nsim); simulate(1, nsim); }
    return 0;
}
