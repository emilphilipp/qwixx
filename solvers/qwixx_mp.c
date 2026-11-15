/* 5-player Qwixx simulation.
 * Hero = DP policy that assumes the game ends after every turn with prob H
 * (opponent-driven end-of-game hazard), or the weighted-skip heuristic (H<0).
 * Opponents = weighted-skip heuristic players with different (theta, penmax).
 * usage: ./qwixx_mp H NSIM
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define NRS 62
#define MAXR 4
#define NR 4
#define P 5
static const int DIR[MAXR] = {0, 0, 1, 1};
static int RP[NRS], RC[NRS], RL[NRS];
static int TR[2][NRS][14];
static float *memo;
static double H = 0;

static int pos_of(int n, int d) { return d == 0 ? n - 1 : 13 - n; }

static void build_rows(void) {
    int id = 0;
    RP[id] = 0; RC[id] = 0; RL[id] = 0; id++;
    for (int p = 1; p <= 10; p++) for (int c = 1; c <= p; c++) { RP[id] = p; RC[id] = c; RL[id] = 0; id++; }
    for (int c = 7; c <= 12; c++) { RP[id] = 11; RC[id] = c; RL[id] = 1; id++; }
    for (int d = 0; d < 2; d++) for (int r = 0; r < NRS; r++) for (int n = 0; n < 14; n++) {
        TR[d][r][n] = -1;
        if (n < 2 || n > 12 || RL[r]) continue;
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

/* value of a state reached after my turn, given a per-turn hazard H that the
   game ends (opponent-driven) right afterwards */
static double W(const int *r, int k, int ph) {
    if (terminal(r, k)) return score(r, k);
    double v = V(r, k, ph);
    return H == 0 ? v : (1 - H) * v + H * score(r, k);
}

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
    for (int a = 0; a < no; a++) { double v = mk[a] ? W(opt[a], k, nph) : W(opt[a], k + 1, nph); if (v > m0) m0 = v; }
    double M[MAXR][6];
    for (int i = 0; i < NR; i++) for (int d = 1; d <= 6; d++) {
        double best = -INFINITY;
        for (int a = 0; a < no; a++) {
            if (nlocks(opt[a]) >= 2) continue;
            for (int j = 0; j < 2; j++) {
                int w = j ? w2 : w1; if (j && w2 == w1) break;
                int n = TR[DIR[i]][opt[a][i]][w + d]; if (n < 0) continue;
                memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
                double v = W(t, k, nph); if (v > best) best = v;
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
    for (int a = 0; a < no; a++) { double v = W(opt[a], k, nph); if (v > best) best = v; }
    return best;
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
    memo[id] = (float)tot;
    return tot;
}

/* ------------------------------------------------------------------ */
/* simulation                                                          */
static uint64_t rs;
static uint64_t splitmix(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static int die(void) { return 1 + (int)(splitmix(&rs) % 6); }
static double PRB(int n) { return (6 - abs(n - 7)) / 36.0; }
static double skipcost(int row, int d, int n) {
    double c = 0; int q = pos_of(n, d);
    for (int x = RP[row] + 1; x < q; x++) c += PRB(d == 0 ? x + 1 : 13 - x);
    return c;
}
static inline int trw(const int *closed, int i, int row, int n) { return closed[i] ? -1 : TR[DIR[i]][row][n]; }

static void hero_turn(int *r, int *k, int active, int ph, int w1, int w2, const int *c, const int *closed) {
    int S = w1 + w2, nph = (ph + 1) % P;
    int opt[5][MAXR], mk[5], t[MAXR], best[MAXR], bestk = *k, no = 1;
    memcpy(opt[0], r, sizeof(int) * NR); mk[0] = 0;
    for (int i = 0; i < NR; i++) {
        int n = trw(closed, i, r[i], S); if (n < 0) continue;
        memcpy(opt[no], r, sizeof(int) * NR); opt[no][i] = n; mk[no] = 1; no++;
    }
    double bv = -INFINITY;
    for (int a = 0; a < no; a++) {
        int kk = (!mk[a] && active) ? *k + 1 : *k;
        double v = W(opt[a], kk, nph);
        if (v > bv) { bv = v; memcpy(best, opt[a], sizeof(int) * NR); bestk = kk; }
        if (!active || nlocks(opt[a]) >= 2) continue;
        for (int i = 0; i < NR; i++) for (int j = 0; j < 2; j++) {
            int w = j ? w2 : w1; if (j && w2 == w1) break;
            int n = trw(closed, i, opt[a][i], w + c[i]); if (n < 0) continue;
            memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
            v = W(t, *k, nph);
            if (v > bv) { bv = v; memcpy(best, t, sizeof(int) * NR); bestk = *k; }
        }
    }
    memcpy(r, best, sizeof(int) * NR); *k = bestk;
}

static void heur_turn(int *r, int *k, int active, int w1, int w2, const int *c, const int *closed,
                      double theta, double penmax) {
    int S = w1 + w2;
    int opt[5][MAXR], mk[5], wr[5], t[MAXR], no = 1;
    double cw[5];
    memcpy(opt[0], r, sizeof(int) * NR); mk[0] = 0; cw[0] = 0; wr[0] = -1;
    for (int i = 0; i < NR; i++) {
        int n = trw(closed, i, r[i], S); if (n < 0) continue;
        memcpy(opt[no], r, sizeof(int) * NR); opt[no][i] = n; mk[no] = 1; wr[no] = i;
        cw[no] = skipcost(r[i], DIR[i], S); no++;
    }
    int bestN = 0, best[MAXR], cheapst[MAXR]; double bestTot = INFINITY, cheap = INFINITY;
    for (int a = 0; a < no; a++) {
        if (mk[a]) {
            if (cw[a] <= theta && (1 > bestN || (1 == bestN && cw[a] < bestTot))) { bestN = 1; bestTot = cw[a]; memcpy(best, opt[a], sizeof(int) * NR); }
            if (cw[a] < cheap) { cheap = cw[a]; memcpy(cheapst, opt[a], sizeof(int) * NR); }
        }
        if (!active || nlocks(opt[a]) >= 2) continue;
        for (int i = 0; i < NR; i++) for (int j = 0; j < 2; j++) {
            int w = j ? w2 : w1; if (j && w2 == w1) break;
            int n = trw(closed, i, opt[a][i], w + c[i]); if (n < 0) continue;
            memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
            double cc = skipcost(opt[a][i], DIR[i], w + c[i]);
            int nm = mk[a] + 1; double tot = cw[a] + cc;
            int ok = cc <= theta && (!mk[a] || cw[a] <= theta);
            if (ok && (nm > bestN || (nm == bestN && tot < bestTot))) { bestN = nm; bestTot = tot; memcpy(best, t, sizeof(int) * NR); }
            if (nm == 1 && cc < cheap) { cheap = cc; memcpy(cheapst, t, sizeof(int) * NR); }
        }
    }
    if (bestN > 0) { memcpy(r, best, sizeof(int) * NR); return; }
    if (active && cheap <= penmax) { memcpy(r, cheapst, sizeof(int) * NR); return; }
    if (active) (*k)++;
}

typedef struct { const char *name; double th[4], pm[4]; } Scenario;
static Scenario SC[] = {
    {"patient x4  ", {0, 0, 0, 0},       {0, 0, 0, 0}},
    {"v.cautious x4",{.02, .02, .02, .02}, {.05, .05, .05, .05}},
    {"cautious x4 ", {.04, .04, .04, .04}, {.12, .12, .12, .12}},
    {"balanced x4 ", {.09, .09, .09, .09}, {.25, .25, .25, .25}},
    {"greedy x4   ", {1, 1, 1, 1},       {1, 1, 1, 1}},
    {"mixed       ", {0, .04, .09, 1},   {0, .12, .25, 1}},
};
#define NSC (int)(sizeof(SC) / sizeof(SC[0]))

int main(int argc, char **argv) {
    double h = argc > 1 ? atof(argv[1]) : -1;
    int nsim = argc > 2 ? atoi(argv[2]) : 20000;
    int hero_dp = h >= 0; H = hero_dp ? h : 0;
    build_rows();
    if (hero_dp) {
        long n = 1; for (int i = 0; i < NR; i++) n *= NRS; n *= 4 * P;
        memo = malloc(sizeof(float) * n);
        for (long i = 0; i < n; i++) memo[i] = NAN;
        clock_t t0 = clock(); int r0[MAXR] = {0};
        double v = V(r0, 0, 0);
        printf("# H=%.3f  V(start, my turn active) = %.3f  (DP %.0fs)\n", H, v, (double)(clock() - t0) / CLOCKS_PER_SEC);
        fflush(stdout);
    }
    printf("# hero=%s H=%.3f  nsim=%d\n", hero_dp ? "DP" : "heuristic", H, nsim);
    printf("# scenario      hero_mean  se   opp_mean  hero_win_share  turns  hero_pen\n");
    for (int s = 0; s < NSC; s++) {
        double sh = 0, sh2 = 0, so = 0, win = 0, turns = 0, pen = 0;
        for (int g = 0; g < nsim; g++) {
            rs = 1234567ULL + 7919ULL * g;
            int seat = (int)(splitmix(&rs) % 5);
            int R[5][MAXR] = {{0}}, K[5] = {0}, closed[MAXR] = {0}, t;
            /* opponents in seats != hero, in scenario order */
            int oidx[5], q = 0; for (int p = 0; p < 5; p++) oidx[p] = (p == seat) ? -1 : q++;
            for (t = 0;; t++) {
                int a = t % 5, w1 = die(), w2 = die(), c[MAXR];
                for (int i = 0; i < NR; i++) c[i] = die();
                for (int p = 0; p < 5; p++) {
                    int active = (p == a);
                    if (p == seat) {
                        if (hero_dp) hero_turn(R[p], &K[p], active, ((t - seat) % 5 + 5) % 5, w1, w2, c, closed);
                        else heur_turn(R[p], &K[p], active, w1, w2, c, closed, .09, .25);
                    } else {
                        int o = oidx[p];
                        heur_turn(R[p], &K[p], active, w1, w2, c, closed, SC[s].th[o], SC[s].pm[o]);
                    }
                }
                for (int i = 0; i < NR; i++) for (int p = 0; p < 5; p++) if (RL[R[p][i]]) closed[i] = 1;
                int nc = 0, over = 0; for (int i = 0; i < NR; i++) nc += closed[i];
                for (int p = 0; p < 5; p++) if (K[p] >= 4) over = 1;
                if (nc >= 2 || over) break;
            }
            double sc[5]; for (int p = 0; p < 5; p++) sc[p] = score(R[p], K[p]);
            double hs = sc[seat], opps = 0; int ties = 0, beaten = 1;
            for (int p = 0; p < 5; p++) if (p != seat) { opps += sc[p]; if (sc[p] > hs) beaten = 0; else if (sc[p] == hs) ties++; }
            if (beaten) win += 1.0 / (1 + ties);
            sh += hs; sh2 += hs * hs; so += opps / 4; turns += t + 1; pen += K[seat];
        }
        double m = sh / nsim, sd = sqrt(sh2 / nsim - m * m);
        printf("%s  %7.2f  %.2f  %7.2f   %.3f   %5.1f  %.2f\n", SC[s].name, m, sd / sqrt(nsim), so / nsim, win / nsim, turns / nsim, pen / nsim);
        fflush(stdout);
    }
    return 0;
}
