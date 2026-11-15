/* 5-player sim: hero = stationary DP table (+ optional per-row closure-threat correction).
 * correction_i = kappa * ( f(r_i; lambda_i) - f(r_i; 0) ), f = single-row DP with row-closure hazard.
 * usage: ./qwixx_thr TABLEDIR NSIM
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NRS 62
#define MAXR 4
#define NR 4
static const int DIR[MAXR] = {0, 0, 1, 1};
static int RP[NRS], RC[NRS], RL[NRS];
static int TR[2][NRS][14];
static double H = 0.035, ACT = 0.2;

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

/* ---------------- table ---------------- */
static uint16_t *TB[4]; static double TMIN, TSCALE; static int NP2;
static void load_table(const char *dir) {
    char fn[512]; snprintf(fn, sizeof fn, "%s/meta.txt", dir);
    FILE *m = fopen(fn, "r"); if (!m) { perror("meta"); exit(1); }
    int npl, nk; double h;
    if (fscanf(m, "%lf %lf %lf %d %d %d", &TMIN, &TSCALE, &h, &npl, &NP2, &nk) != 6) exit(1);
    fclose(m); H = h; ACT = 1.0 / npl;
    for (int k = 0; k < 4; k++) {
        snprintf(fn, sizeof fn, "%s/table_k%d.bin", dir, k);
        FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
        TB[k] = malloc(2L * NP2 * NP2);
        if (fread(TB[k], 2, (long)NP2 * NP2, f) != (size_t)NP2 * NP2) exit(1);
        fclose(f);
    }
}
static double Vtab(const int *r0, int k) {
    int r[MAXR]; memcpy(r, r0, sizeof(int) * NR); canon(r);
    int pRY = r[1] * (r[1] + 1) / 2 + r[0], pGB = r[3] * (r[3] + 1) / 2 + r[2];
    return TMIN + TSCALE * TB[k][(long)pRY * NP2 + pGB];
}

/* ---------------- single-row DP with closure hazard ---------------- */
#define NG 13
static const double LG[NG] = {0, .01, .02, .035, .05, .075, .1, .15, .2, .3, .45, .6, 1.0};
static double F[NG][NRS];
static double Tsc(int c) { return c * (c + 1) / 2.0; }
static void solve_single(int g) {
    double lam = LG[g], rho = 1 - (1 - H) * (1 - lam);
    for (int s = 0; s < NRS; s++) if (RL[s]) F[g][s] = Tsc(RC[s]);
    for (int p = 10; p >= 0; p--) for (int s = 0; s < NRS; s++) {
        if (RL[s] || RP[s] != p) continue;
        double T = Tsc(RC[s]);
        #define VAL(s2) (rho * Tsc(RC[s2]) + (1 - rho) * F[g][s2])
        double mpas[21], ppas[21]; int npw = 0;
        double mact[400], pact[400]; int nact = 0;
        for (int w1 = 1; w1 <= 6; w1++) for (int w2 = w1; w2 <= 6; w2++) {
            double pw = (w1 == w2 ? 1.0 : 2.0) / 36.0;
            int s1 = TR[0][s][w1 + w2];
            double mw = s1 >= 0 ? VAL(s1) : -1e18;
            mpas[npw] = mw; ppas[npw] = pw; npw++;
            for (int cd = 1; cd <= 6; cd++) {
                double best = mw;
                for (int j = 0; j < 2; j++) {
                    int w = j ? w2 : w1; if (j && w2 == w1) break;
                    int a = TR[0][s][w + cd]; if (a >= 0 && VAL(a) > best) best = VAL(a);
                    if (s1 >= 0) { int b = TR[0][s1][w + cd]; if (b >= 0 && VAL(b) > best) best = VAL(b); }
                }
                mact[nact] = best; pact[nact] = pw / 6.0; nact++;
            }
        }
        double x = T;
        for (int it = 0; it < 2000; it++) {
            double Sx = rho * T + (1 - rho) * x, a = 0, b = 0;
            for (int q = 0; q < nact; q++) a += pact[q] * (mact[q] > Sx ? mact[q] : Sx);
            for (int q = 0; q < npw; q++) b += ppas[q] * (mpas[q] > Sx ? mpas[q] : Sx);
            double xn = ACT * a + (1 - ACT) * b;
            if (fabs(xn - x) < 1e-9) { x = xn; break; }
            x = xn;
        }
        F[g][s] = x;
    }
}
static double Fint(double lam, int s) {
    if (RL[s]) return Tsc(RC[s]);
    int g = 0; while (g < NG - 2 && lam > LG[g + 1]) g++;
    double t = (lam - LG[g]) / (LG[g + 1] - LG[g]); if (t < 0) t = 0; if (t > 1) t = 1;
    return F[g][s] * (1 - t) + F[g + 1][s] * t;
}

/* value of a state after my move: table (+ threat correction) */
static double Wc(const int *r, int k, const double *lam, double kappa) {
    if (terminal(r, k)) return score(r, k);
    double sc = score(r, k), v = (1 - H) * Vtab(r, k) + H * sc;
    if (kappa != 0) for (int i = 0; i < NR; i++) if (lam[i] > 0) v += kappa * (Fint(lam[i], r[i]) - F[0][r[i]]);
    return v;
}

/* ---------------- sim ---------------- */
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

static double PL = 0.034;
static double lam_row(int i, int seat, int R[5][MAXR], const int *closed) {
    if (closed[i]) return 1.0;
    double prod = 1;
    for (int p = 0; p < 5; p++) {
        if (p == seat) continue;
        int st = R[p][i], c = RC[st];
        double reach = c >= 5 ? 1 : c == 4 ? .55 : c == 3 ? .25 : c == 2 ? .1 : .03;
        double cost = skipcost(st, DIR[i], DIR[i] == 0 ? 12 : 2);
        prod *= 1 - PL * reach * exp(-cost / 0.12);
    }
    double l = 1 - prod; return l > 0.6 ? 0.6 : l;
}

static void hero_turn(int *r, int *k, int active, int w1, int w2, const int *c, const int *closed,
                      const double *lam, double kappa) {
    int S = w1 + w2, opt[5][MAXR], mk[5], t[MAXR], best[MAXR], bestk = *k, no = 1;
    memcpy(opt[0], r, sizeof(int) * NR); mk[0] = 0;
    for (int i = 0; i < NR; i++) {
        int n = trw(closed, i, r[i], S); if (n < 0) continue;
        memcpy(opt[no], r, sizeof(int) * NR); opt[no][i] = n; mk[no] = 1; no++;
    }
    double bv = -INFINITY;
    for (int a = 0; a < no; a++) {
        int kk = (!mk[a] && active) ? *k + 1 : *k;
        double v = Wc(opt[a], kk, lam, kappa);
        if (v > bv) { bv = v; memcpy(best, opt[a], sizeof(int) * NR); bestk = kk; }
        if (!active || nlocks(opt[a]) >= 2) continue;
        for (int i = 0; i < NR; i++) for (int j = 0; j < 2; j++) {
            int w = j ? w2 : w1; if (j && w2 == w1) break;
            int n = trw(closed, i, opt[a][i], w + c[i]); if (n < 0) continue;
            memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
            v = Wc(t, *k, lam, kappa);
            if (v > bv) { bv = v; memcpy(best, t, sizeof(int) * NR); bestk = *k; }
        }
    }
    memcpy(r, best, sizeof(int) * NR); *k = bestk;
}

static void heur_turn(int *r, int *k, int active, int w1, int w2, const int *c, const int *closed,
                      double theta, double penmax) {
    int S = w1 + w2;
    int opt[5][MAXR], mk[5], t[MAXR], no = 1;
    double cw[5];
    memcpy(opt[0], r, sizeof(int) * NR); mk[0] = 0; cw[0] = 0;
    for (int i = 0; i < NR; i++) {
        int n = trw(closed, i, r[i], S); if (n < 0) continue;
        memcpy(opt[no], r, sizeof(int) * NR); opt[no][i] = n; mk[no] = 1;
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
    {"patient x4   ", {0, 0, 0, 0},       {0, 0, 0, 0}},
    {"v.cautious x4", {.02, .02, .02, .02}, {.05, .05, .05, .05}},
    {"cautious x4  ", {.04, .04, .04, .04}, {.12, .12, .12, .12}},
    {"balanced x4  ", {.09, .09, .09, .09}, {.25, .25, .25, .25}},
    {"greedy x4    ", {1, 1, 1, 1},       {1, 1, 1, 1}},
    {"mixed        ", {0, .04, .09, 1},   {0, .12, .25, 1}},
};
#define NSC (int)(sizeof(SC) / sizeof(SC[0]))
typedef struct { const char *name; double kappa; } Mode;   /* kappa<0 => heuristic hero */
static Mode MODES[] = {{"heuristic", -1}, {"table only", 0}, {"table+thr k=1", 1}, {"table+thr k=1.5", 1.5}, {"table+thr k=2", 2}};
#define NMODE (int)(sizeof(MODES) / sizeof(MODES[0]))

int main(int argc, char **argv) {
    build_rows(); load_table(argv[1]);
    int nsim = argc > 2 ? atoi(argv[2]) : 20000;
    for (int g = 0; g < NG; g++) solve_single(g);
    printf("# H=%.3f ACT=%.2f\n", H, ACT);
    for (int c0 = 0; c0 <= 6; c0 += 3) { /* quick look: single-row value at a few states */
        int st = 0; for (int s2 = 0; s2 < NRS; s2++) if (RP[s2] == (c0 ? 6 : 0) && RC[s2] == c0 && !RL[s2]) st = s2;
        printf("# row(p=%d,c=%d): f(lam=0)=%.2f  f(.1)=%.2f  f(.3)=%.2f  f(1)=%.2f\n", RP[st], RC[st], Fint(0, st), Fint(.1, st), Fint(.3, st), Fint(1, st));
    }
    printf("# hero_mean(se) / hero_win_share / opp_mean   (hero seat random, %d games)\n", nsim);
    for (int s = 0; s < NSC; s++) {
        printf("%s\n", SC[s].name);
        for (int md = 0; md < NMODE; md++) {
            double sh = 0, sh2 = 0, so = 0, win = 0, turns = 0, pen = 0;
            for (int g = 0; g < nsim; g++) {
                rs = 1234567ULL + 7919ULL * g;
                int seat = (int)(splitmix(&rs) % 5);
                int R[5][MAXR] = {{0}}, K[5] = {0}, closed[MAXR] = {0}, t;
                int oidx[5], q = 0; for (int p = 0; p < 5; p++) oidx[p] = (p == seat) ? -1 : q++;
                for (t = 0;; t++) {
                    int a = t % 5, w1 = die(), w2 = die(), c[MAXR];
                    for (int i = 0; i < NR; i++) c[i] = die();
                    double lam[MAXR];
                    if (MODES[md].kappa > 0) for (int i = 0; i < NR; i++) lam[i] = lam_row(i, seat, R, closed);
                    else for (int i = 0; i < NR; i++) lam[i] = 0;
                    for (int p = 0; p < 5; p++) {
                        int active = (p == a);
                        if (p == seat) {
                            if (MODES[md].kappa < 0) heur_turn(R[p], &K[p], active, w1, w2, c, closed, .09, .25);
                            else hero_turn(R[p], &K[p], active, w1, w2, c, closed, lam, MODES[md].kappa);
                        } else heur_turn(R[p], &K[p], active, w1, w2, c, closed, SC[s].th[oidx[p]], SC[s].pm[oidx[p]]);
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
            printf("   %-15s %7.2f (%.2f)   win %.3f   opp %6.2f   turns %.1f  pen %.2f\n", MODES[md].name, m, sd / sqrt(nsim), win / nsim, so / nsim, turns / nsim, pen / nsim);
            fflush(stdout);
        }
    }
    return 0;
}
