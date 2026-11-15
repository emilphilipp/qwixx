/* Stationary ("random phase") Qwixx DP.
 * Each turn I am active with prob ACT=1/NPL, passive otherwise; after every turn the game
 * ends with prob H (opponent-driven).  V(s,k) = expected final score *before* the next turn.
 * Passive turns can leave the state unchanged (self loop) -> solved as a scalar fixed point.
 * usage: ./qwixx_stat H NPL outdir
 * writes outdir/table_k{0..3}.bin (uint16, dense pair-indexed) and outdir/meta.txt
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
static const int DIR[MAXR] = {0, 0, 1, 1};
static int RP[NRS], RC[NRS], RL[NRS];
static int TR[2][NRS][14];
static float *memo;
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
static long idx(const int *r, int k) {
    long x = 0; for (int i = 0; i < NR; i++) x = x * NRS + r[i];
    return x * 4 + k;
}
static double V(const int *r0, int k);
static double W(const int *r, int k) {
    if (terminal(r, k)) return score(r, k);
    return (1 - H) * V(r, k) + H * score(r, k);
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
/* expected value of an ACTIVE turn given the whites (never a self loop: marks or penalty) */
static double active_w(const int *r, int k, int w1, int w2) {
    int opt[5][MAXR], mk[5], t[MAXR];
    int no = white_opts(r, w1 + w2, opt, mk);
    double m0 = -INFINITY;
    for (int a = 0; a < no; a++) { double v = mk[a] ? W(opt[a], k) : W(opt[a], k + 1); if (v > m0) m0 = v; }
    double M[MAXR][6];
    for (int i = 0; i < NR; i++) for (int d = 1; d <= 6; d++) {
        double best = -INFINITY;
        for (int a = 0; a < no; a++) {
            if (nlocks(opt[a]) >= 2) continue;
            for (int j = 0; j < 2; j++) {
                int w = j ? w2 : w1; if (j && w2 == w1) break;
                int n = TR[DIR[i]][opt[a][i]][w + d]; if (n < 0) continue;
                memcpy(t, opt[a], sizeof(int) * NR); t[i] = n;
                double v = W(t, k); if (v > best) best = v;
            }
        }
        M[i][d - 1] = best;
    }
    return expmax(m0, M);
}
static double V(const int *r0, int k) {
    int r[MAXR]; memcpy(r, r0, sizeof(int) * NR); canon(r);
    if (terminal(r, k)) return score(r, k);
    long id = idx(r, k);
    if (!isnan(memo[id])) return memo[id];
    double sc = score(r, k), A = 0, mw[21], pw[21]; int nw = 0;
    for (int w1 = 1; w1 <= 6; w1++) for (int w2 = w1; w2 <= 6; w2++) {
        double p = (w1 == w2 ? 1.0 : 2.0) / 36.0;
        A += p * active_w(r, k, w1, w2);
        int opt[5][MAXR], mk[5]; int no = white_opts(r, w1 + w2, opt, mk);
        double best = -INFINITY;
        for (int a = 1; a < no; a++) { double v = W(opt[a], k); if (v > best) best = v; }
        mw[nw] = best; pw[nw] = p; nw++;
    }
    double a_ = 1 - H, b = H * sc, v = sc;
    for (int it = 0; it < 500; it++) {
        double s = 0, self = a_ * v + b;
        for (int q = 0; q < nw; q++) s += pw[q] * (mw[q] > self ? mw[q] : self);
        double vn = ACT * A + (1 - ACT) * s;
        if (fabs(vn - v) < 1e-8) { v = vn; break; }
        v = vn;
    }
    memo[id] = (float)v;
    return v;
}

int main(int argc, char **argv) {
    H = argc > 1 ? atof(argv[1]) : 0.035;
    int npl = argc > 2 ? atoi(argv[2]) : 5; ACT = 1.0 / npl;
    const char *out = argc > 3 ? argv[3] : "table";
    build_rows();
    long n = 1; for (int i = 0; i < NR; i++) n *= NRS; n *= 4;
    memo = malloc(sizeof(float) * n);
    for (long i = 0; i < n; i++) memo[i] = NAN;
    clock_t t0 = clock(); int r0[MAXR] = {0};
    double v = V(r0, 0);
    long filled = 0; for (long i = 0; i < n; i++) filled += !isnan(memo[i]);
    printf("H=%.3f NPL=%d  V(start)=%.4f  states=%ld  %.0fs\n", H, npl, v, filled, (double)(clock() - t0) / CLOCKS_PER_SEC);
    /* export dense pair-indexed uint16 */
    const int NP2 = NRS * (NRS + 1) / 2;
    double mn = 1e9, mx = -1e9;
    for (long i = 0; i < n; i++) if (!isnan(memo[i])) { if (memo[i] < mn) mn = memo[i]; if (memo[i] > mx) mx = memo[i]; }
    double scale = (mx - mn) / 65535.0;
    char fn[512]; snprintf(fn, sizeof fn, "%s/meta.txt", out);
    FILE *m = fopen(fn, "w"); fprintf(m, "%.6f %.9f %.4f %d %d %d\n", mn, scale, H, npl, NP2, 4); fclose(m);
    uint16_t *buf = malloc(sizeof(uint16_t) * NP2 * NP2);
    for (int k = 0; k < 4; k++) {
        for (int b = 0; b < NRS; b++) for (int a = 0; a <= b; a++) {
            int pRY = b * (b + 1) / 2 + a;
            for (int d = 0; d < NRS; d++) for (int c = 0; c <= d; c++) {
                int pGB = d * (d + 1) / 2 + c;
                int r[4] = {a, b, c, d};
                float v2 = memo[idx(r, k)];
                buf[(long)pRY * NP2 + pGB] = isnan(v2) ? 0 : (uint16_t)lround((v2 - mn) / scale);
            }
        }
        snprintf(fn, sizeof fn, "%s/table_k%d.bin", out, k);
        FILE *f = fopen(fn, "wb"); fwrite(buf, 2, (long)NP2 * NP2, f); fclose(f);
    }
    printf("exported; range [%.3f, %.3f], step %.5f\n", mn, mx, scale);
    return 0;
}
