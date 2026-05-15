/* bench.c — measure time + resources for the C memorization techniques.
 *
 * Mirrors what bench.py does for Python:
 *   * build the FNV-1a open-addressed hash table over 1797 x 64 doubles
 *   * run a mix of exact (50%) + noisy (50%) queries
 *   * for each query: try the hash first, fall back to brute-force NN on miss
 *   * record per-query wall times, build cost, peak RSS via getrusage
 *
 * Output:
 *   * pretty console table
 *   * bench/results_c.json     (machine-readable)
 *
 * Build:
 *   gcc -O3 -Wall -Wextra -o bench/bench bench/bench.c -lm
 * Run:
 *   ./bench/bench [--queries N] [--noise SIGMA] [--seed S] [--csv PATH]
 *
 * Default csv path: digits.csv (produced by digits_memory.py on first run).
 */

#if defined(__APPLE__)
#  define _DARWIN_C_SOURCE
#else
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>

#define MAX_SAMPLES 4096
#define N_FEATURES  64
#define HASH_SIZE   8192   /* power of two, ~4x load factor over 1797 */

typedef struct {
    double features[N_FEATURES];
    int    label;
} Sample;

static Sample samples[MAX_SAMPLES];
static int    n_samples = 0;
static int    hash_table[HASH_SIZE];

/* ---------- timing helpers ---------- */

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static void rusage_snapshot(double *user, double *sys, long *maxrss_bytes) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *user = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1.0e6;
    *sys  = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1.0e6;
#if defined(__APPLE__)
    /* On macOS, ru_maxrss is in BYTES. */
    *maxrss_bytes = ru.ru_maxrss;
#else
    /* On Linux, ru_maxrss is in KILOBYTES. */
    *maxrss_bytes = ru.ru_maxrss * 1024L;
#endif
}

/* ---------- hash table ---------- */

static uint64_t hash_features(const double *f) {
    const unsigned char *p = (const unsigned char *)f;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < sizeof(double) * N_FEATURES; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int features_equal(const double *a, const double *b) {
    return memcmp(a, b, sizeof(double) * N_FEATURES) == 0;
}

static void hash_insert(int idx) {
    uint64_t h = hash_features(samples[idx].features);
    size_t slot = h & (HASH_SIZE - 1);
    while (hash_table[slot] != -1) slot = (slot + 1) & (HASH_SIZE - 1);
    hash_table[slot] = idx;
}

static int hash_find(const double *q) {
    uint64_t h = hash_features(q);
    size_t slot = h & (HASH_SIZE - 1);
    while (hash_table[slot] != -1) {
        if (features_equal(q, samples[hash_table[slot]].features))
            return hash_table[slot];
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    return -1;
}

static int brute_nn(const double *q, double *out_d) {
    int best = 0;
    double best_d = INFINITY;
    for (int i = 0; i < n_samples; i++) {
        const double *s = samples[i].features;
        double d = 0.0;
        for (int j = 0; j < N_FEATURES; j++) {
            double diff = q[j] - s[j];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = i; }
    }
    if (out_d) *out_d = sqrt(best_d);
    return best;
}

/* ---------- csv loader ---------- */

static int load_csv(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return -1;
    }
    char line[8192];
    while (fgets(line, sizeof(line), fp)) {
        if (n_samples >= MAX_SAMPLES) break;
        char *tok = strtok(line, ",\n");
        int i = 0;
        for (; i < N_FEATURES && tok; i++) {
            samples[n_samples].features[i] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
        if (i != N_FEATURES || !tok) continue;
        samples[n_samples].label = atoi(tok);
        n_samples++;
    }
    fclose(fp);
    return 0;
}

/* ---------- gaussian (Box-Muller) ---------- */

static double rand01_state(uint64_t *state) {
    /* xorshift64 */
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return ((*state) >> 11) / (double)(1ULL << 53);
}

static double gaussian_state(uint64_t *state, double sigma) {
    double u1 = rand01_state(state);
    double u2 = rand01_state(state);
    if (u1 < 1e-300) u1 = 1e-300;
    return sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ---------- comparator for percentile sort ---------- */

static int dcmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* ---------- pretty formatting ---------- */

static const char *fmt_us(double us, char *buf, size_t n) {
    if (us < 10)   snprintf(buf, n, "%5.2f us", us);
    else if (us < 1000) snprintf(buf, n, "%5.1f us", us);
    else           snprintf(buf, n, "%5.2f ms", us / 1000.0);
    return buf;
}

static const char *fmt_mb(long bytes, char *buf, size_t n) {
    snprintf(buf, n, "%+5.2f MB", bytes / 1024.0 / 1024.0);
    return buf;
}

static const char *fmt_qps(double qps, char *buf, size_t n) {
    if (qps >= 1e6) snprintf(buf, n, "%5.2fM qps", qps / 1e6);
    else if (qps >= 1e3) snprintf(buf, n, "%5.1fk qps", qps / 1e3);
    else snprintf(buf, n, "%5.0f qps", qps);
    return buf;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    const char *csv = "digits.csv";
    int    n_queries = 1000;
    double noise = 2.0;
    uint64_t seed = 1ULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--queries") && i + 1 < argc)
            n_queries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise") && i + 1 < argc)
            noise = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc)
            csv = argv[++i];
        else {
            fprintf(stderr,
                    "Usage: %s [--csv PATH] [--queries N] [--noise SIGMA] [--seed S]\n",
                    argv[0]);
            return 1;
        }
    }
    if (seed == 0) seed = 1ULL;

    /* baseline before any work */
    double t_load0 = wall_seconds();
    if (load_csv(csv) != 0) return 1;
    double t_load1 = wall_seconds();

    /* build the hash */
    for (int i = 0; i < HASH_SIZE; i++) hash_table[i] = -1;

    double u_before, s_before;
    long rss_before;
    rusage_snapshot(&u_before, &s_before, &rss_before);
    double wall_b0 = wall_seconds();
    for (int i = 0; i < n_samples; i++) hash_insert(i);
    double wall_b1 = wall_seconds();
    double u_after_b, s_after_b;
    long rss_after_b;
    rusage_snapshot(&u_after_b, &s_after_b, &rss_after_b);

    /* prepare queries: half exact (random sample features), half noisy */
    int n_exact = n_queries / 2;
    int n_noisy = n_queries - n_exact;
    double *Q = (double *)calloc((size_t)n_queries * N_FEATURES, sizeof(double));
    int *is_exact = (int *)calloc((size_t)n_queries, sizeof(int));
    if (!Q || !is_exact) { fprintf(stderr, "OOM\n"); return 1; }

    uint64_t qseed = seed;
    /* exact half */
    for (int i = 0; i < n_exact; i++) {
        int idx = (int)(rand01_state(&qseed) * n_samples);
        memcpy(&Q[i * N_FEATURES], samples[idx].features,
               sizeof(double) * N_FEATURES);
        is_exact[i] = 1;
    }
    /* noisy half */
    for (int i = 0; i < n_noisy; i++) {
        int idx = (int)(rand01_state(&qseed) * n_samples);
        double *qrow = &Q[(n_exact + i) * N_FEATURES];
        memcpy(qrow, samples[idx].features, sizeof(double) * N_FEATURES);
        for (int j = 0; j < N_FEATURES; j++)
            qrow[j] += gaussian_state(&qseed, noise);
        is_exact[n_exact + i] = 0;
    }
    /* fisher-yates shuffle so the loop sees interleaved exact/noisy */
    for (int i = n_queries - 1; i > 0; i--) {
        int j = (int)(rand01_state(&qseed) * (i + 1));
        if (i == j) continue;
        double tmp[N_FEATURES];
        memcpy(tmp, &Q[i * N_FEATURES], sizeof(tmp));
        memcpy(&Q[i * N_FEATURES], &Q[j * N_FEATURES], sizeof(tmp));
        memcpy(&Q[j * N_FEATURES], tmp, sizeof(tmp));
        int t = is_exact[i]; is_exact[i] = is_exact[j]; is_exact[j] = t;
    }

    /* warm */
    for (int i = 0; i < 32 && i < n_queries; i++) {
        int idx = hash_find(&Q[i * N_FEATURES]);
        if (idx == -1) brute_nn(&Q[i * N_FEATURES], NULL);
    }

    /* timed query loop — hash first, brute fallback */
    double *timings = (double *)calloc((size_t)n_queries, sizeof(double));
    double u_before_q, s_before_q;
    long rss_before_q;
    rusage_snapshot(&u_before_q, &s_before_q, &rss_before_q);
    double wall_q0 = wall_seconds();

    int hash_hits = 0, tree_hits = 0;
    for (int i = 0; i < n_queries; i++) {
        const double *q = &Q[i * N_FEATURES];
        double t0 = wall_seconds();
        int idx = hash_find(q);
        if (idx == -1) {
            brute_nn(q, NULL);
            tree_hits++;
        } else {
            hash_hits++;
        }
        timings[i] = wall_seconds() - t0;
    }

    double wall_q1 = wall_seconds();
    double u_after_q, s_after_q;
    long rss_after_q;
    rusage_snapshot(&u_after_q, &s_after_q, &rss_after_q);

    /* percentiles */
    qsort(timings, (size_t)n_queries, sizeof(double), dcmp);
    double t_min    = timings[0]                       * 1e6;
    double t_median = timings[n_queries / 2]           * 1e6;
    double t_p95    = timings[(int)(n_queries * 0.95)] * 1e6;
    int idx99 = (int)(n_queries * 0.99);
    if (idx99 >= n_queries) idx99 = n_queries - 1;
    double t_p99 = timings[idx99] * 1e6;
    double t_max = timings[n_queries - 1] * 1e6;
    double t_sum = 0.0;
    for (int i = 0; i < n_queries; i++) t_sum += timings[i];
    double t_mean = (t_sum / n_queries) * 1e6;

    double wall_build = wall_b1 - wall_b0;
    double wall_query = wall_q1 - wall_q0;
    double throughput = (double)n_queries / wall_query;

    /* console */
    char b1[32], b2[32], b3[32], b4[32], b5[32], b6[32], b7[32];
    printf("==============================================================================\n");
    printf(" DIGIT MEMORY · C BENCHMARK\n");
    printf("==============================================================================\n");
    printf(" CSV       : %s\n", csv);
    printf(" Dataset   : %d samples x %d features\n", n_samples, N_FEATURES);
    printf(" Queries   : %d  (50%% exact + 50%% noisy sigma=%.2f)\n",
           n_queries, noise);
    printf(" Loader    : %.2f ms\n", (t_load1 - t_load0) * 1000.0);
    printf("\n");

    printf(" [1/1] hash + brute-force fallback (C)\n");
    printf("   build       wall %s   cpu %5.1f ms user + %4.1f ms sys   rss %s\n",
           fmt_us(wall_build * 1e6, b1, sizeof(b1)),
           (u_after_b - u_before) * 1000.0,
           (s_after_b - s_before) * 1000.0,
           fmt_mb(rss_after_b - rss_before, b2, sizeof(b2)));
    printf("   query loop  wall %5.1f ms   cpu %5.1f ms user + %4.1f ms sys\n",
           wall_query * 1000.0,
           (u_after_q - u_before_q) * 1000.0,
           (s_after_q - s_before_q) * 1000.0);
    printf("   per-query   min %s  median %s  p95 %s  p99 %s\n",
           fmt_us(t_min, b3, sizeof(b3)),
           fmt_us(t_median, b4, sizeof(b4)),
           fmt_us(t_p95, b5, sizeof(b5)),
           fmt_us(t_p99, b6, sizeof(b6)));
    printf("   throughput  %s\n",
           fmt_qps(throughput, b7, sizeof(b7)));
    printf("   path        hash %d  ·  brute %d\n", hash_hits, tree_hits);
    printf("\n------------------------------------------------------------------------------\n");
    printf(" Peak RSS for the whole process: %s\n",
           fmt_mb(rss_after_q, b1, sizeof(b1)));
    printf("------------------------------------------------------------------------------\n");

    /* json */
    FILE *jf = fopen("bench/results_c.json", "w");
    if (jf) {
        fprintf(jf,
            "{\n"
            "  \"cfg\": {\n"
            "    \"csv\": \"%s\",\n"
            "    \"n_samples\": %d,\n"
            "    \"n_features\": %d,\n"
            "    \"n_queries\": %d,\n"
            "    \"noise_sigma\": %.6f,\n"
            "    \"seed\": %llu\n"
            "  },\n"
            "  \"build\": {\n"
            "    \"wall_s\": %.9f,\n"
            "    \"cpu_user_s\": %.9f,\n"
            "    \"cpu_sys_s\": %.9f,\n"
            "    \"rss_delta_bytes\": %ld\n"
            "  },\n"
            "  \"queries\": {\n"
            "    \"wall_s\": %.9f,\n"
            "    \"cpu_user_s\": %.9f,\n"
            "    \"cpu_sys_s\": %.9f\n"
            "  },\n"
            "  \"per_query\": {\n"
            "    \"n\": %d,\n"
            "    \"min_us\": %.6f,\n"
            "    \"median_us\": %.6f,\n"
            "    \"mean_us\": %.6f,\n"
            "    \"p95_us\": %.6f,\n"
            "    \"p99_us\": %.6f,\n"
            "    \"max_us\": %.6f\n"
            "  },\n"
            "  \"throughput_qps\": %.6f,\n"
            "  \"hash_hits\": %d,\n"
            "  \"brute_hits\": %d,\n"
            "  \"peak_rss_bytes\": %ld\n"
            "}\n",
            csv, n_samples, N_FEATURES, n_queries, noise,
            (unsigned long long)seed,
            wall_build, u_after_b - u_before, s_after_b - s_before,
            rss_after_b - rss_before,
            wall_query, u_after_q - u_before_q, s_after_q - s_before_q,
            n_queries, t_min, t_median, t_mean, t_p95, t_p99, t_max,
            throughput, hash_hits, tree_hits, rss_after_q);
        fclose(jf);
        printf(" Wrote machine-readable results to bench/results_c.json\n");
    }

    free(Q);
    free(is_exact);
    free(timings);
    return 0;
}
