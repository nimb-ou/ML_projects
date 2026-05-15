/* bench.c — measure time, memory, and accuracy for the C memorization
 * techniques, using the same train/test split the Python bench produced.
 *
 *   bench.py first runs and exports:
 *     bench/digits_train.csv   (1347 rows: 64 features + 1 label)
 *     bench/digits_test.csv    ( 450 rows: 64 features + 1 label)
 *
 *   bench.c then:
 *     * builds the FNV-1a open-addressed hash on the train set
 *     * for each test row, tries the hash, falls back to brute-force NN
 *     * records per-prediction wall time + whether it matched the true label
 *     * reports accuracy, percentiles, throughput, build cost, peak RSS
 *
 * Build:  gcc -O3 -Wall -Wextra -o bench/bench bench/bench.c -lm
 * Run:    ./bench/bench
 *         ./bench/bench --train PATH --test PATH
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
#define HASH_SIZE   8192

typedef struct {
    double features[N_FEATURES];
    int    label;
} Sample;

static Sample train_set[MAX_SAMPLES];
static int    n_train = 0;
static Sample test_set[MAX_SAMPLES];
static int    n_test = 0;
static int    hash_table[HASH_SIZE];

/* ---------- timing ---------- */

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static void rusage_snapshot(double *u, double *s, long *maxrss) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *u = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1.0e6;
    *s = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1.0e6;
#if defined(__APPLE__)
    *maxrss = ru.ru_maxrss;          /* bytes */
#else
    *maxrss = ru.ru_maxrss * 1024L;  /* kB -> bytes */
#endif
}

/* ---------- hash ---------- */

static uint64_t hash_features(const double *f) {
    const unsigned char *p = (const unsigned char *)f;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < sizeof(double) * N_FEATURES; i++) {
        h ^= p[i]; h *= 1099511628211ULL;
    }
    return h;
}

static int features_equal(const double *a, const double *b) {
    return memcmp(a, b, sizeof(double) * N_FEATURES) == 0;
}

static void hash_insert(int idx) {
    uint64_t h = hash_features(train_set[idx].features);
    size_t slot = h & (HASH_SIZE - 1);
    while (hash_table[slot] != -1) slot = (slot + 1) & (HASH_SIZE - 1);
    hash_table[slot] = idx;
}

static int hash_find(const double *q) {
    uint64_t h = hash_features(q);
    size_t slot = h & (HASH_SIZE - 1);
    while (hash_table[slot] != -1) {
        if (features_equal(q, train_set[hash_table[slot]].features))
            return hash_table[slot];
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    return -1;
}

static int brute_nn(const double *q) {
    int best = 0;
    double best_d = INFINITY;
    for (int i = 0; i < n_train; i++) {
        const double *s = train_set[i].features;
        double d = 0.0;
        for (int j = 0; j < N_FEATURES; j++) {
            double diff = q[j] - s[j];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

/* ---------- csv loader ---------- */

static int load_csv(const char *path, Sample *into, int *count, int cap) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    char line[8192];
    int n = 0;
    while (fgets(line, sizeof(line), fp) && n < cap) {
        char *tok = strtok(line, ",\n");
        int i = 0;
        for (; i < N_FEATURES && tok; i++) {
            into[n].features[i] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
        if (i != N_FEATURES || !tok) continue;
        into[n].label = atoi(tok);
        n++;
    }
    fclose(fp);
    *count = n;
    return 0;
}

/* ---------- percentile sort ---------- */

static int dcmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* ---------- formatting ---------- */

static const char *fmt_us(double us, char *buf, size_t n) {
    if (us < 10)        snprintf(buf, n, "%5.2f us", us);
    else if (us < 1000) snprintf(buf, n, "%5.1f us", us);
    else                snprintf(buf, n, "%5.2f ms", us / 1000.0);
    return buf;
}

static const char *fmt_mb(long bytes, char *buf, size_t n) {
    snprintf(buf, n, "%+5.2f MB", bytes / 1024.0 / 1024.0);
    return buf;
}

static const char *fmt_qps(double qps, char *buf, size_t n) {
    if (qps >= 1e6)      snprintf(buf, n, "%5.2fM qps", qps / 1e6);
    else if (qps >= 1e3) snprintf(buf, n, "%5.1fk qps", qps / 1e3);
    else                 snprintf(buf, n, "%5.0f qps", qps);
    return buf;
}

int main(int argc, char **argv) {
    const char *train_path = "bench/digits_train.csv";
    const char *test_path  = "bench/digits_test.csv";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--train") && i + 1 < argc) train_path = argv[++i];
        else if (!strcmp(argv[i], "--test") && i + 1 < argc) test_path = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--train PATH] [--test PATH]\n", argv[0]);
            fprintf(stderr,
                "Run 'python3 bench/bench.py' first to generate the CSVs.\n");
            return 1;
        }
    }

    if (load_csv(train_path, train_set, &n_train, MAX_SAMPLES) != 0) {
        fprintf(stderr,
                "Run 'python3 bench/bench.py' first to generate %s.\n",
                train_path);
        return 1;
    }
    if (load_csv(test_path, test_set, &n_test, MAX_SAMPLES) != 0) return 1;

    /* build */
    for (int i = 0; i < HASH_SIZE; i++) hash_table[i] = -1;
    double u_before, s_before; long rss_before;
    rusage_snapshot(&u_before, &s_before, &rss_before);
    double wall_b0 = wall_seconds();
    for (int i = 0; i < n_train; i++) hash_insert(i);
    double wall_b1 = wall_seconds();
    double u_after_b, s_after_b; long rss_after_b;
    rusage_snapshot(&u_after_b, &s_after_b, &rss_after_b);

    /* warm */
    for (int i = 0; i < 32 && i < n_test; i++) {
        int idx = hash_find(test_set[i].features);
        if (idx == -1) brute_nn(test_set[i].features);
    }

    /* predict */
    double *timings = (double *)calloc((size_t)n_test, sizeof(double));
    double u_before_q, s_before_q; long rss_before_q;
    rusage_snapshot(&u_before_q, &s_before_q, &rss_before_q);
    double wall_q0 = wall_seconds();

    int hash_hits = 0, tree_hits = 0;
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        const double *q = test_set[i].features;
        double t0 = wall_seconds();
        int idx = hash_find(q);
        int pred;
        if (idx >= 0) {
            pred = train_set[idx].label;
            hash_hits++;
        } else {
            pred = train_set[brute_nn(q)].label;
            tree_hits++;
        }
        timings[i] = wall_seconds() - t0;
        if (pred == test_set[i].label) correct++;
    }

    double wall_q1 = wall_seconds();
    double u_after_q, s_after_q; long rss_after_q;
    rusage_snapshot(&u_after_q, &s_after_q, &rss_after_q);

    qsort(timings, (size_t)n_test, sizeof(double), dcmp);
    double t_min    = timings[0]                       * 1e6;
    double t_median = timings[n_test / 2]              * 1e6;
    double t_p95    = timings[(int)(n_test * 0.95)]    * 1e6;
    int idx99 = (int)(n_test * 0.99); if (idx99 >= n_test) idx99 = n_test - 1;
    double t_p99 = timings[idx99] * 1e6;
    double t_sum = 0.0;
    for (int i = 0; i < n_test; i++) t_sum += timings[i];
    double t_mean = (t_sum / n_test) * 1e6;
    double t_max  = timings[n_test - 1] * 1e6;

    double wall_build = wall_b1 - wall_b0;
    double wall_query = wall_q1 - wall_q0;
    double throughput = (double)n_test / wall_query;
    double accuracy   = (double)correct / n_test;

    /* console */
    char b1[32], b2[32], b3[32], b4[32], b5[32], b6[32], b7[32];
    printf("==============================================================================\n");
    printf(" DIGIT MEMORY · C BENCHMARK  (speed + memory + accuracy)\n");
    printf("==============================================================================\n");
    printf(" Train  : %d rows  (%s)\n", n_train, train_path);
    printf(" Test   : %d rows  (%s)\n", n_test,  test_path);
    printf("\n");
    printf(" Method: hash (exact, cache) + brute-force 1-NN fallback\n");
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
           fmt_us(t_min,    b3, sizeof(b3)),
           fmt_us(t_median, b4, sizeof(b4)),
           fmt_us(t_p95,    b5, sizeof(b5)),
           fmt_us(t_p99,    b6, sizeof(b6)));
    printf("   throughput  %s\n", fmt_qps(throughput, b7, sizeof(b7)));
    printf("   path        hash %d  ·  brute %d  (held-out hash hits = 0 by design)\n",
           hash_hits, tree_hits);
    printf("\n");
    printf(" -- ACCURACY ------------------------------------------------------------------\n");
    printf("   correct     %d / %d\n", correct, n_test);
    printf("   accuracy    %.2f %%\n", accuracy * 100.0);
    printf("\n");
    printf(" Peak RSS for the whole process: %s\n",
           fmt_mb(rss_after_q, b1, sizeof(b1)));

    /* json */
    FILE *jf = fopen("bench/results_c.json", "w");
    if (jf) {
        fprintf(jf,
            "{\n"
            "  \"cfg\": {\n"
            "    \"train_csv\": \"%s\",\n"
            "    \"test_csv\": \"%s\",\n"
            "    \"n_train\": %d,\n"
            "    \"n_test\": %d,\n"
            "    \"n_features\": %d\n"
            "  },\n"
            "  \"build\": { \"wall_s\": %.9f, \"cpu_user_s\": %.9f, "
            "\"cpu_sys_s\": %.9f, \"rss_delta_bytes\": %ld },\n"
            "  \"queries\": { \"wall_s\": %.9f, \"cpu_user_s\": %.9f, "
            "\"cpu_sys_s\": %.9f },\n"
            "  \"per_query\": { \"n\": %d, \"min_us\": %.6f, "
            "\"median_us\": %.6f, \"mean_us\": %.6f, \"p95_us\": %.6f, "
            "\"p99_us\": %.6f, \"max_us\": %.6f },\n"
            "  \"throughput_qps\": %.6f,\n"
            "  \"hash_hits\": %d,\n"
            "  \"brute_hits\": %d,\n"
            "  \"correct\": %d,\n"
            "  \"accuracy\": %.6f,\n"
            "  \"peak_rss_bytes\": %ld\n"
            "}\n",
            train_path, test_path, n_train, n_test, N_FEATURES,
            wall_build, u_after_b - u_before, s_after_b - s_before,
            rss_after_b - rss_before,
            wall_query, u_after_q - u_before_q, s_after_q - s_before_q,
            n_test, t_min, t_median, t_mean, t_p95, t_p99, t_max,
            throughput, hash_hits, tree_hits, correct, accuracy,
            rss_after_q);
        fclose(jf);
        printf(" Wrote machine-readable results to bench/results_c.json\n");
    }

    free(timings);
    return 0;
}
