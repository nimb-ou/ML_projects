/* Digits memory store -- C version.
 *
 * Reads digits.csv (produced once by digits_memory.py) into RAM and answers
 * queries against it. Fully offline.
 *
 * Index:
 *   - open-addressed hash table keyed by the raw bytes of the feature vector
 *     for O(1) exact match
 *   - brute-force linear scan for nearest neighbor (1797 x 64 = trivial)
 *
 * Build: gcc -O2 -o digits_memory digits_memory.c -lm
 * Run:   ./digits_memory [path/to/digits.csv]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 4096
#define N_FEATURES  64
#define HASH_SIZE   8192  /* power of two, ~4x load capacity */

typedef struct {
    double features[N_FEATURES];
    int    label;
} Sample;

static Sample samples[MAX_SAMPLES];
static int    n_samples = 0;

/* Open-addressed hash table: stores sample index, or -1 for empty. */
static int hash_table[HASH_SIZE];

/* FNV-1a 64-bit over the raw byte representation of the features. */
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
    size_t   slot = h & (HASH_SIZE - 1);
    while (hash_table[slot] != -1) {
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    hash_table[slot] = idx;
}

static int hash_find(const double *q) {
    uint64_t h = hash_features(q);
    size_t   slot = h & (HASH_SIZE - 1);
    while (hash_table[slot] != -1) {
        if (features_equal(q, samples[hash_table[slot]].features)) {
            return hash_table[slot];
        }
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    return -1;
}

static int load_csv(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }
    char line[8192];
    while (fgets(line, sizeof(line), fp)) {
        if (n_samples >= MAX_SAMPLES) {
            fprintf(stderr, "Too many rows; raise MAX_SAMPLES\n");
            fclose(fp);
            return -1;
        }
        char *tok = strtok(line, ",\n");
        int   i = 0;
        for (; i < N_FEATURES && tok; i++) {
            samples[n_samples].features[i] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
        if (i != N_FEATURES || !tok) {
            continue; /* skip malformed line */
        }
        samples[n_samples].label = atoi(tok);
        hash_insert(n_samples);
        n_samples++;
    }
    fclose(fp);
    return 0;
}

/* Returns the matched sample index. Sets *out_dist and *out_exact. */
static int query(const double *q, double *out_dist, int *out_exact) {
    int hit = hash_find(q);
    if (hit != -1) {
        *out_dist  = 0.0;
        *out_exact = 1;
        return hit;
    }

    int    best   = 0;
    double best_d = INFINITY;
    for (int i = 0; i < n_samples; i++) {
        const double *s = samples[i].features;
        double        d = 0.0;
        for (int j = 0; j < N_FEATURES; j++) {
            double diff = q[j] - s[j];
            d += diff * diff;
        }
        if (d < best_d) {
            best_d = d;
            best   = i;
        }
    }
    *out_dist  = sqrt(best_d);
    *out_exact = 0;
    return best;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void run_query(const char *tag, const double *q) {
    double dist;
    int    exact;
    double t0   = now_us();
    int    idx  = query(q, &dist, &exact);
    double dt   = now_us() - t0;
    int    label = samples[idx].label;
    printf("%-18s label=%d  dist=%.4f  exact=%d  (%.1f us)\n",
           tag, label, dist, exact, dt);
}

int main(int argc, char **argv) {
    const char *csv = (argc > 1) ? argv[1] : "digits.csv";
    for (int i = 0; i < HASH_SIZE; i++) hash_table[i] = -1;

    if (load_csv(csv) != 0) return 1;
    printf("Loaded %d samples from %s\n\n", n_samples, csv);

    /* Exact match on sample 42. */
    run_query("exact", samples[42].features);

    /* Perturbed copy of sample 42. */
    double perturbed[N_FEATURES];
    memcpy(perturbed, samples[42].features, sizeof(perturbed));
    perturbed[0]  += 0.7;
    perturbed[15] -= 1.3;
    run_query("perturbed", perturbed);

    /* Noisy copy of sample 100 (deterministic noise). */
    double noisy[N_FEATURES];
    memcpy(noisy, samples[100].features, sizeof(noisy));
    unsigned int seed = 1u;
    for (int j = 0; j < N_FEATURES; j++) {
        /* simple LCG, scaled to a small Gaussian-ish range */
        seed = seed * 1103515245u + 12345u;
        double u = ((seed >> 16) & 0x7fff) / 32767.0 - 0.5;
        noisy[j] += u * 4.0;
    }
    run_query("noisy", noisy);

    return 0;
}
