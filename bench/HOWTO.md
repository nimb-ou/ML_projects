# How to benchmark the memorization techniques

This folder is a complete, runnable measurement suite. One command runs every technique discussed in the project on your machine and prints how much **time**, **memory**, and **compute** each one used.

You will see, side by side:

- **Hash exact match** — O(1) dict lookup
- **Brute force NN** — vectorized numpy scan
- **KDTree** — sklearn's axis-aligned spatial index
- **BallTree** — sklearn's hypersphere spatial index
- **Hybrid (hash + BallTree)** — the production pattern: cheap lookup first, expensive fallback only on miss
- **Pure C** — same hybrid pattern, hand-rolled, no dependencies

All measurements are taken on **your CPU**. No mocks. No simulations. The numbers you see are the numbers you would get if you deployed this code today.

---

## TL;DR

```bash
./bench/run_all.sh
```

That's it. Wait ~5 seconds. Read the table at the bottom.

If you want more queries (more stable percentiles, longer run):

```bash
./bench/run_all.sh 5000        # 5k queries
./bench/run_all.sh 50000       # 50k queries, ~1s total
```

---

## What you need

- **macOS** (this guide is written for macOS, but everything works on Linux too)
- **Python 3.10+** with `numpy`, `scikit-learn`, `psutil`
- **A C compiler** — `gcc` or `clang`. macOS ships with `clang` aliased as `gcc`; you have it already.

Install Python deps if you haven't:

```bash
pip install -r requirements.txt
pip install psutil      # one extra dep for the benchmark
```

That's it. No Docker, no virtual envs needed.

---

## What the benchmark measures, in plain terms

For each technique, the suite captures four things:

### 1. Build time

How long it takes to construct the index from the 1797 training samples.

- For the hash: how long to insert 1797 keys into a Python dict.
- For the trees: how long sklearn's C-extension takes to partition space.

Measured with `time.perf_counter()` (Python) and `clock_gettime(CLOCK_MONOTONIC)` (C). Sub-microsecond resolution on modern hardware.

### 2. Memory growth

How much RAM the index actually occupies, measured as the **change in resident set size (RSS)** of the process before and after building.

- Python: `psutil.Process().memory_info().rss` — bytes the OS has handed to your process.
- C: `getrusage(RUSAGE_SELF)` → `ru_maxrss` — peak resident set size since program start.

This is the most honest memory number you can get without instrumenting `malloc` itself. It includes every allocation made by Python, sklearn's C extensions, and any caches the OS hasn't yet reclaimed.

### 3. Per-query latency

For each of the N queries, we measure the wall-clock time **just for the lookup**. The script then computes a distribution:

- **min** — fastest single query
- **median (p50)** — the typical query
- **mean** — average (sensitive to outliers)
- **p95** — 95% of queries are faster than this
- **p99** — 99% of queries are faster than this (the "tail")
- **max** — slowest single query (often a noise event, GC pause, page fault)

The **p99** is the number that matters in production. The median tells you what most queries cost; p99 tells you what your worst customer is going to experience.

### 4. Throughput (queries per second)

Total queries divided by total wall time. This is what you would report on a deployment dashboard.

### 5. CPU time

How much CPU your process actually used during the run.

- **user CPU** — time spent executing your code in userspace.
- **system CPU** — time spent in the kernel (file I/O, page faults, system calls).

If wall time ≫ user CPU, your code is waiting on something (disk, network, contention). For these benchmarks wall ≈ user — everything is in memory and single-threaded.

---

## What's in the query mix?

500 queries that are **byte-identical** to stored samples (will hit the hash) and 500 queries that are **Gaussian-noisy** versions of stored samples with σ=2.0 pixel units (will fall through to nearest-neighbor). The two halves are shuffled before timing so the inner loop sees a realistic interleave, not a hot/cold pattern.

You can change the mix with `--noise`:

```bash
python3 bench/bench.py --queries 5000 --noise 5.0
./bench/bench       --queries 5000 --noise 5.0
```

At σ=0 every query is exact (100% hash hits). At σ=10 the noise dominates the signal and almost everything falls through to NN.

---

## How to read the output

A typical run on an M-class Mac looks like this:

```
 method                       build       rss        p50       p99    throughput
 hash (exact)               559.9 us   +1.00 MB    0.17 us   0.50 us   2.56M qps
 brute force (numpy)        151.1 us   +0.89 MB   20.1 us   28.4 us   49.3k qps
 KDTree                     667.0 us   +0.17 MB   28.7 us   68.0 us   29.8k qps
 BallTree                   731.1 us   +0.12 MB   40.7 us   56.8 us   24.8k qps
 hybrid (hash + BallTree)    1.22 ms   +0.81 MB   35.4 us   54.8 us   45.9k qps
 hash + brute fallback (C)   2.13 ms       --      1.00 us   2.00 us  836.8k qps
```

What you should notice:

1. **The hash is in a different league.** ~0.17 µs median is the cost of one dict lookup. Nothing else can compete on exact match. This is why a cache exists in every system you've ever shipped.

2. **Brute force beats both trees** at this scale. With only 1797 samples, the constant factors of tree traversal swamp the O(log n) savings. The trees win at larger N (try 100k samples and the picture flips).

3. **The hybrid pays a build-time tax** (1.22 ms = hash + tree) but its **p50** is the average of fast hash hits and slower tree misses — exactly what you'd expect from a system where half the queries hit the cache.

4. **Pure C is ~18x faster than the Python hybrid** with the same data, same query mix, same algorithm. That gap is the cost of running through the CPython interpreter. If you care about throughput, that gap is the answer to "should I rewrite this in C?".

---

## Files in this folder

| File | What it is |
|---|---|
| `bench.py` | Python benchmark — runs all five techniques. |
| `bench.c` | C benchmark — measures the hash + brute-fallback pattern. |
| `compare.py` | Reads both JSON outputs and emits a comparison table. |
| `run_all.sh` | Driver — compiles the C bench, runs both, prints the comparison. |
| `results_python.json` | Machine-readable Python output (created by `bench.py`). |
| `results_c.json` | Machine-readable C output (created by the C binary). |
| `HOWTO.md` | This file. |

---

## Running parts individually

If you want to run just one piece:

```bash
# Python only
python3 bench/bench.py --queries 1000

# C only (requires digits.csv from the main project)
gcc -O3 -Wall -Wextra -o bench/bench bench/bench.c -lm
./bench/bench --queries 1000

# Cross-language comparison (requires both JSON files to exist)
python3 bench/compare.py
```

---

## Caveats and notes

### Clock granularity

`clock_gettime(CLOCK_MONOTONIC)` on macOS has roughly **microsecond resolution** for the kind of repeated short reads we do here. That means a single C measurement below ~1 µs will round to 0 or 1 µs. The **totals** are accurate; the per-query percentiles for very fast operations (like a single hash lookup) are bounded below by the timer itself.

The Python side uses `time.perf_counter()`, which on macOS resolves to ~100 ns. You'll see sub-microsecond medians there.

### Noise from other processes

Run with as little else going on as possible. Chrome, Spotify, and Slack will jitter your tail latencies. A clean run is one where p99 is close to median × 2 or so. If p99 is 100× the median, something else on your machine grabbed the CPU.

### Memory readings can be negative

If you run the benchmark twice in the same process, GC may free memory between runs and the RSS delta can dip below zero. That's not a bug — it means Python reclaimed pages between the start and end of the measured block. The build is still small; the OS just gave the pages back.

### What "compute" means here

We report user + system CPU time. We don't measure FLOPs or cache misses — for that you'd want `perf` (Linux) or `dtrace`/`xctrace` (macOS). For decision-making at the application level, CPU time is the right currency.

---

## What to do next

A few experiments worth trying:

```bash
# How does throughput scale with query count?
for N in 100 1000 10000 100000; do
  ./bench/run_all.sh $N 2>&1 | grep "hash (exact)"
done

# How does noise level affect the hybrid hit rate and latency?
for S in 0.0 0.5 1.0 3.0 5.0; do
  python3 bench/bench.py --queries 5000 --noise $S 2>&1 | grep -E "hybrid|hash"
done

# What does the C version look like on a much larger query set?
./bench/bench --queries 1000000        # 1M queries
```

If you want to plot the data, every run writes a JSON file. Open it with pandas:

```python
import json, pandas as pd
py = json.load(open("bench/results_python.json"))
df = pd.DataFrame([{
    "method": r["name"],
    "p50_us": r["per_query"]["median_us"],
    "p99_us": r["per_query"]["p99_us"],
    "qps":    r["throughput_qps"],
} for r in py["results"]])
print(df)
```

That's everything you need. Run `./bench/run_all.sh`, read the table, draw your conclusions.
