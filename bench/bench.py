"""Benchmark the four memorization techniques.

Measures, for each backend (hash exact, brute force, KDTree, BallTree):
    * index build time (wall + CPU)
    * resident memory growth from building the index
    * per-query latency distribution (min / median / mean / p95 / p99 / max)
    * sustained throughput (queries per second)
    * total CPU time consumed during the query loop

Output:
    * console table
    * bench/results_python.json  (machine-readable)

Run:
    python3 bench/bench.py [--queries 1000] [--noise 2.0] [--seed 0]

The query mix is half exact (drawn from the stored set, guaranteed hit) and
half noisy (Gaussian noise on a stored sample, will fall through to NN).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_digits
from sklearn.neighbors import BallTree, KDTree


HERE = Path(__file__).parent
OUT_JSON = HERE / "results_python.json"


# ----- measurement helpers --------------------------------------------------

@dataclass
class Measurement:
    wall_s: float = 0.0
    cpu_user_s: float = 0.0
    cpu_sys_s: float = 0.0
    rss_delta_bytes: int = 0


@contextmanager
def measured(out: dict, key: str):
    """Record wall, cpu, rss delta around the block."""
    gc.collect()
    p = psutil.Process(os.getpid())
    rss_before = p.memory_info().rss
    cpu_before = p.cpu_times()
    wall_before = time.perf_counter()
    yield
    wall = time.perf_counter() - wall_before
    cpu_after = p.cpu_times()
    rss_after = p.memory_info().rss
    out[key] = asdict(Measurement(
        wall_s=wall,
        cpu_user_s=cpu_after.user - cpu_before.user,
        cpu_sys_s=cpu_after.system - cpu_before.system,
        rss_delta_bytes=rss_after - rss_before,
    ))


def percentile_us(timings_s: list[float]) -> dict:
    s = sorted(timings_s)
    n = len(s)
    return {
        "n": n,
        "min_us":    s[0]              * 1e6,
        "median_us": s[n // 2]         * 1e6,
        "mean_us":   (sum(s) / n)      * 1e6,
        "p95_us":    s[int(n * 0.95)]  * 1e6,
        "p99_us":    s[min(n - 1, int(n * 0.99))] * 1e6,
        "max_us":    s[-1]             * 1e6,
    }


# ----- backends -------------------------------------------------------------

def make_queries(X: np.ndarray, n: int, noise_sigma: float,
                 seed: int) -> tuple[np.ndarray, list[bool]]:
    """Half exact (will hit the hash), half noisy (will fall through to NN)."""
    rng = np.random.default_rng(seed)
    n_exact = n // 2
    n_noisy = n - n_exact
    exact_idx = rng.integers(0, len(X), size=n_exact)
    noisy_idx = rng.integers(0, len(X), size=n_noisy)
    exact_q = X[exact_idx]
    noisy_q = X[noisy_idx] + rng.normal(0, noise_sigma, size=(n_noisy, X.shape[1]))
    queries = np.vstack([exact_q, noisy_q])
    is_exact = [True] * n_exact + [False] * n_noisy
    # shuffle so the loop sees a realistic interleave
    order = rng.permutation(n)
    return queries[order], [is_exact[i] for i in order]


def bench_hash(X: np.ndarray, y: np.ndarray,
               queries: np.ndarray, is_exact: list[bool]) -> dict:
    out: dict = {"name": "hash (exact)"}
    with measured(out, "build"):
        store = {X[i].tobytes(): int(y[i]) for i in range(len(y))}

    # warm path
    for q in queries[:32]:
        store.get(q.tobytes())

    timings = [0.0] * len(queries)
    hits = 0
    with measured(out, "queries"):
        for i, q in enumerate(queries):
            k = q.tobytes()
            t0 = time.perf_counter()
            v = store.get(k)
            timings[i] = time.perf_counter() - t0
            if v is not None:
                hits += 1

    out["per_query"] = percentile_us(timings)
    out["hits"] = hits
    out["throughput_qps"] = len(queries) / out["queries"]["wall_s"]
    return out


def bench_brute(X: np.ndarray, y: np.ndarray,
                queries: np.ndarray, is_exact: list[bool]) -> dict:
    out: dict = {"name": "brute force (numpy)"}
    with measured(out, "build"):
        Xb = np.ascontiguousarray(X, dtype=np.float64)
        yb = np.asarray(y)
        # precompute squared norms for faster L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        Xn = (Xb * Xb).sum(axis=1)

    def nn(q):
        d = Xn + (q * q).sum() - 2.0 * (Xb @ q)
        return int(yb[int(np.argmin(d))])

    for q in queries[:32]:
        nn(q)

    timings = [0.0] * len(queries)
    with measured(out, "queries"):
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            nn(q)
            timings[i] = time.perf_counter() - t0

    out["per_query"] = percentile_us(timings)
    out["throughput_qps"] = len(queries) / out["queries"]["wall_s"]
    return out


def bench_tree(name: str, ctor, X: np.ndarray, y: np.ndarray,
               queries: np.ndarray, is_exact: list[bool]) -> dict:
    out: dict = {"name": name}
    with measured(out, "build"):
        tree = ctor(X)

    for q in queries[:32]:
        tree.query(q.reshape(1, -1), k=1)

    timings = [0.0] * len(queries)
    with measured(out, "queries"):
        for i, q in enumerate(queries):
            qq = q.reshape(1, -1)
            t0 = time.perf_counter()
            _, _ = tree.query(qq, k=1)
            timings[i] = time.perf_counter() - t0

    out["per_query"] = percentile_us(timings)
    out["throughput_qps"] = len(queries) / out["queries"]["wall_s"]
    return out


def bench_hybrid(X: np.ndarray, y: np.ndarray,
                 queries: np.ndarray, is_exact: list[bool]) -> dict:
    """Hash first, BallTree fallback — what a real system would deploy."""
    out: dict = {"name": "hybrid (hash + BallTree)"}
    with measured(out, "build"):
        store = {X[i].tobytes(): int(y[i]) for i in range(len(y))}
        tree = BallTree(X)

    for q in queries[:32]:
        v = store.get(q.tobytes())
        if v is None:
            tree.query(q.reshape(1, -1), k=1)

    timings = [0.0] * len(queries)
    hash_hits = 0
    tree_hits = 0
    with measured(out, "queries"):
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            v = store.get(q.tobytes())
            if v is None:
                tree.query(q.reshape(1, -1), k=1)
                tree_hits += 1
            else:
                hash_hits += 1
            timings[i] = time.perf_counter() - t0

    out["per_query"] = percentile_us(timings)
    out["hash_hits"] = hash_hits
    out["tree_hits"] = tree_hits
    out["throughput_qps"] = len(queries) / out["queries"]["wall_s"]
    return out


# ----- output ---------------------------------------------------------------

def fmt_us(v: float) -> str:
    if v < 10:
        return f"{v:6.2f} us"
    if v < 1000:
        return f"{v:6.1f} us"
    return f"{v/1000:5.2f} ms"


def fmt_mb(v: int) -> str:
    if v < 0:
        return f"{v/1024/1024:+5.2f} MB"
    return f"+{v/1024/1024:5.2f} MB"


def fmt_qps(v: float) -> str:
    if v >= 1e6:
        return f"{v/1e6:5.2f}M qps"
    if v >= 1e3:
        return f"{v/1e3:5.1f}k qps"
    return f"{v:5.0f} qps"


def print_results(results: list[dict], cfg: dict) -> None:
    print("=" * 78)
    print(" DIGIT MEMORY · PYTHON BENCHMARK")
    print("=" * 78)
    print(f" Host       : {platform.machine()}  ·  {platform.system()} {platform.release()}")
    print(f" Python     : {platform.python_version()}")
    print(f" Dataset    : {cfg['n_samples']} samples x {cfg['n_features']} features")
    print(f" Queries    : {cfg['n_queries']}  "
          f"(50% exact + 50% noisy σ={cfg['noise_sigma']})")
    print()

    for i, r in enumerate(results, 1):
        print(f" [{i}/{len(results)}] {r['name']}")
        b = r["build"]
        q = r["queries"]
        pq = r["per_query"]
        print(f"   build       wall {fmt_us(b['wall_s']*1e6)}   "
              f"cpu {b['cpu_user_s']*1000:5.1f} ms user + "
              f"{b['cpu_sys_s']*1000:4.1f} ms sys   "
              f"rss {fmt_mb(b['rss_delta_bytes'])}")
        print(f"   query loop  wall {b['wall_s']*1000:6.1f} ms ".replace(
              f"{b['wall_s']*1000:6.1f}", f"{q['wall_s']*1000:6.1f}") +
              f"  cpu {q['cpu_user_s']*1000:5.1f} ms user + "
              f"{q['cpu_sys_s']*1000:4.1f} ms sys")
        print(f"   per-query   min {fmt_us(pq['min_us'])}  "
              f"median {fmt_us(pq['median_us'])}  "
              f"p95 {fmt_us(pq['p95_us'])}  "
              f"p99 {fmt_us(pq['p99_us'])}")
        print(f"   throughput  {fmt_qps(r['throughput_qps'])}")
        if "hash_hits" in r:
            print(f"   path        hash {r['hash_hits']}  ·  tree {r['tree_hits']}")
        elif "hits" in r:
            print(f"   path        hit  {r['hits']}  ·  miss {len(results)*0 + cfg['n_queries']-r['hits']}")
        print()

    # comparison
    print("-" * 78)
    print(f" {'method':<26} {'build':>9} {'rss':>10} "
          f"{'p50':>9} {'p99':>9} {'throughput':>11}")
    print("-" * 78)
    for r in results:
        print(f" {r['name']:<26} "
              f"{fmt_us(r['build']['wall_s']*1e6):>9} "
              f"{fmt_mb(r['build']['rss_delta_bytes']):>10} "
              f"{fmt_us(r['per_query']['median_us']):>9} "
              f"{fmt_us(r['per_query']['p99_us']):>9} "
              f"{fmt_qps(r['throughput_qps']):>11}")
    print("-" * 78)
    print()
    print(f" Wrote machine-readable results to {OUT_JSON.relative_to(Path.cwd()) if OUT_JSON.is_relative_to(Path.cwd()) else OUT_JSON}")


# ----- main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=1000,
                    help="number of queries to run (default 1000)")
    ap.add_argument("--noise", type=float, default=2.0,
                    help="Gaussian sigma for the noisy half (default 2.0)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed (default 0)")
    args = ap.parse_args()

    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    queries, is_exact = make_queries(X, args.queries, args.noise, args.seed)

    cfg = {
        "n_samples":   int(len(X)),
        "n_features":  int(X.shape[1]),
        "n_queries":   int(args.queries),
        "noise_sigma": float(args.noise),
        "seed":        int(args.seed),
        "host":        f"{platform.machine()} {platform.system()} {platform.release()}",
        "python":      platform.python_version(),
    }

    results: list[dict] = []
    results.append(bench_hash(X, y, queries, is_exact))
    results.append(bench_brute(X, y, queries, is_exact))
    results.append(bench_tree("KDTree",
                              lambda XX: KDTree(XX), X, y, queries, is_exact))
    results.append(bench_tree("BallTree",
                              lambda XX: BallTree(XX), X, y, queries, is_exact))
    results.append(bench_hybrid(X, y, queries, is_exact))

    print_results(results, cfg)

    OUT_JSON.write_text(json.dumps({"cfg": cfg, "results": results}, indent=2))


if __name__ == "__main__":
    main()
