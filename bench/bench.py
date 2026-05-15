"""Benchmark every memorization technique AND a set of trained ML
baselines on the same train/test split. Captures speed, memory, AND accuracy.

For each method we record:
    * fit / build time             (wall + CPU)
    * memory growth                (RSS delta)
    * per-prediction latency       (min/median/mean/p95/p99/max)
    * sustained throughput         (qps)
    * test-set accuracy            (top-1)
    * macro F1                     (treats classes equally)

Output:
    * three console tables: timing, accuracy, head-to-head
    * bench/results_python.json    (machine-readable)
    * bench/digits_train.csv       (train features + label)  for the C bench
    * bench/digits_test.csv        (test  features + label)  for the C bench
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import BallTree, KDTree, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")  # quiet sklearn convergence chatter

HERE = Path(__file__).parent
OUT_JSON = HERE / "results_python.json"
TRAIN_CSV = HERE / "digits_train.csv"
TEST_CSV  = HERE / "digits_test.csv"


# ----- measurement ----------------------------------------------------------

@contextmanager
def measured(out: dict, key: str):
    gc.collect()
    p = psutil.Process(os.getpid())
    rss_before = p.memory_info().rss
    cpu_before = p.cpu_times()
    wall_before = time.perf_counter()
    yield
    wall = time.perf_counter() - wall_before
    cpu_after = p.cpu_times()
    rss_after = p.memory_info().rss
    out[key] = {
        "wall_s":          wall,
        "cpu_user_s":      cpu_after.user   - cpu_before.user,
        "cpu_sys_s":       cpu_after.system - cpu_before.system,
        "rss_delta_bytes": rss_after - rss_before,
    }


def percentiles_us(timings_s: list[float]) -> dict:
    s = sorted(timings_s)
    n = len(s)
    return {
        "n":         n,
        "min_us":    s[0]                              * 1e6,
        "median_us": s[n // 2]                         * 1e6,
        "mean_us":   (sum(s) / n)                      * 1e6,
        "p95_us":    s[min(n - 1, int(n * 0.95))]      * 1e6,
        "p99_us":    s[min(n - 1, int(n * 0.99))]      * 1e6,
        "max_us":    s[-1]                             * 1e6,
    }


# ----- methods --------------------------------------------------------------
#
# A "method" is anything that can be `fit(X_train, y_train)`-ed once and then
# `predict(x)`-ed on a single query. We wrap the retrieval techniques in the
# same shape as the sklearn classifiers so they share the timing loop.

class Method:
    family = "?"
    name = "?"

    def fit(self, X, y):
        raise NotImplementedError

    def predict_one(self, x: np.ndarray) -> int:
        raise NotImplementedError


class HashExact(Method):
    family = "retrieval"
    name = "hash (exact, cache-only)"

    def fit(self, X, y):
        self.store = {X[i].tobytes(): int(y[i]) for i in range(len(y))}

    def predict_one(self, x):
        v = self.store.get(x.tobytes())
        return -1 if v is None else v


class BruteNN(Method):
    family = "retrieval"
    name = "brute force 1-NN (numpy)"

    def fit(self, X, y):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.Xn = (self.X * self.X).sum(axis=1)

    def predict_one(self, x):
        d = self.Xn + (x * x).sum() - 2.0 * (self.X @ x)
        return int(self.y[int(np.argmin(d))])


class KDTreeNN(Method):
    family = "retrieval"
    name = "KDTree 1-NN"

    def fit(self, X, y):
        self.tree = KDTree(X)
        self.y = np.asarray(y)

    def predict_one(self, x):
        _, idx = self.tree.query(x.reshape(1, -1), k=1)
        return int(self.y[idx[0, 0]])


class BallTreeNN(Method):
    family = "retrieval"
    name = "BallTree 1-NN"

    def fit(self, X, y):
        self.tree = BallTree(X)
        self.y = np.asarray(y)

    def predict_one(self, x):
        _, idx = self.tree.query(x.reshape(1, -1), k=1)
        return int(self.y[idx[0, 0]])


class KNN(Method):
    family = "retrieval"

    def __init__(self, k: int):
        self.k = k
        self.name = f"{k}-NN (BallTree, distance-weighted)"

    def fit(self, X, y):
        self.clf = KNeighborsClassifier(
            n_neighbors=self.k, weights="distance",
            algorithm="ball_tree")
        self.clf.fit(X, y)

    def predict_one(self, x):
        return int(self.clf.predict(x.reshape(1, -1))[0])


class Hybrid(Method):
    family = "retrieval"
    name = "hybrid (hash + BallTree 1-NN)"

    def fit(self, X, y):
        self.store = {X[i].tobytes(): int(y[i]) for i in range(len(y))}
        self.tree = BallTree(X)
        self.y = np.asarray(y)
        self.hash_hits = 0
        self.tree_hits = 0

    def predict_one(self, x):
        v = self.store.get(x.tobytes())
        if v is not None:
            self.hash_hits += 1
            return v
        _, idx = self.tree.query(x.reshape(1, -1), k=1)
        self.tree_hits += 1
        return int(self.y[idx[0, 0]])


class SklearnWrap(Method):
    family = "classifier"

    def __init__(self, name: str, model):
        self.name = name
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_one(self, x):
        return int(self.model.predict(x.reshape(1, -1))[0])


# ----- benchmarking ---------------------------------------------------------

def run_method(m: Method, X_train, y_train, X_test, y_test) -> dict:
    out: dict = {"name": m.name, "family": m.family}

    with measured(out, "fit"):
        m.fit(X_train, y_train)

    # warm
    for x in X_test[:32]:
        m.predict_one(x)

    timings = [0.0] * len(X_test)
    preds   = np.empty(len(X_test), dtype=np.int64)
    with measured(out, "predict"):
        for i, x in enumerate(X_test):
            t0 = time.perf_counter()
            preds[i] = m.predict_one(x)
            timings[i] = time.perf_counter() - t0

    out["per_query"]      = percentiles_us(timings)
    out["throughput_qps"] = len(X_test) / out["predict"]["wall_s"]
    # accuracy treats predict_one=-1 (hash miss) as a wrong answer
    out["accuracy"]    = float(accuracy_score(y_test, preds))
    out["macro_f1"]    = float(f1_score(y_test, preds, average="macro",
                                        zero_division=0))
    out["weighted_f1"] = float(f1_score(y_test, preds, average="weighted",
                                        zero_division=0))
    out["correct"]     = int((preds == y_test).sum())
    out["total"]       = int(len(y_test))
    if isinstance(m, Hybrid):
        out["hash_hits"] = m.hash_hits
        out["tree_hits"] = m.tree_hits
    return out


# ----- pretty formatting ----------------------------------------------------

def fmt_t(seconds: float) -> str:
    us = seconds * 1e6
    if us < 10:        return f"{us:6.2f} us"
    if us < 1000:      return f"{us:6.1f} us"
    if us < 1_000_000: return f"{us/1000:6.2f} ms"
    return f"{seconds:6.2f} s "


def fmt_us(v: float) -> str:
    if v < 10:    return f"{v:6.2f} us"
    if v < 1000:  return f"{v:6.1f} us"
    return f"{v/1000:5.2f} ms"


def fmt_mb(v: int) -> str:
    return f"{v/1024/1024:+5.2f} MB"


def fmt_qps(v: float) -> str:
    if v >= 1e6: return f"{v/1e6:5.2f}M qps"
    if v >= 1e3: return f"{v/1e3:5.1f}k qps"
    return f"{v:5.0f} qps"


def fmt_pct(v: float) -> str:
    return f"{v*100:6.2f}%"


def print_tables(results: list[dict], cfg: dict) -> None:
    print()
    print("=" * 90)
    print(" DIGIT MEMORY · BENCHMARK  (speed + memory + accuracy)")
    print("=" * 90)
    print(f" Host          : {platform.machine()}  ·  {platform.system()} {platform.release()}")
    print(f" Python        : {platform.python_version()}")
    print(f" Dataset       : {cfg['n_samples']} samples × {cfg['n_features']} features")
    print(f" Train / Test  : {cfg['n_train']} / {cfg['n_test']}  "
          f"(stratified, seed={cfg['split_seed']})")
    print()

    # ----- timing table -----
    print(" -- TIMING / MEMORY -----------------------------------------------------------")
    print(f" {'method':<34} {'fit':>10} {'p50':>10} {'p99':>10} "
          f"{'qps':>11} {'mem':>10}")
    print(" " + "-" * 88)
    for r in results:
        print(f" {r['name']:<34} "
              f"{fmt_t(r['fit']['wall_s']):>10} "
              f"{fmt_us(r['per_query']['median_us']):>10} "
              f"{fmt_us(r['per_query']['p99_us']):>10} "
              f"{fmt_qps(r['throughput_qps']):>11} "
              f"{fmt_mb(r['fit']['rss_delta_bytes']):>10}")
    print()

    # ----- accuracy table -----
    print(" -- ACCURACY ------------------------------------------------------------------")
    print(f" {'method':<34} {'accuracy':>10} {'macro F1':>10} "
          f"{'wtd F1':>10} {'correct':>14}")
    print(" " + "-" * 88)
    for r in results:
        print(f" {r['name']:<34} "
              f"{fmt_pct(r['accuracy']):>10} "
              f"{fmt_pct(r['macro_f1']):>10} "
              f"{fmt_pct(r['weighted_f1']):>10} "
              f"{r['correct']:>6} / {r['total']:<4}")
    print()

    # ----- head-to-head -----
    retrieval = [r for r in results if r["family"] == "retrieval"
                 and r["name"] != "hash (exact, cache-only)"]
    classifier = [r for r in results if r["family"] == "classifier"]
    if retrieval and classifier:
        best_r = max(retrieval, key=lambda r: r["accuracy"])
        best_c = max(classifier, key=lambda r: r["accuracy"])
        print(" -- HEAD-TO-HEAD --------------------------------------------------------------")
        print(f"   best retrieval method  : {best_r['name']}")
        print(f"     accuracy             : {fmt_pct(best_r['accuracy']).strip()}")
        print(f"     fit time             : {fmt_t(best_r['fit']['wall_s']).strip()}")
        print(f"     p50 inference        : {fmt_us(best_r['per_query']['median_us']).strip()}")
        print()
        print(f"   best trained classifier: {best_c['name']}")
        print(f"     accuracy             : {fmt_pct(best_c['accuracy']).strip()}")
        print(f"     fit time             : {fmt_t(best_c['fit']['wall_s']).strip()}")
        print(f"     p50 inference        : {fmt_us(best_c['per_query']['median_us']).strip()}")
        print()
        delta = (best_r["accuracy"] - best_c["accuracy"]) * 100
        sign = "+" if delta >= 0 else ""
        speedup = best_c["fit"]["wall_s"] / max(best_r["fit"]["wall_s"], 1e-9)
        print(f"   retrieval - classifier : {sign}{delta:.2f} accuracy points,  "
              f"fit time {speedup:.0f}× faster")
        print()
    print(f" Wrote machine-readable results to "
          f"{OUT_JSON.relative_to(Path.cwd()) if OUT_JSON.is_relative_to(Path.cwd()) else OUT_JSON}")


# ----- main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="train/test split seed (default 42)")
    ap.add_argument("--test-size", type=float, default=0.25,
                    help="test fraction (default 0.25)")
    args = ap.parse_args()

    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.seed, stratify=y)

    # export for C bench
    np.savetxt(TRAIN_CSV, np.hstack([X_train, y_train.reshape(-1, 1)]),
               delimiter=",", fmt="%g")
    np.savetxt(TEST_CSV,  np.hstack([X_test,  y_test.reshape(-1, 1)]),
               delimiter=",", fmt="%g")

    cfg = {
        "n_samples":   int(len(X)),
        "n_features":  int(X.shape[1]),
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)),
        "split_seed":  int(args.seed),
        "host":        f"{platform.machine()} {platform.system()} {platform.release()}",
        "python":      platform.python_version(),
    }

    methods: list[Method] = [
        HashExact(),
        BruteNN(),
        KDTreeNN(),
        BallTreeNN(),
        KNN(3),
        KNN(5),
        Hybrid(),
        SklearnWrap("Logistic Regression",
                    LogisticRegression(max_iter=2000, n_jobs=1)),
        SklearnWrap("Gaussian Naive Bayes", GaussianNB()),
        SklearnWrap("Random Forest (100)",
                    RandomForestClassifier(n_estimators=100,
                                           random_state=args.seed, n_jobs=1)),
        SklearnWrap("SVM (RBF)",
                    SVC(kernel="rbf", gamma="scale")),
        SklearnWrap("MLP (1x100)",
                    MLPClassifier(hidden_layer_sizes=(100,), max_iter=400,
                                  random_state=args.seed)),
    ]

    results = []
    for m in methods:
        print(f"  · {m.name:<34} ...", end="", flush=True)
        r = run_method(m, X_train, y_train, X_test, y_test)
        results.append(r)
        print(f" acc={r['accuracy']*100:.2f}%   "
              f"p50={r['per_query']['median_us']:.1f} us   "
              f"fit={r['fit']['wall_s']*1000:.1f} ms")

    print_tables(results, cfg)
    OUT_JSON.write_text(json.dumps({"cfg": cfg, "results": results}, indent=2))


if __name__ == "__main__":
    main()
