# How to benchmark the memorization techniques (with accuracy)

This folder is a complete, runnable measurement suite. **One command runs every technique discussed in the project on your machine and prints time, memory, compute, and accuracy** — and pits the retrieval-based methods against trained ML classifiers on the same held-out test set.

You will see, side by side:

**Retrieval methods**

- Hash exact match (cache-only)
- Brute force 1-NN
- KDTree 1-NN
- BallTree 1-NN
- 3-NN and 5-NN (distance-weighted, BallTree)
- Hybrid (hash + BallTree 1-NN)
- C implementation (hash + brute-force fallback)

**Trained classifiers (sklearn)**

- Logistic Regression
- Gaussian Naive Bayes
- Random Forest (100 trees)
- SVM (RBF kernel)
- MLP (1 hidden layer of 100 units)

All measurements are taken on **your CPU**. No mocks. No simulations. The accuracy numbers are computed on a proper stratified 75/25 train/test split, seed 42 by default.

---

## TL;DR

```bash
./bench/run_all.sh
```

Wait ~10 seconds. Read the three tables at the end.

If you want a different train/test split:

```bash
./bench/run_all.sh 7      # seed 7
./bench/run_all.sh 123    # seed 123
```

---

## What you need

- **macOS** (or Linux — this works there too)
- **Python 3.10+** with `numpy`, `scikit-learn`, `psutil`
- **A C compiler** — `gcc` or `clang`. macOS ships with `clang` aliased as `gcc`; you have it already.

Install the Python deps if you haven't:

```bash
pip install -r requirements.txt
```

That's it. No Docker, no virtual env needed.

---

## What the benchmark measures

For each method (retrieval and classifier), the suite captures:

### 1. Fit / build time

How long it takes to prepare the model for inference.

- **Retrieval methods**: time to build the index (dict insertion, tree construction).
- **Classifiers**: full training time (gradient descent, tree growing, etc.).

### 2. Memory growth

RSS delta from before to after building the model, via `psutil` (Python) and `getrusage()` (C). This is the most honest "how much RAM does the model take" number you can get.

### 3. Per-prediction latency

The wall-clock cost of predicting a single label. The script reports the full distribution: min, median, mean, p95, p99, max. **p99 is the number that matters in production** — that's what your slowest customer experiences.

### 4. Throughput

Total test predictions divided by total wall time. The qps you would report on a deployment dashboard.

### 5. Accuracy

The fraction of held-out test samples for which the predicted label matches the true label. The same metric every benchmark in the literature uses.

### 6. Macro F1 / Weighted F1

Per-class F1 score, averaged either treating all classes equally (macro) or weighted by class support (weighted). Both are reported because they answer different questions:

- **Accuracy**: "out of every test sample, how often did we get it right?"
- **Macro F1**: "averaged over the ten digit classes, how good are we?"
- **Weighted F1**: like macro F1 but weighted by how common each class is in the test set.

For this dataset the three are almost identical because classes are balanced (~45 samples per class in the test set). On an imbalanced dataset they would diverge sharply.

---

## What you get out

Three tables, printed in order.

### Table 1: TIMING / MEMORY

```
 method                                    fit        p50        p99         qps        mem
 hash (exact, cache-only)             466.4 us    0.25 us    0.37 us   2.99M qps   +0.75 MB
 brute force 1-NN (numpy)              70.0 us    14.9 us    19.7 us   65.0k qps   +0.00 MB
 KDTree 1-NN                           2.08 ms    40.9 us    73.5 us   24.3k qps   +0.16 MB
 ...
 Logistic Regression                  51.70 ms    17.2 us    36.2 us   55.1k qps   +2.67 MB
 Random Forest (100)                 118.82 ms    1.08 ms    1.40 ms     910 qps   +6.16 MB
 SVM (RBF)                            13.34 ms    64.9 us    98.9 us   15.1k qps   +0.34 MB
 MLP (1x100)                         312.12 ms    26.7 us    67.8 us   35.9k qps   +5.72 MB
```

### Table 2: ACCURACY

```
 method                               accuracy   macro F1     wtd F1        correct
 hash (exact, cache-only)                0.00%      0.00%      0.00%      0 / 450
 brute force 1-NN (numpy)               98.44%     98.42%     98.43%    443 / 450
 ...
 Logistic Regression                    96.22%     96.19%     96.23%    433 / 450
 Random Forest (100)                    96.00%     95.96%     95.97%    432 / 450
 SVM (RBF)                              99.11%     99.11%     99.11%    446 / 450
 MLP (1x100)                            97.78%     97.76%     97.78%    440 / 450
```

### Table 3: HEAD-TO-HEAD

```
   best retrieval method  : brute force 1-NN
     accuracy             : 98.44%
     fit time             : 70 µs
     p50 inference        : 14.9 µs

   best trained classifier: SVM (RBF)
     accuracy             : 99.11%
     fit time             : 13.34 ms
     p50 inference        : 64.9 µs

   retrieval − classifier : −0.67 accuracy points, fit time 191× faster
```

---

## How to read the output

**The hash row has 0% accuracy and that's the whole point.**

Hash exact match means "byte-identical lookup." Test samples are by construction **not** identical to anything in the train set — that's what makes them held-out. So the hash never hits in the accuracy test. The hash is the **cache** in front of the index, not the index itself. Its purpose is to short-circuit repeat queries in production at sub-microsecond cost. The hybrid row shows you what the system actually does: hash first, fall through to 1-NN on miss.

**1-NN retrieval is competitive with trained classifiers.**

You'll typically see something like:

| Method | Accuracy | Fit time | Notes |
|---|---:|---:|---|
| Brute 1-NN | ~98.4% | <1 ms | No training |
| LogisticRegression | ~96% | ~50 ms | Linear |
| Random Forest | ~96-97% | ~120 ms | 100 trees |
| MLP | ~97-98% | ~300 ms | Small NN |
| SVM (RBF) | ~99% | ~15 ms | Highest accuracy |

Retrieval ties or beats most trained baselines on this dataset, and the only thing that beats it (SVM) trains ~200× slower. That's the bargain retrieval offers: train in microseconds, get near-state-of-the-art accuracy.

**Trees vs. brute force at this scale.**

KDTree and BallTree are *slower* than brute force here because the dataset is small (1347 train) and high-dimensional (64). The tree-traversal overhead doesn't pay off until N is much larger. With 100k+ samples the trees pull ahead. Always benchmark.

**The C version matches Python accuracy exactly.**

Both use the same Euclidean 1-NN; both find the same nearest neighbor; both predict the same label. Accuracy is 98.44% in both. The C version is faster per prediction (~30 µs vs ~35 µs Python hybrid) because there's no interpreter overhead.

---

## Files in this folder

| File | What it is |
|---|---|
| `bench.py` | Python benchmark — 7 retrieval methods + 5 trained classifiers. |
| `bench.c` | C benchmark — hash + brute-force fallback, reads the train/test CSVs. |
| `compare.py` | Cross-language comparison. |
| `run_all.sh` | One-command driver. |
| `digits_train.csv` | Train split (1347 rows). Written by `bench.py`, used by `bench.c`. |
| `digits_test.csv`  | Test  split ( 450 rows). Written by `bench.py`, used by `bench.c`. |
| `results_python.json` | Machine-readable Python results. |
| `results_c.json`      | Machine-readable C results. |
| `HOWTO.md` | This file. |

---

## Running parts individually

```bash
# Python only
python3 bench/bench.py
python3 bench/bench.py --seed 7 --test-size 0.20

# C only (needs the CSVs from Python first)
python3 bench/bench.py >/dev/null      # writes digits_train.csv + digits_test.csv
gcc -O3 -Wall -Wextra -o bench/bench bench/bench.c -lm
./bench/bench

# Comparison (after both have run)
python3 bench/compare.py
```

---

## Caveats and notes

- **Clock granularity.** `clock_gettime(CLOCK_MONOTONIC)` on macOS resolves to ~1 µs for the tight loop we run. The C per-query medians for very fast operations round to whole microseconds. Totals (throughput) are exact.
- **Quiet machine matters.** Close Chrome, Slack, Spotify. Background work jitters tail latencies. A clean run is one where p99 ≈ 2-3× the median; if p99 is 100× the median, something else grabbed the CPU.
- **Memory readings can be negative.** Python GC may free memory between runs. A `-0.05 MB` is not a bug.
- **Why the hash has 0% accuracy** — see the section above. It's the cache, not the classifier.
- **Why brute force is sometimes faster than KDTree** — at N=1347 in 64-D, the tree-traversal constants swamp the O(log n) savings. The trees win convincingly at N > 50k.

---

## What to do next

A few experiments worth running:

```bash
# How does accuracy change with the split seed?
for S in 0 1 7 42 123; do
  ./bench/run_all.sh $S 2>&1 | grep -E "brute force 1-NN|SVM \(RBF\)|Random Forest" | head -3
done

# Which classes does 1-NN struggle on?
python3 - <<'PY'
import json, pandas as pd
from collections import defaultdict
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
d = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(
    d.data, d.target, test_size=0.25, random_state=42, stratify=d.target)
tree = BallTree(X_tr)
_, idx = tree.query(X_te, k=1)
pred = y_tr[idx[:, 0]]
cm = pd.crosstab(pd.Series(y_te, name="true"), pd.Series(pred, name="pred"))
print(cm)
PY

# How fast can you push the C version?
./bench/bench --train bench/digits_train.csv --test bench/digits_test.csv

# Plot timing vs accuracy
python3 - <<'PY'
import json, matplotlib.pyplot as plt
data = json.load(open("bench/results_python.json"))
for r in data["results"]:
    plt.scatter(r["per_query"]["median_us"], r["accuracy"]*100, s=80)
    plt.annotate(r["name"], (r["per_query"]["median_us"], r["accuracy"]*100),
                 fontsize=8, alpha=0.7)
plt.xscale("log"); plt.xlabel("p50 inference (µs)")
plt.ylabel("accuracy (%)"); plt.title("Accuracy vs. latency")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("bench/accuracy_vs_latency.png", dpi=140)
print("saved bench/accuracy_vs_latency.png")
PY
```

That's everything. Run `./bench/run_all.sh`, read the three tables, draw your own conclusions about whether retrieval or a trained classifier is the right tool for your problem.
