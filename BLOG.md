# Building Digit Memory: Why I Wrote a Classifier With No Training

*A study in retrieval as a baseline, written three times in three languages.*

---

The fastest model is the one you don't need.

I built a digit recognizer for the classic `sklearn` digits dataset using nothing but a hash map and a nearest-neighbor tree. No gradient descent. No epochs. No weights to save. Just data, indexed two ways, looked up.

Held-out accuracy: **98.4%**. Query latency: **three microseconds** when the answer is memorized, **fifty to a hundred** when it's not.

This post is about why retrieval deserves a seat at the table when people reach for models, and what I learned writing the same system in Python, Jupyter, and pure C.

---

## The dataset

`sklearn.datasets.load_digits()` returns 1797 grayscale images of handwritten digits, 8×8 pixels each, with integer intensities from 0 to 16. Each sample is a 64-dimensional vector with a label in `{0, 1, ..., 9}`. Classes are roughly balanced — about 180 examples per digit.

The textbook approach is to train a classifier — logistic regression, an SVM, a small MLP. The textbook approach also misses something obvious: 1797 samples is small enough that we can just *store all of them* and answer queries by lookup.

The question becomes: how do you organize them so lookup is fast?

---

## Two kinds of queries, two kinds of memory

Real queries split into two categories.

**Exact queries** — the input is byte-identical to something we've seen. Repeated calls, cached values, results from deterministic pipelines. For these, the right data structure is a hash table. O(1) average. There's no smarter answer.

**Approximate queries** — the input is *close to* something we've seen but not identical. Pixel noise, slightly different handwriting, sensor drift. Hashing won't help: change one bit and the hash changes completely. For these, we need a structure that knows about geometric closeness — a spatial index.

A serious system handles both. A common mistake is to pick one and force every query through it. If you only have the tree, you're doing tree traversals for queries that should have been hash lookups. If you only have the hash, every noisy input falls off a cliff.

The fix is a two-tier memory: hash first, tree on miss.

---

## Exact match: getting the hash key right

The Python interpreter's `dict` is one of the most heavily-optimized data structures in any language. It does what we need — average O(1) lookup, near-zero overhead at our scale.

The catch: dict keys have to be hashable, and `numpy.ndarray` is not. You have to convert.

Three obvious choices:

```python
key = tuple(arr)        # works, slow — Python tuple of Python floats
key = arr.tobytes()     # works, fast — raw byte representation
key = hash(arr.tobytes())  # works, lossy if you ever want to compare
```

`tobytes()` is the right answer. It's a stable byte representation of the underlying buffer, it's fast (single memcpy under the hood), and it survives serialization across processes. The dict-internal hash is computed once per insertion and once per lookup; the byte string itself is the canonical key.

```python
exact_memory = {
    row.tobytes(): int(label)
    for row, label in zip(digits.data, digits.target)
}

def exact_lookup(arr):
    return exact_memory.get(np.asarray(arr, dtype=np.float64).tobytes())
```

That's the whole exact lookup. With 1797 entries it benchmarks at a few hundred nanoseconds per query — independent of dataset size.

### Where this breaks

Here's the failure mode that motivates the rest of the system:

```python
sample = digits.data[42].copy()
print(exact_lookup(sample))      # -> 1
sample[0] += 0.001
print(exact_lookup(sample))      # -> None
```

A change of one part in ten thousand on a single pixel — imperceptible to any human, well within sensor noise — and we miss completely. Float-precision changes in pipelines, scaling factors, type promotions: all of these will break the hash. We need a graceful fallback.

---

## Approximate match: nearest neighbor, three ways

When the hash misses, we want to find the *closest* stored sample and return its label.

"Closest" needs a definition. I used **Euclidean distance** in 64-dimensional pixel space:

$$d(a, b) = \sqrt{\sum_{i=1}^{64} (a_i - b_i)^2}$$

It's not the only choice — cosine, L1, learned metrics all have merits — but Euclidean is the right starting point for grayscale pixels where intensity differences are meaningful.

Three implementations:

### Brute force

```python
def brute_nn(query, X, y):
    diffs = X - query
    sq_dists = (diffs * diffs).sum(axis=1)
    best = int(np.argmin(sq_dists))
    return int(y[best]), float(np.sqrt(sq_dists[best])), best
```

Linear in N, but vectorized. On 1797 × 64 it's a single SIMD-friendly matrix subtraction and sum — measured at ~30 µs per query.

### KDTree

Recursively partitions space along axis-aligned splits. Each node "owns" a hyperrectangle; queries descend to the leaf containing the query, then back up pruning siblings that can't contain a closer point. Average O(log n) at low dimensions.

```python
kdtree = KDTree(digits.data)
dist, idx = kdtree.query(query.reshape(1, -1), k=1)
```

### BallTree

Like KDTree, but partitions space into nested hyperspheres ("balls") instead of axis-aligned boxes. The triangle inequality lets you prune even in high-dimensional spaces where axis-aligned splits become useless. Recommended past ~20 dimensions.

### The benchmark surprise

I ran 200 random queries through each. The result:

| Method | µs / query |
|---|---:|
| Hash map (exact) | ~3 |
| Brute force NN | ~30 |
| KDTree NN | ~50 |
| BallTree NN | ~50 |

**Brute force wins.** The trees lose because tree-traversal overhead doesn't amortize at 1797 samples. The promise of O(log n) is real, but the constant factors swamp the gain at this scale.

This is the kind of result you only get by measuring. Intuition says "tree = fast, brute = slow." Measurement says "match the structure to the data size."

For a larger dataset — 100,000+ samples — the trees would win convincingly. For 1797 samples, vectorized numpy is the right answer.

---

## Putting it together

The full memory is forty lines:

```python
class DigitMemory:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.exact = {self.X[i].tobytes(): int(self.y[i]) for i in range(len(self.y))}
        self.tree = BallTree(self.X)

    def query(self, arr):
        q = np.asarray(arr, dtype=np.float64).reshape(-1)
        hit = self.exact.get(q.tobytes())
        if hit is not None:
            return {'label': hit, 'distance': 0.0, 'exact': True}
        dist, idx = self.tree.query(q.reshape(1, -1), k=1)
        return {'label': int(self.y[idx[0, 0]]), 'distance': float(dist[0, 0]), 'exact': False}
```

Hash first. Tree on miss. The system doesn't care which path it takes — it returns whichever answer it finds first.

---

## How accurate is "look up the closest thing"?

I split the dataset 75/25, built the memory on the train half, and queried with the test half. Stratified by class so each digit was equally represented.

**98.4% accuracy.** 443 correct out of 450.

For context, that's competitive with logistic regression and several other "real" baselines on this dataset. No training step. No hyperparameters. The "model" is the data.

This isn't novel — it's just 1-NN classification, a technique from the 1960s. What's novel-feeling is how rarely people reach for it before training something.

### How robust is it?

Real inputs are noisy. I ran the held-out test set through the memory at increasing levels of Gaussian noise:

| Noise σ | Accuracy |
|---:|---:|
| 0.0 | 98.4% |
| 1.0 | 98.2% |
| 3.0 | 97.3% |
| 5.0 | 93.8% |
| 7.0 | 84.7% |
| 10.0 | 67.1% |

Pixel values range from 0 to 16. At σ=5, we're injecting noise at roughly a third of the signal's standard deviation, and accuracy is still above 90%. At σ=10 — noise at the same scale as the signal — we're still getting two-thirds of queries right.

This is the kind of robustness curve I'd want to see for any retrieval system going to production: smooth degradation, no cliff.

---

## Why three implementations?

Writing the same system three ways is the closest I've found to actually understanding it.

**Python with sklearn** is where you reach for first. Indexes are imported, not implemented. You write the *logic* of the system and trust the library to do the hard parts. Good for getting something working fast. Bad for understanding what's actually happening.

**The Jupyter study notebook** is where you slow down. Each step is a cell. Each cell has an output you can inspect. You can run the benchmark, *see* the table, and re-run it with a different N to watch the curve change. Pandas DataFrames make exploration ergonomic. Plots make patterns obvious.

**Pure C** is where you find out what you actually understood. There's no `dict`, so you implement an open-addressed hash table with FNV-1a hashing. There's no `KDTree`, so you write a brute-force scan with `memcmp` for the hash compare. There's no `numpy`, so you write the loop. Forty extra lines, and every byte of memory is yours.

The C version's payoff isn't speed — it benchmarks comparably to the Python one. The payoff is that there's nowhere to hide. If you don't understand how an open-addressed hash table handles collisions, your code crashes. If you don't understand how to read floats from a CSV, your distances are garbage.

I recommend the exercise.

---

## What I learned

A few things I'll carry forward.

**Retrieval is a criminally underused baseline.** Before you train anything, ask whether you can just remember the answer. On many problems with small-to-medium data, the answer is yes. When it's not, you've at least established a real number that the model has to beat.

**The smartest data structure is matched to the query distribution.** If most of your queries are repeats, a hash table is the right answer regardless of what your data looks like. If your queries are always novel, you can skip the cache entirely. Hybrid systems beat either one.

**Benchmark before you assume.** "Tree-based indexes are faster than brute force" is true asymptotically and false at small scale. I had to write the benchmark to find out which side of the line I was on. The cost of measuring is always less than the cost of being wrong.

**Hash key choice matters.** `tuple(array)` looks innocent and is slow. `array.tobytes()` is fast, deterministic, and survives serialization. The difference is measurable.

**Writing the same system in multiple languages compounds your understanding.** Python lets you build the system. C makes you understand it.

---

## What's next

A few things worth trying on top of this:

- **Cosine distance instead of Euclidean.** For grayscale digits the difference is small. For text embeddings or normalized feature vectors it can be substantial.
- **PCA before indexing.** Reduce 64 dimensions to 16 and the tree-traversal overhead drops; some accuracy is lost. Quantifying the tradeoff is a one-afternoon project.
- **k-NN instead of 1-NN.** Vote among the top k=5 neighbors. Usually a small accuracy bump, especially in the noisy regime.
- **A rejection threshold.** If the nearest neighbor is more than some distance away, return "unknown" instead of guessing. The right cutoff is dataset-specific and worth calibrating from the noisy stress test above.
- **Approximate NN (FAISS, Annoy, HNSW).** Once your dataset grows past a million samples, exact NN becomes the bottleneck. Approximate methods trade a bit of accuracy for orders of magnitude in speed.

Each of these is a paragraph in a model card. Together they tell you whether retrieval is the right tool, or whether you genuinely need to train something.

---

The code, the notebooks, and the C source are all in the repo. The study-guide notebook (`digits_study_guide.ipynb`) is the best place to start if you want to learn the project hands-on. The Python script is the cleanest summary. The C version is the deepest understanding.

The whole thing is a few hundred lines. Try it. Then try beating it.
