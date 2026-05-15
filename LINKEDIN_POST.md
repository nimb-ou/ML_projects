# LinkedIn Post

*The share-post. Designed for a hook in the first line, a payoff in the middle, and a soft call-to-action at the end. Plain text — paste straight into LinkedIn.*

---

The fastest model is the one you don't need.

I built a digit recognizer for the classic sklearn digits dataset using zero training. No gradient descent. No epochs. No weights to save.

Just a hash map and a nearest-neighbor tree.

98.4 percent held-out accuracy. Three microseconds per query when the answer is memorized. Fifty when it isn't.

Most ML projects start with "what model should we use." This one started with a different question: what if you just stored the data, organized it well, and looked things up?

Turns out — for a dataset this size — that's enough.

I wrote it three times to actually understand it:

— A production-style Python script with SQLite persistence and sklearn's KDTree.

— A pure C implementation. No dependencies. Hand-rolled open-addressed hash table with FNV-1a hashing. Brute-force nearest neighbor in 30 lines. Reads a CSV the Python version exports.

— A 43-cell Jupyter notebook that builds the whole system from scratch with pandas DataFrames, three different nearest-neighbor backends benchmarked side by side, and a noise-robustness stress test.

A few things I learned that I'll carry forward:

1. Retrieval is a criminally underused baseline. Before tuning a model, ask whether you can just remember the answer. When you can, the "model" is the data.

2. The smartest data structure is matched to the query distribution. Exact lookups should hit a hash. Approximate lookups should hit a tree. Most production systems pretend they only need one.

3. Tree-based nearest neighbor isn't always faster than brute force. At 1797 samples and 64 dimensions, vectorized numpy beat KDTree and BallTree. The asymptotics flip at larger N. Benchmark before you assume.

4. `array.tobytes()` is the right hash key for numpy arrays. `tuple(array)` is slower. The difference is measurable.

5. Writing the same system in three languages compounds your understanding. Python lets you build it. C makes you understand it.

The full project — code, write-up, study-guide notebook, and a robustness curve from sigma=0 to sigma=10 — is in the repo linked below.

If you've ever trained a model on a problem that could've been a lookup table, this one might be worth fifteen minutes.

---

#MachineLearning #DataStructures #Python #C #LearningInPublic
