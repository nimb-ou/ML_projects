# LinkedIn Project Entry

*Paste this into the "Projects" section of your LinkedIn profile. Headline + description.*

---

**Project name**

Digit Memory — A Memory-First Approach to Classification

**Associated with**

(your education / role)

**Description**

A study in retrieval as classification, built in three implementations for the classic sklearn digits dataset.

The system stores every training sample and indexes it two ways: an O(1) hash map keyed on the byte representation of the feature vector (for exact match), and an O(log n) tree (KDTree / BallTree) for nearest neighbor when no exact match exists. No model is trained. No gradients are computed. Queries return in microseconds.

Three implementations of the same system:

- A production-style Python script with SQLite persistence and an sklearn-based nearest-neighbor index.
- A pure C implementation with a hand-rolled open-addressed FNV-1a hash table and brute-force linear scan — zero external dependencies, fully self-contained after the dataset is exported once.
- A 43-cell Jupyter study notebook that builds the system from scratch with pandas DataFrames, benchmarks across three NN backends, and a held-out stress test.

Key results:

- 98.4 percent held-out accuracy on a stratified 75/25 train/test split, with no training step.
- ~3 microsecond exact lookup, ~30–100 microsecond nearest-neighbor fallback on an M-class CPU.
- Graceful accuracy degradation under injected Gaussian noise — from 98.4 percent at sigma = 0 to 67 percent at sigma = 10, plotted as a robustness curve.
- Cross-language consistency: Python and C return identical labels for the same query.

The project demonstrates that retrieval, done right, deserves consideration as a baseline before reaching for a trained model — and that implementing the same system at three levels of abstraction is one of the fastest ways to actually understand it.

**Skills**

Python, C, NumPy, scikit-learn, pandas, Jupyter, SQLite, data structures, nearest-neighbor search, benchmarking, technical writing

**Link**

(your GitHub URL)
