# Digit Memory

A memory-based digit recognizer for the classic `sklearn` digits dataset. No model training — every sample is stored, indexed two ways, and looked up. Three implementations: Python, Jupyter, and pure C.

**Held-out accuracy: 98.4%.** **Query latency: ~3 µs (exact) / ~50–100 µs (nearest neighbor).** No epochs, no gradients, no GPU.

---

## What it does

Given an 8×8 grayscale digit image (64 floats), return the digit.

- **Exact match** via an in-memory hash map keyed on the byte representation of the feature vector. O(1).
- **Nearest neighbor** fallback when no exact match exists — KDTree / BallTree in Python, brute-force scan in C. O(log n) average for the tree, O(n) for brute force (still microseconds at this scale).

That's the whole system. The trick is doing the right thing in the right order: try the cheap lookup first, fall back to the expensive one only when needed.

## Repo contents

### Code

| File | What it is |
|---|---|
| [`digits_memory.py`](digits_memory.py) | Production-style Python implementation. Downloads the dataset once, persists to SQLite, builds the in-memory index, runs demo queries. |
| [`digits_memory.c`](digits_memory.c) | Pure C version. No dependencies. Open-addressed hash table for exact match + brute-force linear scan for NN. Reads the CSV exported by the Python version. |
| [`digits_memory.ipynb`](digits_memory.ipynb) | Lean notebook mirroring the Python script. Useful for quick interactive exploration. |
| [`digits_study_guide.ipynb`](digits_study_guide.ipynb) | A 43-cell project notebook that builds the system from scratch with pandas DataFrames, plots, benchmarks, and a held-out accuracy stress test. Read this if you want to *learn* the project rather than just run it. |

### Written content

| File | What it is |
|---|---|
| [`BLOG.md`](BLOG.md) | The full write-up of the design and what I learned. |
| [`LINKEDIN_PROJECT.md`](LINKEDIN_PROJECT.md) | Short polished project description for a LinkedIn profile. |
| [`LINKEDIN_POST.md`](LINKEDIN_POST.md) | The share-post for LinkedIn. |

### Designed assets (v2)

| File | What it is |
|---|---|
| [`design/digit_memory_case_study.pdf`](design/digit_memory_case_study.pdf) | Eight-page designed case study: cover, hypothesis, architecture, numbers, dataset gallery, robustness, three implementations, colophon. Built with the design philosophy in `design/DESIGN_PHILOSOPHY.md` ("Computational Quiet"). |
| [`design/digit_memory_poster.pdf`](design/digit_memory_poster.pdf) | Single-page tabloid poster with the headline numbers, two-tier diagram, and digit gallery. |
| [`design/digit_memory_deck.pptx`](design/digit_memory_deck.pptx) | 8-slide widescreen deck for verbal presentation. Mirrors the case study sections. |
| [`design/digit_memory_case_study.docx`](design/digit_memory_case_study.docx) | Word version of the case study for handoff to non-technical reviewers. |
| [`design/architecture.png`](design/architecture.png) | Standalone architecture diagram (1600x1000). Embedded in the case study and slide deck. |
| [`design/DESIGN_PHILOSOPHY.md`](design/DESIGN_PHILOSOPHY.md) | The visual philosophy behind the assets. |
| [`design/generate.py`](design/generate.py) · [`design/generate_docs.py`](design/generate_docs.py) | Build scripts. `python3 design/generate.py && python3 design/generate_docs.py` rebuilds every asset. |

## Quick start

### Python

```bash
pip install -r requirements.txt
python3 digits_memory.py
```

First run downloads the dataset and writes `digits.db` (SQLite) and `digits.csv` (for the C build). Subsequent runs are fully offline.

Expected output:

```
Loaded 1797 samples into memory.
Exact query     -> label=1  dist=0.0000  exact=True  (3.5 us)
Perturbed query -> label=1  dist=1.4765  exact=False  (95.9 us)
Noisy query     -> label=4  dist=14.6307  exact=False  (50.1 us)
```

### Jupyter

```bash
jupyter notebook digits_study_guide.ipynb
```

Run top to bottom. Each section builds on the previous one — load, explore, build the memory, stress test.

### C

```bash
gcc -O2 -o digits_memory digits_memory.c -lm
./digits_memory          # reads digits.csv from cwd
```

The C build needs the CSV that the Python version exports on its first run. After that, the C binary is fully self-contained.

## The two-tier lookup, in pictures

```
                          query vector
                                |
                                v
                     +----------+----------+
                     |  hash map (bytes)   |
                     |  O(1) exact match   |
                     +----------+----------+
                                |
                  hit ----------+---------- miss
                   |                          |
                   v                          v
              return label             +-----+-----+
                                       |  KDTree / |
                                       |  BallTree |
                                       |  k=1      |
                                       +-----+-----+
                                             |
                                             v
                                     return NN's label
```

## Numbers

Held-out accuracy on a 75/25 train/test split, stratified by class:

| Noise sigma (pixel units) | Accuracy |
|---:|---:|
| 0.0 | 98.4% |
| 1.0 | 98.2% |
| 3.0 | 97.3% |
| 5.0 | 93.8% |
| 10.0 | 67.1% |

Speed comparison on 200 random queries (M-class CPU):

| Method | µs / query |
|---|---:|
| Hash map (exact) | ~3 |
| Brute force NN | ~30 |
| KDTree NN | ~50 |
| BallTree NN | ~50 |

KDTree and BallTree are tree-search heuristics; their overhead doesn't pay off until N is much larger. With 1797 samples in 64 dimensions, vectorized brute force is competitive.

## License

MIT — see [LICENSE](LICENSE).
