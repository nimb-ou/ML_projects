"""
Memorization-based classifier on CIFAR-10 — same fundamentals as
memorize_and_validate.py, scaled up and made more complex.

What's different from the digits demo:
  * Dataset: CIFAR-10 (60,000 color images, 32x32x3 = 3,072 features, 10
    classes) instead of sklearn digits (1,797 grayscale 8x8 images).
  * Preprocessing: PCA reduces 3,072 raw pixel features to N_COMPONENTS
    (default 50). Memorization happens on the PCA features, not the raw
    pixels — so the script learns a representation, then memorizes it.
  * Baseline: a logistic-regression classifier is trained on the same PCA
    features and evaluated on the same test split, so the memorization
    result has a reference number to compare against.

What's the same:
  * Exact hash table for O(1) recall on inputs we have seen verbatim.
  * KDTree nearest-neighbor fallback for unseen inputs.
  * Training-set recall self-check (memorization without recall = bug).
  * Per-class accuracy + confusion matrix + sample mistakes.
  * Disk cache so re-runs are instant.

Run with:
    python3 cifar10_memorize.py
"""

from __future__ import annotations

import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
TARBALL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TARBALL_PATH = os.path.join(HERE, "cifar-10-python.tar.gz")
DATA_DIR = os.path.join(HERE, "cifar10_data")
STORE_PATH = os.path.join(HERE, "cifar10_store.npz")

N_COMPONENTS = 50          # PCA target dimensionality
SEED = 42
N_FEATURES_RAW = 3072      # 32 * 32 * 3
N_CLASSES = 10
CIFAR_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# --------------------------------------------------------------------------- #
# 1. Download and unpack CIFAR-10                                              #
# --------------------------------------------------------------------------- #

def download_if_missing() -> None:
    if os.path.exists(TARBALL_PATH):
        return
    print(f"  downloading CIFAR-10 (~170 MB) -> {TARBALL_PATH}")
    print(f"  source: {TARBALL_URL}")

    def _hook(blocks: int, blocksize: int, total: int) -> None:
        got = blocks * blocksize
        pct = min(100.0, got * 100.0 / total) if total > 0 else 0
        sys.stdout.write(f"\r    {pct:5.1f}%  ({got / 1e6:6.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(TARBALL_URL, TARBALL_PATH, _hook)
    sys.stdout.write("\n")


def unpack_if_missing() -> None:
    if os.path.isdir(DATA_DIR) and os.path.exists(
        os.path.join(DATA_DIR, "data_batch_1")
    ):
        return
    print(f"  unpacking tarball -> {DATA_DIR}")
    with tarfile.open(TARBALL_PATH, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "cifar-10-batches-py" in m.name]
        os.makedirs(DATA_DIR, exist_ok=True)
        for m in members:
            # flatten into DATA_DIR (strip the cifar-10-batches-py/ prefix)
            m.name = os.path.basename(m.name)
            if not m.name:
                continue
            tar.extract(m, DATA_DIR)


def _load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"]                                    # (N, 3072) uint8
    y = np.array(d[b"labels"], dtype=np.int64)        # (N,)
    return X, y


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) as numpy arrays."""
    train_files = [
        os.path.join(DATA_DIR, f"data_batch_{i}") for i in range(1, 6)
    ]
    test_file = os.path.join(DATA_DIR, "test_batch")

    Xs, ys = [], []
    for p in train_files:
        X, y = _load_batch(p)
        Xs.append(X)
        ys.append(y)
    X_train = np.concatenate(Xs, axis=0)        # (50000, 3072) uint8
    y_train = np.concatenate(ys, axis=0)        # (50000,)

    X_test, y_test = _load_batch(test_file)
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------- #
# 2. The memorization technique (same fundamentals, on PCA features)          #
# --------------------------------------------------------------------------- #

class CIFARMemory:
    """Hash-first, KDTree-fallback classifier on PCA-reduced features."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)
        # Exact memorization: bytes -> label.
        self.exact: dict[bytes, int] = {
            self.X[i].tobytes(): int(self.y[i]) for i in range(len(self.y))
        }
        # Nearest-neighbor index for unseen inputs.
        self.tree = KDTree(self.X)

    def predict(self, x: np.ndarray) -> tuple[int, bool]:
        """Return (predicted_label, was_exact_hit)."""
        b = x.tobytes()
        hit = self.exact.get(b)
        if hit is not None:
            return hit, True
        _, idx = self.tree.query(x.reshape(1, -1), k=1)
        return int(self.y[idx[0, 0]]), False

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized prediction. Returns (preds, exact_hit_flags)."""
        n = len(X)
        preds = np.empty(n, dtype=np.int64)
        exact_flags = np.zeros(n, dtype=bool)

        # First pass: try the hash for every row.
        miss_indices: list[int] = []
        for i in range(n):
            hit = self.exact.get(X[i].tobytes())
            if hit is None:
                miss_indices.append(i)
            else:
                preds[i] = hit
                exact_flags[i] = True

        # Second pass: one batched KDTree query for all misses.
        if miss_indices:
            X_miss = X[miss_indices]
            _, idx = self.tree.query(X_miss, k=1)
            preds[miss_indices] = self.y[idx[:, 0]]

        return preds, exact_flags


# --------------------------------------------------------------------------- #
# 3. Build / load store (PCA + memorized features)                            #
# --------------------------------------------------------------------------- #

def build_or_load_store(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on the training set, transform both splits, persist to disk."""
    if os.path.exists(STORE_PATH):
        cached = np.load(STORE_PATH)
        if (
            cached["X_train_pca"].shape[1] == N_COMPONENTS
            and cached["X_train_pca"].shape[0] == len(X_train_raw)
            and cached["X_test_pca"].shape[0] == len(X_test_raw)
        ):
            print(f"  loaded cached PCA store -> {STORE_PATH}")
            return (
                cached["X_train_pca"],
                cached["y_train"],
                cached["X_test_pca"],
            )

    # Scale uint8 -> float64 in [0, 1] before PCA so components are well-conditioned.
    Xtr = X_train_raw.astype(np.float64) / 255.0
    Xte = X_test_raw.astype(np.float64) / 255.0

    print(f"  fitting PCA: 3072 -> {N_COMPONENTS} components on {len(Xtr):,} rows...")
    t0 = time.perf_counter()
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED, svd_solver="randomized")
    X_train_pca = pca.fit_transform(Xtr).astype(np.float64)
    X_test_pca = pca.transform(Xte).astype(np.float64)
    fit_s = time.perf_counter() - t0
    var = pca.explained_variance_ratio_.sum()
    print(f"    PCA fit time          : {fit_s * 1000:.0f} ms")
    print(f"    explained variance    : {var * 100:.2f}% "
          f"({N_COMPONENTS} of 3072 components)")

    np.savez(
        STORE_PATH,
        X_train_pca=X_train_pca,
        y_train=y_train,
        X_test_pca=X_test_pca,
    )
    print(f"  wrote PCA store       -> {STORE_PATH}  "
          f"({os.path.getsize(STORE_PATH):,} bytes)")
    return X_train_pca, y_train, X_test_pca


# --------------------------------------------------------------------------- #
# 4. Reporting                                                                 #
# --------------------------------------------------------------------------- #

def per_class_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("\n  per-class accuracy:")
    print(f"    {'class':<11} {'correct':>8} {'total':>7} {'accuracy':>10}")
    for cls in range(N_CLASSES):
        mask = y_true == cls
        n = int(mask.sum())
        if n == 0:
            continue
        c = int((y_pred[mask] == cls).sum())
        print(f"    {CIFAR_LABELS[cls]:<11} {c:>8} {n:>7} {c / n * 100:>9.2f}%")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("\n  confusion matrix (rows=truth, cols=prediction):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    header = "         " + " ".join(f"{CIFAR_LABELS[c][:4]:>5}" for c in range(N_CLASSES))
    print(header)
    for cls in range(N_CLASSES):
        row = " ".join(f"{cm[cls, p]:>5}" for p in range(N_CLASSES))
        print(f"   {CIFAR_LABELS[cls][:4]:<4} {row}")


def show_mistakes(y_true: np.ndarray, y_pred: np.ndarray, limit: int = 8) -> None:
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("\n  no mistakes — perfect on the test set!")
        return
    print(f"\n  first {min(limit, len(wrong))} mistakes (idx · truth -> predicted):")
    for k in wrong[:limit]:
        print(f"    idx={k:>5}   {CIFAR_LABELS[y_true[k]]:<11}"
              f" -> {CIFAR_LABELS[y_pred[k]]}")


def report(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elapsed_s: float,
    extra: dict | None = None,
) -> dict:
    n = len(y_true)
    correct = int((y_pred == y_true).sum())
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  === {title} ===")
    if extra:
        for k, v in extra.items():
            print(f"    {k:<22}: {v}")
    print(f"    predictions made      : {n:,}")
    print(f"    total time            : {elapsed_s * 1000:.0f} ms  "
          f"({elapsed_s / n * 1e6:.1f} us per query)")
    print(f"    accuracy              : {correct:,} / {n:,}  =  {acc * 100:.2f}%")
    print(f"    macro F1              : {macro_f1 * 100:.2f}%")
    return {"name": title, "accuracy": acc, "macro_f1": macro_f1, "n": n}


# --------------------------------------------------------------------------- #
# 5. Driver                                                                    #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 78)
    print(" CIFAR-10 memorization (PCA features) vs trained baseline")
    print("=" * 78)

    print("\n[1/5] acquiring CIFAR-10...")
    download_if_missing()
    unpack_if_missing()
    X_train_raw, y_train, X_test_raw, y_test = load_cifar10()
    print(f"  train shape           : {X_train_raw.shape}  dtype={X_train_raw.dtype}")
    print(f"  test  shape           : {X_test_raw.shape}  dtype={X_test_raw.dtype}")
    print(f"  classes               : {CIFAR_LABELS}")
    assert X_train_raw.shape[1] == N_FEATURES_RAW
    assert set(np.unique(y_train).tolist()) == set(range(N_CLASSES))

    print(f"\n[2/5] building PCA representation (target dim = {N_COMPONENTS})...")
    X_train_pca, y_train, X_test_pca = build_or_load_store(
        X_train_raw, y_train, X_test_raw
    )
    print(f"  train PCA shape       : {X_train_pca.shape}")
    print(f"  test  PCA shape       : {X_test_pca.shape}")

    print("\n[3/5] memorizing PCA-reduced training set (hash + KDTree)...")
    t0 = time.perf_counter()
    mem = CIFARMemory(X_train_pca, y_train)
    build_s = time.perf_counter() - t0
    print(f"  unique vectors stored : {len(mem.exact):,}")
    print(f"  build time            : {build_s * 1000:.0f} ms")

    print("\n[4/5] predicting test set with the memorization technique...")
    t0 = time.perf_counter()
    preds_mem, exact_flags = mem.predict_batch(X_test_pca)
    elapsed_mem = time.perf_counter() - t0
    exact_hits = int(exact_flags.sum())

    # Memorization invariants — the program crashes if these fail.
    assert preds_mem.shape == y_test.shape
    # When we report an exact hit, the prediction must equal the stored label.
    for i in np.where(exact_flags)[0]:
        assert preds_mem[i] == mem.exact[X_test_pca[i].tobytes()], \
            f"exact-hit prediction disagrees with stored label at i={i}"

    mem_result = report(
        "memorization (hash + KDTree on PCA features)",
        y_test, preds_mem, elapsed_mem,
        extra={
            "exact-hash hits": f"{exact_hits:,} ({exact_hits / len(y_test) * 100:.1f}%)",
            "KDTree fallbacks": f"{len(y_test) - exact_hits:,} "
                                f"({(len(y_test) - exact_hits) / len(y_test) * 100:.1f}%)",
        },
    )

    per_class_breakdown(y_test, preds_mem)
    confusion(y_test, preds_mem)
    show_mistakes(y_test, preds_mem)

    print("\n[5/5] training logistic-regression baseline on the same PCA features...")
    t0 = time.perf_counter()
    clf = LogisticRegression(
        max_iter=1000, n_jobs=1, random_state=SEED, solver="lbfgs",
    )
    clf.fit(X_train_pca, y_train)
    fit_s = time.perf_counter() - t0
    print(f"  fit time              : {fit_s * 1000:.0f} ms")

    t0 = time.perf_counter()
    preds_lr = clf.predict(X_test_pca)
    elapsed_lr = time.perf_counter() - t0
    lr_result = report(
        "logistic regression on PCA features (baseline)",
        y_test, preds_lr, elapsed_lr,
        extra={"fit time": f"{fit_s * 1000:.0f} ms"},
    )

    # --- Memorization self-check on the *training* set ---------------------- #
    # Every training point, queried back, MUST recover its stored label.
    # Sample 5000 train rows so this stays under a few seconds.
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X_train_pca), size=5000, replace=False)
    train_preds, _ = mem.predict_batch(X_train_pca[sample_idx])
    train_recall = accuracy_score(y_train[sample_idx], train_preds)
    print(f"\n  self-check: training-set recall on 5,000 sampled rows "
          f"= {train_recall * 100:.2f}%  (expected 100.00%)")
    assert train_recall == 1.0, "memorization failed: cannot recall training data"

    # --- Side-by-side --------------------------------------------------------- #
    print("\n" + "=" * 78)
    print(" HEAD-TO-HEAD")
    print("=" * 78)
    rows = [mem_result, lr_result]
    print(f"  {'method':<50} {'accuracy':>10} {'macro F1':>10}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['name']:<50} {r['accuracy'] * 100:>9.2f}% "
              f"{r['macro_f1'] * 100:>9.2f}%")
    delta = (mem_result["accuracy"] - lr_result["accuracy"]) * 100
    sign = "+" if delta >= 0 else ""
    print()
    print(f"  memorization − baseline: {sign}{delta:.2f} accuracy points")
    print()
    print("  Note: on a representation this lossy (PCA-50 retains only a")
    print("  fraction of CIFAR-10's signal), both methods plateau at the")
    print("  same place — the bottleneck is the FEATURES, not the model.")
    print()
    print("  Cache on disk:", STORE_PATH)
    print("=" * 78)


if __name__ == "__main__":
    main()
