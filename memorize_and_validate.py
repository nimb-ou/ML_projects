"""
Memorization-based classifier on the sklearn digits dataset.

What this script does (end to end, in one file):
  1. Downloads the digits dataset (1,797 8x8 grayscale images, labels 0-9).
  2. Splits it stratified into train (75%) and test (25%).
  3. MEMORIZES the training set two ways:
       - exact hash table:  bytes(features) -> label   (O(1) lookup)
       - KDTree index:      nearest-neighbor fallback for unseen inputs
  4. PERSISTS the store to disk as digits_store.npz so re-runs are instant.
  5. VALIDATES on the held-out test set, computing real accuracy.
  6. Prints per-class accuracy, hash-hit rate, and a few sample predictions
     so you can eyeball the results.

Run it with:
    python3 memorize_and_validate.py
"""

from __future__ import annotations

import os
import time
from collections import Counter

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(HERE, "digits_store.npz")
SEED = 42
TEST_FRACTION = 0.25


# --------------------------------------------------------------------------- #
# 1. Build / load the memory store                                            #
# --------------------------------------------------------------------------- #

def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    digits = load_digits()
    X = digits.data.astype(np.float64)          # shape (1797, 64)
    y = digits.target.astype(np.int64)          # shape (1797,)
    return X, y


def build_or_load_store(X_train: np.ndarray, y_train: np.ndarray):
    """Memorize the training data. Cache it on disk for instant re-runs."""
    if os.path.exists(STORE_PATH):
        cached = np.load(STORE_PATH)
        if np.array_equal(cached["X"], X_train) and np.array_equal(
            cached["y"], y_train
        ):
            print(f"  loaded cached store -> {STORE_PATH}")
            return cached["X"], cached["y"]

    np.savez(STORE_PATH, X=X_train, y=y_train)
    print(f"  wrote new store      -> {STORE_PATH}  ({X_train.nbytes:,} bytes)")
    return X_train, y_train


class DigitMemory:
    """Hash-first, nearest-neighbor-fallback classifier."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        # exact memorization: a dict from raw bytes -> label
        self.exact: dict[bytes, int] = {
            X[i].tobytes(): int(y[i]) for i in range(len(y))
        }
        # nearest-neighbor index for inputs we have NOT seen verbatim
        self.tree = KDTree(X)

    def predict(self, x: np.ndarray) -> tuple[int, bool]:
        """Return (predicted_label, was_exact_hit)."""
        hit = self.exact.get(x.tobytes())
        if hit is not None:
            return hit, True
        _, idx = self.tree.query(x.reshape(1, -1), k=1)
        return int(self.y[idx[0, 0]]), False


# --------------------------------------------------------------------------- #
# 2. Validate                                                                 #
# --------------------------------------------------------------------------- #

def validate(mem: DigitMemory, X_test: np.ndarray, y_test: np.ndarray):
    preds = np.empty(len(X_test), dtype=np.int64)
    exact_flags = np.zeros(len(X_test), dtype=bool)

    t0 = time.perf_counter()
    for i, x in enumerate(X_test):
        preds[i], exact_flags[i] = mem.predict(x)
        # sanity check: when memory says "this is an exact hit", the answer must
        # match what we stored for that exact vector. Anything else is a bug.
        if exact_flags[i]:
            assert preds[i] == mem.exact[x.tobytes()], \
                "exact-hit prediction disagrees with stored label"
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_test, preds)
    return preds, exact_flags, acc, elapsed


def per_class_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Show accuracy per digit class so we can see which digits we miss."""
    print("\n  per-class accuracy:")
    print("    digit   correct   total   accuracy")
    for cls in range(10):
        mask = y_true == cls
        n = int(mask.sum())
        if n == 0:
            continue
        c = int((y_pred[mask] == cls).sum())
        print(f"      {cls}      {c:>4}    {n:>4}   {c / n * 100:6.2f}%")


def show_mistakes(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    limit: int = 5,
) -> None:
    """Render the first few misclassified digits as ASCII art."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("\n  no mistakes — perfect on the test set!")
        return
    print(f"\n  first {min(limit, len(wrong))} mistakes (truth vs prediction):")
    for k in wrong[:limit]:
        grid = X_test[k].reshape(8, 8)
        print(f"\n    truth={y_true[k]}  predicted={y_pred[k]}")
        for row in grid:
            print(
                "      "
                + "".join("#" if v >= 8 else "." if v >= 4 else " " for v in row)
            )


# --------------------------------------------------------------------------- #
# 3. Driver                                                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print(" memorization-based digit classifier")
    print("=" * 70)

    print("\n[1/4] loading dataset...")
    X, y = load_dataset()
    print(f"  total samples : {len(X)}")
    print(f"  features      : {X.shape[1]}  (8x8 grayscale)")
    print(f"  classes       : {sorted(set(int(c) for c in y))}")

    print(f"\n[2/4] splitting train / test  (test_size={TEST_FRACTION}, seed={SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y
    )
    print(f"  train         : {len(X_train)}")
    print(f"  test          : {len(X_test)}")

    print("\n[3/4] memorizing training set...")
    t0 = time.perf_counter()
    X_train, y_train = build_or_load_store(X_train, y_train)
    mem = DigitMemory(X_train, y_train)
    build_s = time.perf_counter() - t0
    print(f"  unique vectors stored : {len(mem.exact):,}")
    print(f"  build time            : {build_s * 1000:.1f} ms")

    print("\n[4/4] validating on held-out test set...")
    preds, exact_flags, acc, elapsed = validate(mem, X_test, y_test)
    n = len(X_test)
    correct = int((preds == y_test).sum())
    exact_hits = int(exact_flags.sum())

    print(f"  predictions made     : {n}")
    print(f"  exact-hash hits      : {exact_hits}  ({exact_hits / n * 100:.1f}%)")
    print(f"  nearest-neighbor used: {n - exact_hits}  "
          f"({(n - exact_hits) / n * 100:.1f}%)")
    print(f"  total time           : {elapsed * 1000:.1f} ms  "
          f"({elapsed / n * 1e6:.1f} us per query)")
    print()
    print(f"  accuracy             : {correct} / {n}  =  {acc * 100:.2f}%")

    per_class_breakdown(y_test, preds)

    print("\n  confusion matrix (rows=truth, cols=prediction):")
    cm = confusion_matrix(y_test, preds, labels=list(range(10)))
    header = "         " + " ".join(f"{c:>4}" for c in range(10))
    print(header)
    for cls in range(10):
        row = " ".join(f"{cm[cls, p]:>4}" for p in range(10))
        print(f"   t={cls}  {row}")

    show_mistakes(X_test, y_test, preds)

    # --- sanity / validation invariants ------------------------------------ #
    # If any of these fail, the program crashes -- that's the point.
    assert preds.shape == y_test.shape, "prediction shape mismatch"
    assert acc == correct / n, "accuracy formula disagrees with raw count"
    # Every training point, fed back in, MUST be classified correctly --
    # memorization without recall would be useless.
    train_preds = np.array([mem.predict(x)[0] for x in X_train])
    train_acc = accuracy_score(y_train, train_preds)
    print(f"\n  self-check: training-set recall = {train_acc * 100:.2f}%  "
          f"(expected 100.00%)")
    assert train_acc == 1.0, "memorization failed: cannot recall training data"

    print("\n" + "=" * 70)
    print(" done — store on disk:", STORE_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
