"""Digits memory store — download once, then run fully offline.

Storage layout:
    digits.db   SQLite database, persistent on disk
    digits.csv  Plain-CSV export consumed by the C version

In-memory index (built at startup from the SQLite blob):
    self.exact   bytes(features) -> label        O(1) hash lookup
    self.tree    sklearn KDTree on the features  O(log n) nearest neighbor

Query logic: exact match first, fall back to nearest neighbor.
"""

import os
import sqlite3
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "digits.db")
CSV_PATH = os.path.join(HERE, "digits.csv")
N_FEATURES = 64


def build_store(force: bool = False) -> None:
    """Download the digits dataset and persist it. No-op if already cached."""
    if os.path.exists(DB_PATH) and os.path.exists(CSV_PATH) and not force:
        return

    print("Downloading digits dataset (one-time)...")
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE digits ("
        "id INTEGER PRIMARY KEY, "
        "features BLOB NOT NULL, "
        "label INTEGER NOT NULL)"
    )
    cur.executemany(
        "INSERT INTO digits VALUES (?, ?, ?)",
        [(i, X[i].tobytes(), int(y[i])) for i in range(len(y))],
    )
    conn.commit()
    conn.close()

    np.savetxt(
        CSV_PATH,
        np.hstack([X, y.reshape(-1, 1)]),
        delimiter=",",
        fmt="%g",
    )
    print(f"Stored {len(y)} samples in {DB_PATH}")
    print(f"Exported {CSV_PATH} for the C version")


class DigitMemory:
    """RAM-resident index built from the SQLite store."""

    def __init__(self) -> None:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT features, label FROM digits").fetchall()
        conn.close()

        self.X = np.vstack(
            [np.frombuffer(b, dtype=np.float64) for b, _ in rows]
        )
        self.y = np.array([lab for _, lab in rows], dtype=np.int64)

        self.exact: dict[bytes, int] = {
            self.X[i].tobytes(): int(self.y[i]) for i in range(len(self.y))
        }
        self.tree = KDTree(self.X)

    def query(self, arr) -> tuple[int, float, bool]:
        """Return (label, distance, was_exact)."""
        q = np.asarray(arr, dtype=np.float64).reshape(-1)
        if q.size != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {q.size}")

        hit = self.exact.get(q.tobytes())
        if hit is not None:
            return hit, 0.0, True

        dist, idx = self.tree.query(q.reshape(1, -1), k=1)
        return int(self.y[idx[0, 0]]), float(dist[0, 0]), False


def _demo(mem: DigitMemory) -> None:
    print(f"\nLoaded {len(mem.y)} samples into memory.")

    sample = mem.X[42]
    t0 = time.perf_counter()
    label, dist, exact = mem.query(sample)
    dt_us = (time.perf_counter() - t0) * 1e6
    print(
        f"Exact query     -> label={label}  dist={dist:.4f}  "
        f"exact={exact}  ({dt_us:.1f} us)"
    )

    perturbed = sample.copy()
    perturbed[0] += 0.7
    perturbed[15] -= 1.3
    t0 = time.perf_counter()
    label, dist, exact = mem.query(perturbed)
    dt_us = (time.perf_counter() - t0) * 1e6
    print(
        f"Perturbed query -> label={label}  dist={dist:.4f}  "
        f"exact={exact}  ({dt_us:.1f} us)"
    )

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 2.0, size=N_FEATURES)
    t0 = time.perf_counter()
    label, dist, exact = mem.query(mem.X[100] + noise)
    dt_us = (time.perf_counter() - t0) * 1e6
    print(
        f"Noisy query     -> label={label}  dist={dist:.4f}  "
        f"exact={exact}  ({dt_us:.1f} us)"
    )


if __name__ == "__main__":
    build_store()
    _demo(DigitMemory())
