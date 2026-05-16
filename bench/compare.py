"""Print a side-by-side comparison of Python vs C results.

Reads:
    bench/results_python.json   (produced by bench.py)
    bench/results_c.json        (produced by ./bench/bench)

Output:
    a single table comparing every Python method against the C
    hash + brute-fallback on the same held-out test set.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def fmt_us(v):
    if v < 10:    return f"{v:6.2f} us"
    if v < 1000:  return f"{v:6.1f} us"
    return f"{v/1000:5.2f} ms"


def fmt_qps(v):
    if v >= 1e6: return f"{v/1e6:5.2f}M qps"
    if v >= 1e3: return f"{v/1e3:5.1f}k qps"
    return f"{v:5.0f} qps"


def fmt_pct(v):
    return f"{v*100:6.2f}%"


def load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    py = load(HERE / "results_python.json")
    c  = load(HERE / "results_c.json")
    if py is None or c is None:
        print("Both bench/results_python.json and bench/results_c.json must exist.")
        print("Run: ./bench/run_all.sh")
        return

    print("=" * 90)
    print(" CROSS-LANGUAGE COMPARISON")
    print("=" * 90)
    print(f" Test set : {py['cfg']['n_test']} held-out samples, "
          f"split seed {py['cfg']['split_seed']}")
    print()
    print(f" {'method':<36} {'lang':>5}  {'p50':>9}  {'p99':>9}  "
          f"{'throughput':>11}  {'accuracy':>10}")
    print("-" * 90)
    for r in py["results"]:
        print(f" {r['name']:<36} {'PY':>5}  "
              f"{fmt_us(r['per_query']['median_us']):>9}  "
              f"{fmt_us(r['per_query']['p99_us']):>9}  "
              f"{fmt_qps(r['throughput_qps']):>11}  "
              f"{fmt_pct(r['accuracy']):>10}")
    print(f" {'hash + brute fallback':<36} {'C':>5}  "
          f"{fmt_us(c['per_query']['median_us']):>9}  "
          f"{fmt_us(c['per_query']['p99_us']):>9}  "
          f"{fmt_qps(c['throughput_qps']):>11}  "
          f"{fmt_pct(c['accuracy']):>10}")
    print("-" * 90)

    # head-to-head on hybrid
    py_hybrid = next((r for r in py["results"]
                      if "hybrid" in r["name"]), None)
    if py_hybrid:
        py_p50 = py_hybrid["per_query"]["median_us"]
        c_p50  = c["per_query"]["median_us"]
        py_qps = py_hybrid["throughput_qps"]
        c_qps  = c["throughput_qps"]
        py_acc = py_hybrid["accuracy"]
        c_acc  = c["accuracy"]
        speedup = c_qps / py_qps if py_qps > 0 else 0
        print()
        print(" Same workload, same data, both languages:")
        print(f"   Python hybrid:  p50 {fmt_us(py_p50).strip():>10}  "
              f"{fmt_qps(py_qps).strip():>10}  "
              f"acc {fmt_pct(py_acc).strip()}")
        print(f"   C hybrid     :  p50 {fmt_us(c_p50).strip():>10}  "
              f"{fmt_qps(c_qps).strip():>10}  "
              f"acc {fmt_pct(c_acc).strip()}")
        print(f"   C is {speedup:.1f}× the throughput of Python "
              f"with identical accuracy.")


if __name__ == "__main__":
    main()
