"""Print a side-by-side comparison of Python vs C benchmark results.

Reads bench/results_python.json and bench/results_c.json (both produced by
the previous bench runs) and emits a single comparison table.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def fmt_us(v: float) -> str:
    if v < 10:
        return f"{v:6.2f} us"
    if v < 1000:
        return f"{v:6.1f} us"
    return f"{v/1000:5.2f} ms"


def fmt_qps(v: float) -> str:
    if v >= 1e6:
        return f"{v/1e6:5.2f}M qps"
    if v >= 1e3:
        return f"{v/1e3:5.1f}k qps"
    return f"{v:5.0f} qps"


def fmt_mb(v: int) -> str:
    return f"{v/1024/1024:+5.2f} MB"


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    py = load(HERE / "results_python.json")
    c  = load(HERE / "results_c.json")
    if py is None or c is None:
        print("Run ./bench/run_all.sh first.")
        return

    print("=" * 78)
    print(" CROSS-LANGUAGE COMPARISON")
    print("=" * 78)
    print(f" Queries  : {py['cfg']['n_queries']}")
    print(f" Dataset  : {py['cfg']['n_samples']} x {py['cfg']['n_features']}")
    print()

    print(f" {'method':<32} {'lang':>6}  {'build':>10}  {'p50':>10}  "
          f"{'p99':>10}  {'throughput':>12}")
    print("-" * 78)
    for r in py["results"]:
        print(f" {r['name']:<32} {'PY':>6}  "
              f"{fmt_us(r['build']['wall_s']*1e6):>10}  "
              f"{fmt_us(r['per_query']['median_us']):>10}  "
              f"{fmt_us(r['per_query']['p99_us']):>10}  "
              f"{fmt_qps(r['throughput_qps']):>12}")
    print(f" {'hash + brute fallback':<32} {'C':>6}  "
          f"{fmt_us(c['build']['wall_s']*1e6):>10}  "
          f"{fmt_us(c['per_query']['median_us']):>10}  "
          f"{fmt_us(c['per_query']['p99_us']):>10}  "
          f"{fmt_qps(c['throughput_qps']):>12}")
    print("-" * 78)

    # comparable rows: Python hybrid vs C
    py_hybrid = next((r for r in py["results"]
                      if r["name"].startswith("hybrid")), None)
    if py_hybrid:
        py_p50 = py_hybrid["per_query"]["median_us"]
        c_p50 = c["per_query"]["median_us"]
        py_qps = py_hybrid["throughput_qps"]
        c_qps = c["throughput_qps"]
        speedup = (c_qps / py_qps) if py_qps > 0 else 0
        print()
        print(f" Same workload, same data, both languages:")
        print(f"   Python hybrid p50   : {fmt_us(py_p50)}")
        print(f"   C       hybrid p50  : {fmt_us(c_p50)}")
        print(f"   C is {speedup:.1f}x the throughput of Python hybrid.")


if __name__ == "__main__":
    main()
