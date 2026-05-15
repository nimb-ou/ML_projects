#!/usr/bin/env bash
#
# Run every benchmark in sequence and print a side-by-side comparison.
#
# Usage: ./bench/run_all.sh [N_QUERIES]
#
# Examples:
#   ./bench/run_all.sh             # 1000 queries (default)
#   ./bench/run_all.sh 5000        # 5000 queries
#   ./bench/run_all.sh 50000       # heavier run, ~1s total

set -euo pipefail

cd "$(dirname "$0")/.."

N="${1:-1000}"

# --- prerequisites ---------------------------------------------------------
if [ ! -f "digits.csv" ]; then
  echo "[setup] digits.csv not found — running digits_memory.py to generate it."
  python3 digits_memory.py >/dev/null
fi

# --- compile C bench -------------------------------------------------------
echo "[build] compiling bench/bench.c ..."
gcc -O3 -Wall -Wextra -o bench/bench bench/bench.c -lm

echo
# --- Python bench ----------------------------------------------------------
python3 bench/bench.py --queries "$N"
echo

# --- C bench ---------------------------------------------------------------
./bench/bench --queries "$N"
echo

# --- comparison ------------------------------------------------------------
python3 bench/compare.py
