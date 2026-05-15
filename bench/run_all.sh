#!/usr/bin/env bash
#
# Run the full bench suite (Python + C) and print a comparison.
#
# Usage:  ./bench/run_all.sh [SEED]
#
# Examples:
#   ./bench/run_all.sh           # default seed (42)
#   ./bench/run_all.sh 7         # different seed for the train/test split

set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${1:-42}"

# --- ensure the underlying dataset CSV exists -----------------------------
if [ ! -f "digits.csv" ]; then
  echo "[setup] digits.csv not found — running digits_memory.py once."
  python3 digits_memory.py >/dev/null
fi

# --- compile the C bench ---------------------------------------------------
echo "[build] compiling bench/bench.c ..."
gcc -O3 -Wall -Wextra -o bench/bench bench/bench.c -lm

echo
# --- Python bench (also writes the train/test CSVs the C bench needs) -----
python3 bench/bench.py --seed "$SEED"
echo

# --- C bench (uses bench/digits_train.csv and bench/digits_test.csv) ------
./bench/bench
echo

# --- comparison ------------------------------------------------------------
python3 bench/compare.py
