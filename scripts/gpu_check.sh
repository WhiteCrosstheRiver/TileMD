#!/usr/bin/env bash
# Full suite, including GPU tests. Writes a text log under reports/logs/.
# Usage: scripts/gpu_check.sh M0   -> reports/logs/M0_gpu_check.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="${1:-gpu_check}"
BUILD_DIR="${BUILD_DIR:-build}"
mkdir -p "$ROOT/reports/logs"
LOG="$ROOT/reports/logs/${NAME}_gpu_check.log"

{
  "$BASH" "$ROOT/scripts/check.sh"
  echo
  echo "=== GPU tests ==="
  ctest_args=(--test-dir "$BUILD_DIR" -L gpu --output-on-failure)
  if grep -q '^CMAKE_CONFIGURATION_TYPES:' "$BUILD_DIR/CMakeCache.txt"; then
    ctest_args+=(-C Release)
  fi
  ctest "${ctest_args[@]}"
} 2>&1 | tee "$LOG"
