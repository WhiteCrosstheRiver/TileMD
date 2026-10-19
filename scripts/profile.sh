#!/usr/bin/env bash
# Nsight Compute text export for one force kernel. Does not write a .ncu-rep.
# Usage: scripts/profile.sh <output.txt> -- <command>
# The GPU must be free. Metrics are the Speed-phase set from REVIEW_AND_PLAN_v2 T-P1.
set -euo pipefail
if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: scripts/profile.sh <output.txt> -- <command>" >&2
  exit 2
fi
out="$1"
shift 2
metrics="dram__bytes.sum,lts__t_bytes.sum,l1tex__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum"
ncu_bin="${NCU:-ncu}"
mkdir -p "$(dirname "$out")"
"$ncu_bin" --clock-control none --metrics "$metrics" --page raw "$@" >"$out"
echo "wrote $out"
