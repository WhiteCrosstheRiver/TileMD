#!/usr/bin/env bash
# One cluster job: build TileMD, record the environment, time the existing paths.
# The partition and account lines are placeholders. This machine has no Slurm.
#SBATCH --job-name=tilemd-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
cmake -S . -B build-cluster -G Ninja -DCMAKE_BUILD_TYPE=Release -DTILEMD_BUILD_TESTS=OFF
cmake --build build-cluster --target tilemd_bench --parallel
python3 bench/scripts/collect_env.py >"bench/results/${SLURM_JOB_ID:-local}_env.json"
python3 bench/scripts/run_bench.py --binary build-cluster/tilemd_bench \
  --paths atom fused-tile --sizes 32000 256000 2048000
echo "LAMMPS-KOKKOS, when the module exists: lmp -k on g 1 -sf kk -pk kokkos newton off neigh full"
