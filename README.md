# TileMD

GPU molecular-dynamics engine whose pair forces are evaluated as dense tiles. Stage 1 is a Lennard-Jones validation engine. The milestone plan is [docs/PLAN.md](docs/PLAN.md).

Licensed under [GPL-3.0](LICENSE).

## Prerequisites

- CMake ≥ 3.24
- CUDA Toolkit ≥ 12.4 (nvcc)
- A C++17 host compiler
- Python ≥ 3.11 for `bench/scripts`
- `clang-format` 22.1.8 on `PATH` (the version CI installs)

Default GPU architectures are `sm_80` and `sm_90`. For another card pass `-DTILEMD_ARCHS=<cc>` (an RTX 3050 is `86`).

## Build, test, and record the machine

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/tilemd --version
bash scripts/check.sh
bash scripts/gpu_check.sh M0
python3 bench/scripts/collect_env.py --cmake-cache build/CMakeCache.txt
```

`scripts/check.sh` is the CI entry point: format check, Release build, CPU tests. GPU tests skip when no device is visible or the binary has no image for that device. `scripts/gpu_check.sh M0` writes `reports/logs/M0_gpu_check.log`.

On Windows, `scripts/check.sh` loads the Visual Studio C++ environment and configures Ninja, so `nvcc` must be on `PATH`. Benchmarks against GPUMD and LAMMPS start at milestone M1; `collect_env.py` is the environment record those runs attach.

## Layout

Engine code lives under `src/`. The M3 GPU engine is `src/md/engine.cu` (thread-per-atom neighbor list, LJ force, velocity Verlet). The double-precision LJ oracle is `reference/lj_reference.cpp`; `tilemd_make_fixtures` regenerates `tests/fixtures`. GPU correctness tests are `tests/gpu/lj_gpu_test.cu`, run with `bash scripts/gpu_check.sh M3`. See [docs/PLAN.md](docs/PLAN.md) section 3.
