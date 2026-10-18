# Changelog

## [Unreleased]

### Added
- M0 repository skeleton: CMake CUDA project, CPU and GPU smoke tests, CI, and benchmark environment recorder.
- M2 double-precision LJ reference, seven fixtures, and a LAMMPS force comparison for `melt_4000`.
- M3 thread-per-atom neighbor list, LJ force, and velocity Verlet on the GPU.
- M3 32k NVE comparison against CPU LAMMPS, plus drift and RDF drivers for the remaining G1 runs.
- M3 RDF on the 32k melt. The L1 distance to LAMMPS g(r) is 0.00147, inside the 0.02 gate. The 10^6-step drift comparison is still running.
- M4 stable Morton reorder at each neighbor rebuild, with output still in the original atom order.
- M4 device-side rebuild decision. The step loop no longer copies status back on every step.
- M4 CUDA graph of 50 steps, with a segment snapshot that restores and grows the list on overflow.
- M4 energy and virial only on thermo steps, reduced on device into a ring drained at segment boundaries.
- M4 launch bounds and read-only position loads on the force and neighbor kernels. Block size stays 128.
- M4 conditional graph node around the neighbor rebuild, with the early-exit kernels kept as the fallback.
- M5 CPU cluster-pair list: x–y columns, stable z order, 8-atom clusters, bounding-box pruning.
- M5 CPU tile force on that list, checked against the double-precision oracle.
- M5 GPU tile force: one warp per cluster, selectable beside the atom-neighbor engine. The cluster list is still built on the host.
- M5 tile-efficiency measurement on `melt_4000` and the 32k melt.
- M5 cluster-pair rows are appended in cluster order. The earlier row starts overlapped, and the corrected tile forces match the double-precision oracle on the fixtures. Efficiency is 0.102 at 4k and 0.0916 at 32k.
- M5 column-width sweep. Targets of 64, 128, and 256 atoms per column keep the same inside-pair count and the same forces. None of them reaches the 40–60% efficiency expectation.
- M6 CPU leapfrog integrator. Over 100 steps it matches velocity Verlet to round-off on the double-precision oracle.
- M6 fused GPU leapfrog step on the cluster-tile path. Pair coordinates come from `pos4`. The host cluster list is rebuilt when the skin displacement is exceeded, and this kernel has not been launched.
- M6 `pos4` coordinate rounding. On `melt_4000` it moves the force by `5.81e-5` relative to the double-precision oracle, outside the `1e-5` tolerance. One hundred leapfrog steps on the 40-atom box move positions by at most `5.69e-8`.
- M7 CPU prototype of the TF32 and 3×TF32 pair score. It is not on the default force path. On the 4000-atom melt the 3×TF32 force error is outside the 1e-5 tolerance.
- M7 compact-cell remeasure of that score. A cell of about 4 atoms lowers the melt force error from 7.9e-5 to 1.9e-5, which is still outside the tolerance.
- M8 CPU pair-attention API. Lennard-Jones matches the double-precision oracle. A two-pass EAM funcfl evaluation matches LAMMPS `pair_style eam` on displaced fcc copper, including 100 steps of metal-unit velocity Verlet.
- M8 device Lennard-Jones attention kernel. It compiles and has not been launched while the 32k validation occupies the GPU.
- M8 pair-attention passes on the cluster-pair list. Lennard-Jones matches the oracle, and the copper EAM tile walk matches LAMMPS to `1.6e-13`. One hundred metal-unit steps on that walk stay within `1.4e-14` Å of the LAMMPS trajectory.
