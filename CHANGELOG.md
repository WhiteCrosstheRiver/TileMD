# Changelog

## [Unreleased]

### Added
- M0 repository skeleton: CMake CUDA project, CPU and GPU smoke tests, CI, and benchmark environment recorder.
- M2 double-precision LJ reference, seven fixtures, and a LAMMPS force comparison for `melt_4000`.
- M3 thread-per-atom neighbor list, LJ force, and velocity Verlet on the GPU.
- M3 32k NVE comparison against CPU LAMMPS, plus drift and RDF drivers for the remaining G1 runs.
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
