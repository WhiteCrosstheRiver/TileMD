# Changelog

## [Unreleased]

### Added
- M0 repository skeleton: CMake CUDA project, CPU and GPU smoke tests, CI, and benchmark environment recorder.
- M2 double-precision LJ reference, seven fixtures, and a LAMMPS force comparison for `melt_4000`.
- M3 thread-per-atom neighbor list, LJ force, and velocity Verlet on the GPU.
- M3 32k NVE comparison against CPU LAMMPS, plus drift and RDF drivers for the remaining G1 runs.
- M4 stable Morton reorder at each neighbor rebuild, with output still in the original atom order.
