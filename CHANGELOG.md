# Changelog

## [Unreleased]

### Added
- M0 repository skeleton: CMake CUDA project, CPU and GPU smoke tests, CI, and benchmark environment recorder.
- Key-value input parser for the Appendix D sketch. An fcc melt gets zero-momentum Gaussian velocities. `tilemd input.in` runs that file on the CPU reference, writes thermo to stdout, and appends extended-XYZ frames when `dump_every` is set. Those frames reload through `Properties`, including the velocity column. `units metal` and `eam` run a funcfl table with the metal-unit velocity Verlet. A 10^5-step copper NVE drift matches the LAMMPS energy slope, `-1.94e-9` eV/atom/ps. A setfl reader matches LAMMPS `eam/alloy` on a four-atom Cu-Ni cell to `1.5e-15` relative force, and 100 NVE steps stay within `6.2e-15` Å. The stage-1 metal input walks the cluster-pair list. A metal `velocity` line draws `k_B T / (m mvv2e)` with each element's mass and removes the mass-weighted momentum. Metal Langevin is BAOAB; zero friction matches Verlet, and zero temperature damps a force-free pair by `e^{-γ t}`. It runs that setfl file with one mass per element. One thousand Cu-Ni steps stay within `1.4e-13` Å of LAMMPS. A 10^5-step energy series on that four-atom gas agrees through 3 ps and then separates.
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
- M6 CPU BAOAB Langevin thermostat. The random draw is Philox4x32-10 keyed by seed, step, and atom id. Zero friction matches velocity Verlet. The device step does not apply it yet.
- M6 fused GPU leapfrog step on the cluster-tile path. Pair coordinates come from `pos4`. The host cluster list is rebuilt when the skin displacement is exceeded, and this kernel has not been launched.
- M6 `pos4` coordinate rounding. On `melt_4000` it moves the force by `5.81e-5` relative to the double-precision oracle, outside the `1e-5` tolerance. One hundred leapfrog steps on the 40-atom box move positions by at most `5.69e-8`.
- M6 fused-step traffic on the 32k melt. Distinct global addresses in one step are 2163 bytes per atom, above the 60–110 byte target, because each kept cluster reloads `pos4`. This is a source count, not an Nsight measurement.
- M4 non-finite abort. The first non-finite atom is recorded, `emergency.extxyz` is written in original-gid order, and the message names the step and that atom. This path has not been launched.
- M4 periodic dump. With `dump_every` set, `Run` ends the segment at the next multiple and appends an extended-XYZ frame in original-gid order. The fused path stores the leapfrog velocity. This path has not been launched.
- M6 fused graph. With per-step rebuild off, `graph_unroll` fused kernels are captured on alternating `pos4` buffers. Thermo is chosen from the device step counter, and the cluster list is replaced on the device only when the displacement is past half the skin. A wrapped coordinate is put back on the reference branch before the stored image shift. On the CPU, 1000 such steps rebuild 64 times and stay within `1.34e-11` of rebuilding every step. The device sort key matches the host grid on the CPU, including a wrapped coordinate and a negative zero. The pair walk and the bounding box are the same functions the host list uses. A coordinate at z = −1 in a box of length 10 is stored as 9. A list that does not fit restores the segment and reruns it one step at a time. The column sort still runs when the flag is clear. The default still rebuilds every step on the host, and this graph has not been launched.
- M4 captured group length. `graph_unroll` selects how many atom-neighbor steps go into one CUDA graph. The default 50 divides a 1000-step segment evenly; 30 leaves a 10-step tail. The fused tile path still launches one step at a time.
- M6 fused thermo. A thermo step reduces potential, virial, and integer-time kinetic energy once per cluster. The final step is recorded by one extra force evaluation. Eight steps with interval 2 record times 0, 2, 4, 6, and 8, matching velocity Verlet on the CPU. The device path has not been launched.
- M7 CPU prototype of the TF32 and 3×TF32 pair score. It is not on the default force path. On the 4000-atom melt the 3×TF32 force error is outside the 1e-5 tolerance.
- M7 compact-cell remeasure of that score. A cell of about 4 atoms lowers the melt force error from 7.9e-5 to 1.9e-5, which is still outside the tolerance.
- M8 CPU pair-attention API. Lennard-Jones matches the double-precision oracle. A two-pass EAM funcfl evaluation matches LAMMPS `pair_style eam` on displaced fcc copper, including 100 steps of metal-unit velocity Verlet.
- M8 device Lennard-Jones attention kernel. One warp walks one i-cluster on the host-built list. A host replay of that lane map matches the oracle on the fixtures. The device launch waits while the 32k drift occupies the GPU.
- M5 cluster-tile GPU test now uses a 12³ box. The previous 8³ box has only two cells and the engine rejects it.
- M8 pair-attention passes on the cluster-pair list. Lennard-Jones matches the oracle, and the copper EAM tile walk matches LAMMPS to `1.6e-13`. One hundred metal-unit steps on that walk stay within `1.4e-14` Å of the LAMMPS trajectory.
