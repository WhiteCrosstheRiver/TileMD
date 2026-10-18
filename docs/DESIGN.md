# Design

Living notes for data layout, kernels, and invariants. A milestone fills a section when it introduces the behavior.

## Data layout

## Kernels

## Invariants

## Host run loop

## Precision

## GPU v0

The engine in `src/md/engine.cu` is one thread per atom. Each neighbor rebuild stable-sorts atoms by a 10-bit Morton code of the cell and permutes every per-atom array, including `gid`. Downloads scatter positions, velocities, and forces back to the original global id. Within a cell, the previous order is kept. Neighbor slots are then sorted by that global id so the pair sum follows the oracle's atom order. A cell list builds the neighbor list: the cell edge is `(r_c + skin) * (1 + 1e-4) + 1e-5`, `floor(L / edge)` must be at least 3, and a 27-cell periodic stencil covers the cutoff. CUB stable-sorts `(cell, atom)` pairs to fill the cells. Each atom's neighbors are then insertion-sorted by index so the pair order matches the oracle's `j` loop.

The master coordinates and velocities are double. Neighbor search uses a wrapped float shadow and a cutoff padded by the same factor as the cell edge, so a pair the force kernel would accept is not dropped by float rounding. The force kernel reads the double coordinates, evaluates the LJ polynomial in double, and stores float forces. There are no floating-point atomics on the step path.

Energy and the six virial components are one host running total over that sorted list. On `boundary_500` the pair terms are about `1e34` and cancel to about `1e11`. Adding each atom's partial first, then adding the partials, moves the `xz` component by `4e6` (relative `1.60e-5`, outside the `1e-5` gate) even though the neighbor set is the oracle's set. The single total matches the oracle's association.

Velocity Verlet uses mass 1. Inside `Run`, the displacement check is a block reduction and an `atomicMax` into `DeviceStatus`. A one-thread kernel sets `need_rebuild` from that value or from `rebuild_every_step`. The rebuild kernels return immediately when the flag is clear, so the step loop does not copy status to the host. `Run` captures 50 steps in one CUDA graph and replays it, synchronizing and reading `DeviceStatus` every 1000 steps. Each captured step publishes `need_rebuild` into a conditional node. The sort and the neighbor rebuild are the body of that node, so they are skipped when the flag is clear. The kernels still return immediately when the flag is clear; that path is used if the conditional node cannot be captured. Energy and the six virial components on the step path are computed only when the completed step is a multiple of `thermo_every` (`0` leaves them off). A block reduction, then one thread walking those partials in block order, writes a double record into `thermo_ring`. The host copies that ring out at the segment boundary. `ComputeForces` still sums energy on the host in global-id order; that is the path the correctness gates use. Each segment starts with a device snapshot of positions, velocities, images, types, global ids, the step counter, and the thermo count. On overflow the snapshot is restored, `max_neigh` grows, the list is rebuilt, the graph is captured again, and the segment is rerun. `ComputeForces` still grows `max_neigh` on the host when the list overflows. The pair table, box, cell counts, shift mode, neighbor cutoff, rebuild trigger, and thermo interval live in constant memory and are copied again before every public launch. The force and neighbor kernels use `__launch_bounds__(128)` and `__ldg` for positions. The stored force is still float and the pair geometry is still double. Block sizes 64 and 256 are not selected; they need a timed sweep on a free GPU.

## Cluster pairs

The CPU builder in `src/neighbor/cluster_list.cpp` is the M5 list. Columns lie on an x–y grid. A zero column count picks a grid with about 32 atoms per column. Inside a column, atoms are stable-sorted by z and the length is padded to a multiple of 8. Padding slots use type `-1` and are omitted from the cluster bounding box, so the interaction mask is the type rather than a sentinel coordinate. A cluster pair is stored when the periodic bounding-box distance is below `reach`. The shift index packs `sx, sy, sz`, each in `{-1, 0, 1}`. `ComputeTileForces` walks that list in cluster order and applies the image shift stored with each pair. On a 40-atom shifted-force system it matches the double-precision oracle. With the planned 32-atom columns, tile efficiency (pairs inside the cutoff divided by pairs computed) is 0.0229 on `melt_4000` and 0.0085 on the 32k LJ melt. An 8-atom z-slice is several cutoffs long, so most pairs inside a kept cluster sit outside `rc`. Widening the columns until a cluster is roughly cubic did not raise that ratio, and the one-box stencil then double-counted some pairs, so the 32-atom grid stays.

`src/md/tile_force.cu` is the matching GPU kernel: one warp per i-cluster, lane `i_local + 8 * j_sub`. Each group of four j-clusters is broadcast with warp shuffles, and the four lane partials are added in j-sub order. `HostSystem::use_cluster_tiles` selects this path for `ComputeForces` and for velocity Verlet. The cluster list is still built on the host at every force evaluation. The default engine remains the atom-neighbor list. Tile thermo and a device-side cluster rebuild are not on this path.

## CPU reference

The M2 oracle in `reference/lj_reference.cpp` is the double-precision pair loop from PLAN.md section 4.1. It visits every ordered pair. Minimum image uses `std::rint`, the double-precision form of the `rintf` rule in decision D10. Per-atom energy is half the pair energy, and the virial includes the full-list factor of one half. `lj_shift = force` subtracts `U'(r_c)` so both the energy and the force vanish at the cutoff. Fixtures with `N <= 4000` use this brute-force loop. A cell list for larger systems is not in this milestone.

Velocity Verlet stores `v(t)`. Leapfrog stores `v(t − dt/2)`, prepared as `v(0) − ½ dt f(0) / m`. On a 40-atom system for 100 steps of `dt = 0.005`, the two trajectories agree to round-off: the largest position difference is `8.9e-16` and the largest integer-time velocity difference is `2.1e-15`. The fused GPU step is not in this tree.
