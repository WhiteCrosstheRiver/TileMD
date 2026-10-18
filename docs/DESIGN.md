# Design

Living notes for data layout, kernels, and invariants. A milestone fills a section when it introduces the behavior.

## Data layout

## Kernels

## Invariants

## Host run loop

## Precision

## GPU v0

The M3 engine in `src/md/engine.cu` is one thread per atom. Atom arrays stay in input order. A cell list only builds the neighbor list: the cell edge is `(r_c + skin) * (1 + 1e-4) + 1e-5`, `floor(L / edge)` must be at least 3, and a 27-cell periodic stencil covers the cutoff. CUB stable-sorts `(cell, atom)` pairs to fill the cells. Each atom's neighbors are then insertion-sorted by index so the pair order matches the oracle's `j` loop.

The master coordinates and velocities are double. Neighbor search uses a wrapped float shadow and a cutoff padded by the same factor as the cell edge, so a pair the force kernel would accept is not dropped by float rounding. The force kernel reads the double coordinates, evaluates the LJ polynomial in double, and stores float forces. There are no floating-point atomics on the step path.

Energy and the six virial components are one host running total over that sorted list. On `boundary_500` the pair terms are about `1e34` and cancel to about `1e11`. Adding each atom's partial first, then adding the partials, moves the `xz` component by `4e6` (relative `1.60e-5`, outside the `1e-5` gate) even though the neighbor set is the oracle's set. The single total matches the oracle's association.

Velocity Verlet uses mass 1. Each step may sync with the host. The list is rebuilt every step, or when the squared float displacement from the last rebuild exceeds `(skin / 2)^2`. If the list overflows, `max_neigh` grows by half, rounded up to a multiple of 32, and the build is retried. The pair table, box, cell counts, shift mode, and neighbor cutoff live in constant memory and are copied again before every public launch.

## CPU reference

The M2 oracle in `reference/lj_reference.cpp` is the double-precision pair loop from PLAN.md section 4.1. It visits every ordered pair. Minimum image uses `std::rint`, the double-precision form of the `rintf` rule in decision D10. Per-atom energy is half the pair energy, and the virial includes the full-list factor of one half. `lj_shift = force` subtracts `U'(r_c)` so both the energy and the force vanish at the cutoff. Fixtures with `N <= 4000` use this brute-force loop. A cell list for larger systems is not in this milestone.
