# Design

Living notes for data layout, kernels, and invariants. A milestone fills a section when it introduces the behavior.

## Data layout

## Kernels

## Invariants

## Host run loop

## Precision

## CPU reference

The M2 oracle in `reference/lj_reference.cpp` is the double-precision pair loop from PLAN.md section 4.1. It visits every ordered pair. Minimum image uses `std::rint`, the double-precision form of the `rintf` rule in decision D10. Per-atom energy is half the pair energy, and the virial includes the full-list factor of one half. `lj_shift = force` subtracts `U'(r_c)` so both the energy and the force vanish at the cutoff. Fixtures with `N <= 4000` use this brute-force loop. A cell list for larger systems is not in this milestone.
