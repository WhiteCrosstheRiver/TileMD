# Numerics

Stage 1 stores positions and accumulates forces in double precision. M7 asks whether the pair distance can instead be formed as a tensor-core matrix product. This note records the CPU prototype of that product. It is not on the default force path.

## Score

For one i atom and one j atom, with coordinates relative to the i-tile center and j already shifted onto the periodic image that owns the pair:

```
A = [x_i, y_i, z_i, |r_i|^2, 1, 0, 0, 0]
B = [-2 x_j, -2 y_j, -2 z_j, 1, |r_j|^2, 0, 0, 0]
D = A · B = |r_i - r_j|^2
```

`|r|^2` is a binary32 sum of squares. The algebra is exact in real arithmetic. In floating point it cancels the large part of `|r_i|^2 + |r_j|^2`. The cancellation factor is `(|r|/d)^2`, where `d` is the pair distance. A tile center that sits on the atoms makes `|r|` a fraction of the tile diameter instead of a fraction of the box.

## TF32 rounding

TF32 keeps the binary32 exponent and 10 mantissa bits. The other 13 mantissa bits are rounded to nearest even. Zeros, subnormals, infinities, and NaNs are left unchanged. `reference/tf32_pair.cpp` does this with integer operations on the binary32 bits. It does not use a tensor-core instruction.

Three products are compared, each accumulated in binary32:

- FP32: `Σ a_k b_k`.
- TF32: `Σ tf32(a_k) tf32(b_k)`.
- 3×TF32: `a_hi = tf32(a)`, `a_lo = tf32(a - a_hi)`, and the same split for `b`. The sum is `a_hi b_hi + a_hi b_lo + a_lo b_hi`. The low-low product is omitted, as in the usual three-term TF32 expansion.

The force uses this approximate `D` only for the scalar `s(D)`. The direction is the double-precision minimum-image vector. A pair whose approximate `D` is not inside `(0, rc^2)` is dropped, which is the mask the tensor-core kernel would apply.

## What was measured

`tests/unit/tf32_pair_test.cpp`, Release, MSVC. Relative `r^2` error is `|D - d^2| / d^2`. Relative force error is the maximum force-vector error divided by the root-mean-square reference force, the same ratio as T1.

Pairs placed about 20 length units from the origin, then repeated about their own centroid:

| frame | FP32 | TF32 | 3×TF32 |
|---|---:|---:|---:|
| absolute, error scaled by `max(1, d^2)` | 8.3e-5 | 0.384 | 1.0e-4 |
| tile centroid, same scale | 1.3e-7 | 5.5e-4 | 2.2e-7 |

Plain TF32 on absolute coordinates loses the distance. The tile frame is what makes a 10-bit mantissa conceivable. 3×TF32 is far closer than plain TF32 in both frames. On absolute coordinates it is slightly worse than FP32 (1.0e-4 against 8.3e-5): the discarded low-low term still matters when `|r|` is large.

A 32-atom cloud with spacing 1.2, one centroid for the whole cloud, 3×TF32 force relative error `9.9e-7`. That is inside the T1 bound of `1e-5`. The cloud is compact, so the centroid really is a tile center.

The seven fixtures use the centroid of each atom's current 8-atom z-slice as the tile center. Those slices are the M5 clusters. They are not compact: an 8-atom slice of a column still spans a large fraction of the box, and the cancellation factor `(|r|/d)^2` stays large. Maximum relative `r^2` error for pairs with `d^2 > 0.25`, and the force error of the full 3×TF32 evaluation:

| fixture | tile `(|r|/d)^2` | tile 3×TF32 `r^2`, `d^2 > 0.25` | tile TF32 `r^2`, `d^2 > 0.25` | force relative error |
|---|---:|---:|---:|---:|
| fcc_500 | 1.8 | 8.9e-7 | 1.7e-3 | absolute residual 1.1e-5; the reference force is round-off |
| cutoff_500 | 26 | 1.9e-6 | 2.0e-2 | 1.1e-5 |
| melt_4000 | 10 | 4.9e-6 | 1.2e-2 | 7.9e-5 |
| dilute_1000 | 1.5e2 | 1.9e-5 | 6.0e-2 | 1.3e-3 |
| twotype_2000 | 3.0e2 | 1.8e-5 | 6.8e-2 | 5.6e-3 |
| dense_2000 | 9.4e2 | 9.5e-6 | 2.2e-2 | 5.0e-3 |
| boundary_500 | 2.2e6 | 7.5e-6 | 1.6e-2 | 19 |

On every fixture the separated-pair 3×TF32 distance beats plain TF32 by two to four orders of magnitude. The melt force error, `7.9e-5`, is outside T1. The boundary, dense, two-type, and dilute fixtures are worse because a few very small `d` make `s(D)` singular: a `1e-5` absolute error in `d^2` is a large relative error, and the 8-atom slice does not bring both atoms near the origin. `fcc_500` has a cancelling reference force, so the relative ratio is not meaningful; the absolute residual is `1.1e-5`.

## Consequence for G4

T1 is not met on `melt_4000`. Drift and the speed comparison with M6 were not run. There is no `mma.sync` kernel. 3×TF32 stays in `reference/tf32_pair.cpp` and is not called by the engine. A compact tile, as in the 32-atom cloud, does get inside T1, so the open question is the tile geometry rather than the three-product expansion itself. Plain TF32 does not.
