#pragma once

#include <cmath>
#include <type_traits>

#include "neighbor/cluster_order.hpp"

namespace tilemd {

// One ordered pair's contribution on atom i. Energy, virial, and |r·F| already
// include the factor 1/2 used by the double-precision oracle.
template <typename T>
struct LjTerm {
  T fx = 0;
  T fy = 0;
  T fz = 0;
  T energy = 0;
  T wxx = 0;
  T wyy = 0;
  T wzz = 0;
  T wxy = 0;
  T wxz = 0;
  T wyz = 0;
  T pair_abs = 0;
  bool inside = false;
};

// `shift` is 0 (none), 1 (energy), or 2 (force). Float uses __frcp_rn on device.
template <typename T>
TILEMD_HD inline LjTerm<T> LjPair(T dx, T dy, T dz, T sigma2, T eps4, T eps24, T rc, T rc2,
                                  T u_shift, T s_rc, int shift) {
  LjTerm<T> out;
  const T r2 = dx * dx + dy * dy + dz * dz;
  if (!(r2 < rc2) || r2 == T(0)) return out;

  T inv_r2;
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, float>::value) {
    inv_r2 = __frcp_rn(r2);
  } else {
    inv_r2 = T(1) / r2;
  }
#else
  inv_r2 = T(1) / r2;
#endif
  const T sr2 = sigma2 * inv_r2;
  const T sr6 = sr2 * sr2 * sr2;
  T s = eps24 * inv_r2 * sr6 * (T(2) * sr6 - T(1));
  T u = eps4 * sr6 * (sr6 - T(1)) - u_shift;
  if (shift == 2) {
#if defined(__CUDA_ARCH__)
    const T r = sqrt(r2);
#else
    const T r = std::sqrt(r2);
#endif
    u += (r - rc) * s_rc * rc;
    s -= s_rc * rc / r;
  }

  out.inside = true;
  out.fx = s * dx;
  out.fy = s * dy;
  out.fz = s * dz;
  out.energy = T(0.5) * u;
  const T dot = dx * out.fx + dy * out.fy + dz * out.fz;
#if defined(__CUDA_ARCH__)
  out.pair_abs = T(0.5) * fabs(dot);
#else
  out.pair_abs = T(0.5) * std::abs(dot);
#endif
  out.wxx = T(0.5) * dx * out.fx;
  out.wyy = T(0.5) * dy * out.fy;
  out.wzz = T(0.5) * dz * out.fz;
  out.wxy = T(0.5) * dx * out.fy;
  out.wxz = T(0.5) * dx * out.fz;
  out.wyz = T(0.5) * dy * out.fz;
  return out;
}

}  // namespace tilemd
