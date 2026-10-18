#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define TILEMD_HD __host__ __device__
#else
#define TILEMD_HD
#endif

namespace tilemd {

// One Lennard-Jones pair, in the layout copied to the device. `defined == 0`
// means the type pair was not supplied.
struct LjRow {
  double sigma2 = 0.0;
  double eps4 = 0.0;
  double eps24 = 0.0;
  double rc = 0.0;
  double rc2 = 0.0;
  double u_shift = 0.0;
  double s_rc = 0.0;
  int defined = 0;
};

// `force_shift` selects the shifted-force scalar. The result is s(r^2) and the
// pair energy before the full-list factor 1/2.
TILEMD_HD inline void LjEvaluate(const LjRow& row, double r2, int force_shift, double* s,
                                 double* u) {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = row.sigma2 * inv_r2;
  const double sr6 = sr2 * sr2 * sr2;
  double force = row.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
  double energy = row.eps4 * sr6 * (sr6 - 1.0) - row.u_shift;
  if (force_shift) {
#if defined(__CUDA_ARCH__)
    const double r = sqrt(r2);
#else
    const double r = std::sqrt(r2);
#endif
    energy += (r - row.rc) * row.s_rc * row.rc;
    force -= row.s_rc * row.rc / r;
  }
  *s = force;
  *u = energy;
}

}  // namespace tilemd
