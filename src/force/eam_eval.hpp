#pragma once

#include <cmath>

#include "force/eam.hpp"

#ifndef TILEMD_HD
#if defined(__CUDACC__)
#define TILEMD_HD __host__ __device__
#else
#define TILEMD_HD
#endif
#endif

namespace tilemd {

TILEMD_HD inline double EamSqrt(double value) {
#if defined(__CUDA_ARCH__)
  return sqrt(value);
#else
  return std::sqrt(value);
#endif
}

// LAMMPS cubic on one knot table. Index 0 is unused; knots 1..n are live.
TILEMD_HD inline void SampleKnots(const EamKnot* spline, int n, double x, double inv_dx,
                                  int m_floor, double* value, double* derivative) {
  double p = x * inv_dx + 1.0;
  int m = static_cast<int>(p);
  if (m < m_floor) m = m_floor;
  if (m > n - 1) m = n - 1;
  p -= static_cast<double>(m);
  if (p > 1.0) p = 1.0;
  const double* c = spline[m].c;
  *derivative = (c[0] * p + c[1]) * p + c[2];
  *value = ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

TILEMD_HD inline int EamPairIndex(int type_i, int type_j) {
  int row = type_i;
  int col = type_j;
  if (row < col) {
    row = type_j;
    col = type_i;
  }
  return row * (row + 1) / 2 + col;
}

// Embedding energy. Writes F'(rho) through `derivative`.
TILEMD_HD inline double EamEmbed(const EamKnot* frho, int nrho, double rdrho, double rhomax,
                                 double value, double* derivative) {
  double embedding = 0.0;
  SampleKnots(frho, nrho, value, rdrho, 1, &embedding, derivative);
  if (value > rhomax) embedding += *derivative * (value - rhomax);
  return embedding;
}

TILEMD_HD inline PairOutput EamFuncflPair(const EamKnot* rho, const EamKnot* z2, const double* fp,
                                          int nr, double rdr, double cut2, int pass, double r2,
                                          int i, int j) {
  if (!(r2 < cut2) || !(r2 > 0.0)) return {};
  const double r = EamSqrt(r2);
  double density = 0.0;
  double density_prime = 0.0;
  SampleKnots(rho, nr, r, rdr, 1, &density, &density_prime);
  if (pass == 0) {
    PairOutput out;
    out.value = density;
    return out;
  }
  double z_squared = 0.0;
  double z_squared_prime = 0.0;
  SampleKnots(z2, nr, r, rdr, 1, &z_squared, &z_squared_prime);
  const double recip = 1.0 / r;
  const double phi = z_squared * recip;
  const double phip = z_squared_prime * recip - phi * recip;
  const double psip = fp[i] * density_prime + fp[j] * density_prime + phip;
  PairOutput out;
  out.force = -psip * recip;
  out.energy = phi;
  return out;
}

TILEMD_HD inline PairOutput EamSetflPair(const EamKnot* rho_j, const EamKnot* rho_i,
                                         const EamKnot* z2, const double* fp, int nr, double rdr,
                                         double cut2, int pass, double r2, int i, int j) {
  if (!(r2 < cut2) || !(r2 > 0.0)) return {};
  const double r = EamSqrt(r2);
  double density_j = 0.0;
  double density_prime_j = 0.0;
  SampleKnots(rho_j, nr, r, rdr, 1, &density_j, &density_prime_j);
  if (pass == 0) {
    PairOutput out;
    out.value = density_j;
    return out;
  }
  double density_i = 0.0;
  double density_prime_i = 0.0;
  SampleKnots(rho_i, nr, r, rdr, 1, &density_i, &density_prime_i);
  double z_squared = 0.0;
  double z_squared_prime = 0.0;
  SampleKnots(z2, nr, r, rdr, 1, &z_squared, &z_squared_prime);
  const double recip = 1.0 / r;
  const double phi = z_squared * recip;
  const double phip = z_squared_prime * recip - phi * recip;
  const double psip = fp[i] * density_prime_j + fp[j] * density_prime_i + phip;
  PairOutput out;
  out.force = -psip * recip;
  out.energy = phi;
  return out;
}

}  // namespace tilemd
