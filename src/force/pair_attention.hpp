#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tilemd {

// One ordered pair, as seen by the i atom. `force` multiplies (r_i - r_j).
// `energy` is added with the full-list factor 1/2. `value` is summed onto i
// and handed to the potential after the pass (EAM density, for example).
struct PairOutput {
  double force = 0.0;
  double energy = 0.0;
  double value = 0.0;
};

struct AttentionResult {
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
  std::vector<double> energy;
  double potential_energy = 0.0;
  // xx, yy, zz, xy, xz, yz. Full list, including the factor 1/2.
  double virial[6] = {};
  std::vector<double> wxx;
  std::vector<double> wyy;
  std::vector<double> wzz;
  std::vector<double> wxy;
  std::vector<double> wxz;
  std::vector<double> wyz;
};

inline void AllocateAtomVirial(AttentionResult* out, int n) {
  const std::size_t count = static_cast<std::size_t>(n);
  out->wxx.assign(count, 0.0);
  out->wyy.assign(count, 0.0);
  out->wzz.assign(count, 0.0);
  out->wxy.assign(count, 0.0);
  out->wxz.assign(count, 0.0);
  out->wyz.assign(count, 0.0);
}

// Potential:
//   static constexpr int kNumPasses;
//   double Cutoff() const;
//   void Prepare(int n);
//   PairOutput PairScalar(int pass, double r2, int type_i, int type_j, int i, int j);
//   double AtomUpdate(int pass, int i, double value);  // returns energy to add
//
// Passes run in order. AtomUpdate of pass p finishes before pass p + 1, so an
// embedding derivative computed in pass 0 is visible to pass 1. The force is
// reconstructed as r_i * (sum s) - (sum s r_j), which is sum s (r_i - r_j).
template <class Potential>
AttentionResult EvaluateAttention(Potential& potential, const double* box, const double* x,
                                  const double* y, const double* z, const int* type, int n) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) throw std::invalid_argument("box length must be positive");
  }
  potential.Prepare(n);
  const double cut = potential.Cutoff();
  if (!(cut > 0.0)) throw std::invalid_argument("cutoff must be positive");
  const double cut2 = cut * cut;

  AttentionResult out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  out.energy.assign(static_cast<size_t>(n), 0.0);
  AllocateAtomVirial(&out, n);

  for (int pass = 0; pass < Potential::kNumPasses; ++pass) {
    std::vector<double> sum_x(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_y(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_z(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_1(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_value(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      const double xi = x[i];
      const double yi = y[i];
      const double zi = z[i];
      const int ti = type[i];
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const double dx = xi - x[j] - box[0] * std::rint((xi - x[j]) / box[0]);
        const double dy = yi - y[j] - box[1] * std::rint((yi - y[j]) / box[1]);
        const double dz = zi - z[j] - box[2] * std::rint((zi - z[j]) / box[2]);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 < cut2) || !(r2 > 0.0)) continue;
        const PairOutput term = potential.PairScalar(pass, r2, ti, type[j], i, j);
        const double xj = xi - dx;
        const double yj = yi - dy;
        const double zj = zi - dz;
        sum_x[static_cast<size_t>(i)] += term.force * xj;
        sum_y[static_cast<size_t>(i)] += term.force * yj;
        sum_z[static_cast<size_t>(i)] += term.force * zj;
        sum_1[static_cast<size_t>(i)] += term.force;
        sum_value[static_cast<size_t>(i)] += term.value;
        out.energy[static_cast<size_t>(i)] += 0.5 * term.energy;
        out.potential_energy += 0.5 * term.energy;
        const double fx = term.force * dx;
        const double fy = term.force * dy;
        const double fz = term.force * dz;
        const double wxx = 0.5 * dx * fx;
        const double wyy = 0.5 * dy * fy;
        const double wzz = 0.5 * dz * fz;
        const double wxy = 0.5 * dx * fy;
        const double wxz = 0.5 * dx * fz;
        const double wyz = 0.5 * dy * fz;
        out.wxx[static_cast<size_t>(i)] += wxx;
        out.wyy[static_cast<size_t>(i)] += wyy;
        out.wzz[static_cast<size_t>(i)] += wzz;
        out.wxy[static_cast<size_t>(i)] += wxy;
        out.wxz[static_cast<size_t>(i)] += wxz;
        out.wyz[static_cast<size_t>(i)] += wyz;
        out.virial[0] += wxx;
        out.virial[1] += wyy;
        out.virial[2] += wzz;
        out.virial[3] += wxy;
        out.virial[4] += wxz;
        out.virial[5] += wyz;
      }
    }
    for (int i = 0; i < n; ++i) {
      out.fx[static_cast<size_t>(i)] +=
          x[i] * sum_1[static_cast<size_t>(i)] - sum_x[static_cast<size_t>(i)];
      out.fy[static_cast<size_t>(i)] +=
          y[i] * sum_1[static_cast<size_t>(i)] - sum_y[static_cast<size_t>(i)];
      out.fz[static_cast<size_t>(i)] +=
          z[i] * sum_1[static_cast<size_t>(i)] - sum_z[static_cast<size_t>(i)];
      const double extra = potential.AtomUpdate(pass, i, sum_value[static_cast<size_t>(i)]);
      out.energy[static_cast<size_t>(i)] += extra;
      out.potential_energy += extra;
    }
  }
  return out;
}

}  // namespace tilemd
