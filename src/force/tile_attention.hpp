#pragma once

#include <stdexcept>
#include <vector>

#include "force/pair_attention.hpp"
#include "neighbor/cluster_list.hpp"

namespace tilemd {

// The pair-attention passes, walked on the cluster-pair list instead of the
// full pair loop. `AtomUpdate` of pass p still finishes before pass p + 1.
// `skin` only widens the list; the potential cutoff is unchanged.
template <class Potential>
AttentionResult EvaluateAttentionTiles(Potential& potential, const double* box, const double* x,
                                       const double* y, const double* z, const int* type, int n,
                                       double skin) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  if (skin < 0.0) throw std::invalid_argument("skin must be non-negative");
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) throw std::invalid_argument("box length must be positive");
  }
  potential.Prepare(n);
  const double cut = potential.Cutoff();
  if (!(cut > 0.0)) throw std::invalid_argument("cutoff must be positive");
  const double cut2 = cut * cut;
  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, cut + skin);
  const ClusterPairs list = BuildClusterPairs(grid);

  std::vector<double> origin_x(static_cast<size_t>(n));
  std::vector<double> origin_y(static_cast<size_t>(n));
  std::vector<double> origin_z(static_cast<size_t>(n));
  for (int slot = 0; slot < static_cast<int>(grid.gid.size()); ++slot) {
    const int gid = grid.gid[static_cast<size_t>(slot)];
    if (gid < 0) continue;
    origin_x[static_cast<size_t>(gid)] = grid.x[static_cast<size_t>(slot)];
    origin_y[static_cast<size_t>(gid)] = grid.y[static_cast<size_t>(slot)];
    origin_z[static_cast<size_t>(gid)] = grid.z[static_cast<size_t>(slot)];
  }

  AttentionResult out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  out.energy.assign(static_cast<size_t>(n), 0.0);

  for (int pass = 0; pass < Potential::kNumPasses; ++pass) {
    std::vector<double> sum_x(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_y(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_z(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_1(static_cast<size_t>(n), 0.0);
    std::vector<double> sum_value(static_cast<size_t>(n), 0.0);
    for (int ci = 0; ci < grid.ncluster; ++ci) {
      for (int k = list.ci_start[static_cast<size_t>(ci)];
           k < list.ci_start[static_cast<size_t>(ci) + 1]; ++k) {
        const int cj = list.cj[static_cast<size_t>(k)];
        const int shift = list.shift[static_cast<size_t>(k)];
        const int sx = shift % 3 - 1;
        const int sy = (shift / 3) % 3 - 1;
        const int sz = shift / 9 - 1;
        for (int local_i = 0; local_i < kClusterSize; ++local_i) {
          const int slot_i = ci * kClusterSize + local_i;
          if (grid.type[static_cast<size_t>(slot_i)] == kMaskType) continue;
          const int gid_i = grid.gid[static_cast<size_t>(slot_i)];
          const int ti = grid.type[static_cast<size_t>(slot_i)];
          const double xi = grid.x[static_cast<size_t>(slot_i)];
          const double yi = grid.y[static_cast<size_t>(slot_i)];
          const double zi = grid.z[static_cast<size_t>(slot_i)];
          for (int local_j = 0; local_j < kClusterSize; ++local_j) {
            const int slot_j = cj * kClusterSize + local_j;
            if (grid.type[static_cast<size_t>(slot_j)] == kMaskType) continue;
            if (ci == cj && local_i == local_j && sx == 0 && sy == 0 && sz == 0) continue;
            const int gid_j = grid.gid[static_cast<size_t>(slot_j)];
            const double dx =
                xi - (grid.x[static_cast<size_t>(slot_j)] + static_cast<double>(sx) * grid.box[0]);
            const double dy =
                yi - (grid.y[static_cast<size_t>(slot_j)] + static_cast<double>(sy) * grid.box[1]);
            const double dz =
                zi - (grid.z[static_cast<size_t>(slot_j)] + static_cast<double>(sz) * grid.box[2]);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 < cut2) || !(r2 > 0.0)) continue;
            const PairOutput term = potential.PairScalar(
                pass, r2, ti, grid.type[static_cast<size_t>(slot_j)], gid_i, gid_j);
            const double xj = xi - dx;
            const double yj = yi - dy;
            const double zj = zi - dz;
            sum_x[static_cast<size_t>(gid_i)] += term.force * xj;
            sum_y[static_cast<size_t>(gid_i)] += term.force * yj;
            sum_z[static_cast<size_t>(gid_i)] += term.force * zj;
            sum_1[static_cast<size_t>(gid_i)] += term.force;
            sum_value[static_cast<size_t>(gid_i)] += term.value;
            out.energy[static_cast<size_t>(gid_i)] += 0.5 * term.energy;
            out.potential_energy += 0.5 * term.energy;
            const double fx = term.force * dx;
            const double fy = term.force * dy;
            const double fz = term.force * dz;
            out.virial[0] += 0.5 * dx * fx;
            out.virial[1] += 0.5 * dy * fy;
            out.virial[2] += 0.5 * dz * fz;
            out.virial[3] += 0.5 * dx * fy;
            out.virial[4] += 0.5 * dx * fz;
            out.virial[5] += 0.5 * dy * fz;
          }
        }
      }
    }
    for (int i = 0; i < n; ++i) {
      out.fx[static_cast<size_t>(i)] +=
          origin_x[static_cast<size_t>(i)] * sum_1[static_cast<size_t>(i)] -
          sum_x[static_cast<size_t>(i)];
      out.fy[static_cast<size_t>(i)] +=
          origin_y[static_cast<size_t>(i)] * sum_1[static_cast<size_t>(i)] -
          sum_y[static_cast<size_t>(i)];
      out.fz[static_cast<size_t>(i)] +=
          origin_z[static_cast<size_t>(i)] * sum_1[static_cast<size_t>(i)] -
          sum_z[static_cast<size_t>(i)];
      const double extra = potential.AtomUpdate(pass, i, sum_value[static_cast<size_t>(i)]);
      out.energy[static_cast<size_t>(i)] += extra;
      out.potential_energy += extra;
    }
  }
  return out;
}

}  // namespace tilemd
