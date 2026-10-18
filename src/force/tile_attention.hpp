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
  AllocateAtomVirial(&out, n);

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
            const double wxx = 0.5 * dx * fx;
            const double wyy = 0.5 * dy * fy;
            const double wzz = 0.5 * dz * fz;
            const double wxy = 0.5 * dx * fy;
            const double wxz = 0.5 * dx * fz;
            const double wyz = 0.5 * dy * fz;
            out.wxx[static_cast<size_t>(gid_i)] += wxx;
            out.wyy[static_cast<size_t>(gid_i)] += wyy;
            out.wzz[static_cast<size_t>(gid_i)] += wzz;
            out.wxy[static_cast<size_t>(gid_i)] += wxy;
            out.wxz[static_cast<size_t>(gid_i)] += wxz;
            out.wyz[static_cast<size_t>(gid_i)] += wyz;
            out.virial[0] += wxx;
            out.virial[1] += wyy;
            out.virial[2] += wzz;
            out.virial[3] += wxy;
            out.virial[4] += wxz;
            out.virial[5] += wyz;
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

// Host replay of the device lane map: lane = i_local + 8 * j_sub, four
// j-clusters at a time, then a sum over the four lanes that own one i atom.
// Pass p finishes `AtomUpdate` before pass p + 1, so an EAM embedding
// derivative is visible to the force pass. This function is not a GPU launch.
template <class Potential>
AttentionResult WarpLaneAttention(Potential& potential, const double* box, const double* x,
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
  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, cut + skin);
  const ClusterPairs list = BuildClusterPairs(grid);

  AttentionResult out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  out.energy.assign(static_cast<size_t>(n), 0.0);
  AllocateAtomVirial(&out, n);

  struct Lane {
    int ti = kMaskType;
    int gi = -1;
    double xi = 0.0;
    double yi = 0.0;
    double zi = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double a1 = 0.0;
    double av = 0.0;
    double ae = 0.0;
    double axx = 0.0;
    double ayy = 0.0;
    double azz = 0.0;
    double axy = 0.0;
    double axz = 0.0;
    double ayz = 0.0;
  };

  for (int pass = 0; pass < Potential::kNumPasses; ++pass) {
    std::vector<double> sum_value(static_cast<size_t>(n), 0.0);
    for (int ci = 0; ci < grid.ncluster; ++ci) {
      Lane lane[32];
      for (int lane_id = 0; lane_id < 32; ++lane_id) {
        const int i_local = lane_id % kClusterSize;
        const int slot_i = ci * kClusterSize + i_local;
        lane[lane_id].ti = grid.type[static_cast<size_t>(slot_i)];
        lane[lane_id].gi = grid.gid[static_cast<size_t>(slot_i)];
        lane[lane_id].xi = grid.x[static_cast<size_t>(slot_i)];
        lane[lane_id].yi = grid.y[static_cast<size_t>(slot_i)];
        lane[lane_id].zi = grid.z[static_cast<size_t>(slot_i)];
      }
      const int begin = list.ci_start[static_cast<size_t>(ci)];
      const int end = list.ci_start[static_cast<size_t>(ci) + 1];
      for (int pair0 = begin; pair0 < end; pair0 += 4) {
        double xj[32] = {};
        double yj[32] = {};
        double zj[32] = {};
        int tj[32];
        int gj[32];
        int active[32];
        int other[32];
        int sx[32];
        int sy[32];
        int sz[32];
        for (int lane_id = 0; lane_id < 32; ++lane_id) {
          const int i_local = lane_id % kClusterSize;
          const int j_sub = lane_id / kClusterSize;
          const int pair = pair0 + j_sub;
          active[lane_id] = pair < end ? 1 : 0;
          other[lane_id] = active[lane_id] ? list.cj[static_cast<size_t>(pair)] : 0;
          sx[lane_id] = 0;
          sy[lane_id] = 0;
          sz[lane_id] = 0;
          if (active[lane_id]) {
            const int shift = list.shift[static_cast<size_t>(pair)];
            sx[lane_id] = shift % 3 - 1;
            sy[lane_id] = (shift / 3) % 3 - 1;
            sz[lane_id] = shift / 9 - 1;
          }
          const int slot_j = other[lane_id] * kClusterSize + i_local;
          tj[lane_id] = active[lane_id] ? grid.type[static_cast<size_t>(slot_j)] : kMaskType;
          gj[lane_id] = active[lane_id] ? grid.gid[static_cast<size_t>(slot_j)] : -1;
          if (active[lane_id] && tj[lane_id] != kMaskType) {
            xj[lane_id] = grid.x[static_cast<size_t>(slot_j)] + sx[lane_id] * grid.box[0];
            yj[lane_id] = grid.y[static_cast<size_t>(slot_j)] + sy[lane_id] * grid.box[1];
            zj[lane_id] = grid.z[static_cast<size_t>(slot_j)] + sz[lane_id] * grid.box[2];
          }
        }
        for (int atom = 0; atom < kClusterSize; ++atom) {
          for (int lane_id = 0; lane_id < 32; ++lane_id) {
            const int i_local = lane_id % kClusterSize;
            const int j_sub = lane_id / kClusterSize;
            const int source = atom + kClusterSize * j_sub;
            const int bt = tj[source];
            if (!active[lane_id] || lane[lane_id].ti == kMaskType || bt == kMaskType) continue;
            if (ci == other[lane_id] && atom == i_local && sx[lane_id] == 0 && sy[lane_id] == 0 &&
                sz[lane_id] == 0) {
              continue;
            }
            const double dx = lane[lane_id].xi - xj[source];
            const double dy = lane[lane_id].yi - yj[source];
            const double dz = lane[lane_id].zi - zj[source];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0)) continue;
            const PairOutput term =
                potential.PairScalar(pass, r2, lane[lane_id].ti, bt, lane[lane_id].gi, gj[source]);
            lane[lane_id].ax += term.force * (lane[lane_id].xi - dx);
            lane[lane_id].ay += term.force * (lane[lane_id].yi - dy);
            lane[lane_id].az += term.force * (lane[lane_id].zi - dz);
            lane[lane_id].a1 += term.force;
            lane[lane_id].av += term.value;
            lane[lane_id].ae += 0.5 * term.energy;
            const double fix = term.force * dx;
            const double fiy = term.force * dy;
            const double fiz = term.force * dz;
            lane[lane_id].axx += 0.5 * dx * fix;
            lane[lane_id].ayy += 0.5 * dy * fiy;
            lane[lane_id].azz += 0.5 * dz * fiz;
            lane[lane_id].axy += 0.5 * dx * fiy;
            lane[lane_id].axz += 0.5 * dx * fiz;
            lane[lane_id].ayz += 0.5 * dy * fiz;
          }
        }
      }
      for (int i_local = 0; i_local < kClusterSize; ++i_local) {
        const Lane& owner = lane[i_local];
        if (owner.ti == kMaskType || owner.gi < 0) continue;
        double rx = 0.0;
        double ry = 0.0;
        double rz = 0.0;
        double r1 = 0.0;
        double rv = 0.0;
        double re = 0.0;
        double rxx = 0.0;
        double ryy = 0.0;
        double rzz = 0.0;
        double rxy = 0.0;
        double rxz = 0.0;
        double ryz = 0.0;
        for (int j_sub = 0; j_sub < 4; ++j_sub) {
          const Lane& src = lane[i_local + kClusterSize * j_sub];
          rx += src.ax;
          ry += src.ay;
          rz += src.az;
          r1 += src.a1;
          rv += src.av;
          re += src.ae;
          rxx += src.axx;
          ryy += src.ayy;
          rzz += src.azz;
          rxy += src.axy;
          rxz += src.axz;
          ryz += src.ayz;
        }
        const int gi = owner.gi;
        out.fx[static_cast<size_t>(gi)] += owner.xi * r1 - rx;
        out.fy[static_cast<size_t>(gi)] += owner.yi * r1 - ry;
        out.fz[static_cast<size_t>(gi)] += owner.zi * r1 - rz;
        out.energy[static_cast<size_t>(gi)] += re;
        out.potential_energy += re;
        sum_value[static_cast<size_t>(gi)] = rv;
        out.wxx[static_cast<size_t>(gi)] += rxx;
        out.wyy[static_cast<size_t>(gi)] += ryy;
        out.wzz[static_cast<size_t>(gi)] += rzz;
        out.wxy[static_cast<size_t>(gi)] += rxy;
        out.wxz[static_cast<size_t>(gi)] += rxz;
        out.wyz[static_cast<size_t>(gi)] += ryz;
        out.virial[0] += rxx;
        out.virial[1] += ryy;
        out.virial[2] += rzz;
        out.virial[3] += rxy;
        out.virial[4] += rxz;
        out.virial[5] += ryz;
      }
    }
    for (int i = 0; i < n; ++i) {
      const double extra = potential.AtomUpdate(pass, i, sum_value[static_cast<size_t>(i)]);
      out.energy[static_cast<size_t>(i)] += extra;
      out.potential_energy += extra;
    }
  }
  return out;
}

}  // namespace tilemd
