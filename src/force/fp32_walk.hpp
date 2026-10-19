#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "force/lj_pair.hpp"
#include "force/shadow_delta.hpp"
#include "md/fused_atom.hpp"
#include "neighbor/cell_list.hpp"
#include "neighbor/cluster_order.hpp"

namespace tilemd {

// Serial FP32 neighbor walk. This is the same pair math as the device kernel
// with one thread per atom, in neighbor-list order.
struct Fp32Walk {
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  double potential_energy = 0.0;
  double virial[6] = {};
  double pair_abs = 0.0;
};

inline void KickDriftAxis(double* coordinate, double* velocity, float force, double dt, double mass,
                          double length, int* image) {
  const double updated = *velocity + dt * static_cast<double>(force) / mass;
  *velocity = updated;
  *coordinate = WrapImage(*coordinate + dt * updated, length, image);
}

inline Fp32Walk HostFp32Force(const double* box, const double* x, const double* y, const double* z,
                              const int* nn, const int* nl, int n, const Fp32Lj& pair) {
  std::vector<float> local_x(n), local_y(n), local_z(n);
  std::vector<int> cell_x(n), cell_y(n), cell_z(n);
  for (int i = 0; i < n; ++i) {
    local_x[i] = ShadowLocal(x[i]);
    local_y[i] = ShadowLocal(y[i]);
    local_z[i] = ShadowLocal(z[i]);
    cell_x[i] = ShadowBin(x[i]);
    cell_y[i] = ShadowBin(y[i]);
    cell_z[i] = ShadowBin(z[i]);
  }
  int nfull[3];
  float remainder[3];
  for (int axis = 0; axis < 3; ++axis) {
    nfull[axis] = static_cast<int>(std::floor(box[axis] / kShadowCell));
    remainder[axis] =
        static_cast<float>(box[axis] - static_cast<double>(nfull[axis]) * kShadowCell);
  }

  Fp32Walk out;
  out.fx.assign(n, 0.f);
  out.fy.assign(n, 0.f);
  out.fz.assign(n, 0.f);
  for (int i = 0; i < n; ++i) {
    float fx = 0.f, fy = 0.f, fz = 0.f, energy = 0.f;
    float wxx = 0.f, wyy = 0.f, wzz = 0.f, wxy = 0.f, wxz = 0.f, wyz = 0.f, pair_abs = 0.f;
    const int count = nn[i];
    for (int k = 0; k < count; ++k) {
      const int j = nl[static_cast<long long>(k) * n + i];
      const float dx =
          ShadowMinImage(cell_x[i], local_x[i], cell_x[j], local_x[j], nfull[0], remainder[0]);
      const float dy =
          ShadowMinImage(cell_y[i], local_y[i], cell_y[j], local_y[j], nfull[1], remainder[1]);
      const float dz =
          ShadowMinImage(cell_z[i], local_z[i], cell_z[j], local_z[j], nfull[2], remainder[2]);
      const LjTerm<float> term =
          LjPair<float>(dx, dy, dz, pair.sigma2, pair.eps4, pair.eps24, pair.rc, pair.rc2,
                        pair.u_shift, pair.s_rc, pair.shift);
      fx += term.fx;
      fy += term.fy;
      fz += term.fz;
      energy += term.energy;
      wxx += term.wxx;
      wyy += term.wyy;
      wzz += term.wzz;
      wxy += term.wxy;
      wxz += term.wxz;
      wyz += term.wyz;
      pair_abs += term.pair_abs;
    }
    out.fx[i] = fx;
    out.fy[i] = fy;
    out.fz[i] = fz;
    out.potential_energy += static_cast<double>(energy);
    out.virial[0] += static_cast<double>(wxx);
    out.virial[1] += static_cast<double>(wyy);
    out.virial[2] += static_cast<double>(wzz);
    out.virial[3] += static_cast<double>(wxy);
    out.virial[4] += static_cast<double>(wxz);
    out.virial[5] += static_cast<double>(wyz);
    out.pair_abs += static_cast<double>(pair_abs);
  }
  return out;
}

struct Fp32SkinStats {
  int rebuilds = 0;
  float max_disp2 = 0.f;
};

// One thread's worth of the fused step: FP32 force, kick-drift, then a new
// cell list only after some atom has moved past half the skin. The list is
// built at rc + skin, and a rebuild puts every per-atom array into Morton
// slot order. Equality with the trigger does not rebuild.
inline Fp32SkinStats HostFp32SkinSteps(const double* box, double* x, double* y, double* z,
                                       double* vx, double* vy, double* vz, int* image_x,
                                       int* image_y, int* image_z, int n, const Fp32Lj& pair,
                                       double dt, double mass, double skin, int steps) {
  if (n <= 0 || steps <= 0) throw std::runtime_error("skin steps need atoms and steps");
  if (!(skin > 0.0) || !(dt > 0.0) || !(mass > 0.0)) {
    throw std::runtime_error("skin, dt, and mass must be positive");
  }
  const double reach = static_cast<double>(pair.rc) + skin;
  const float trigger2 = static_cast<float>((skin * 0.5) * (skin * 0.5));
  auto put_in_slot_order = [&]() {
    std::vector<double> xs(x, x + n);
    std::vector<double> ys(y, y + n);
    std::vector<double> zs(z, z + n);
    std::vector<double> vxs(vx, vx + n);
    std::vector<double> vys(vy, vy + n);
    std::vector<double> vzs(vz, vz + n);
    std::vector<int> ixs(image_x, image_x + n);
    std::vector<int> iys(image_y, image_y + n);
    std::vector<int> izs(image_z, image_z + n);
    const std::vector<int> order = MortonOrder(box, x, y, z, n, reach);
    ApplyOrder(&xs, order);
    ApplyOrder(&ys, order);
    ApplyOrder(&zs, order);
    ApplyOrder(&vxs, order);
    ApplyOrder(&vys, order);
    ApplyOrder(&vzs, order);
    ApplyOrder(&ixs, order);
    ApplyOrder(&iys, order);
    ApplyOrder(&izs, order);
    std::copy(xs.begin(), xs.end(), x);
    std::copy(ys.begin(), ys.end(), y);
    std::copy(zs.begin(), zs.end(), z);
    std::copy(vxs.begin(), vxs.end(), vx);
    std::copy(vys.begin(), vys.end(), vy);
    std::copy(vzs.begin(), vzs.end(), vz);
    std::copy(ixs.begin(), ixs.end(), image_x);
    std::copy(iys.begin(), iys.end(), image_y);
    std::copy(izs.begin(), izs.end(), image_z);
  };
  put_in_slot_order();
  std::vector<double> xref(x, x + n);
  std::vector<double> yref(y, y + n);
  std::vector<double> zref(z, z + n);
  CellList list = BuildCellList(box, x, y, z, n, reach);
  Fp32SkinStats stats;
  for (int step = 0; step < steps; ++step) {
    const Fp32Walk force = HostFp32Force(box, x, y, z, list.nn.data(), list.nl.data(), n, pair);
    float step_disp2 = 0.f;
    for (int i = 0; i < n; ++i) {
      KickDriftAxis(x + i, vx + i, force.fx[i], dt, mass, box[0], image_x + i);
      KickDriftAxis(y + i, vy + i, force.fy[i], dt, mass, box[1], image_y + i);
      KickDriftAxis(z + i, vz + i, force.fz[i], dt, mass, box[2], image_z + i);
      const double dx = OnReferenceBranch(x[i], xref[i], box[0]) - xref[i];
      const double dy = OnReferenceBranch(y[i], yref[i], box[1]) - yref[i];
      const double dz = OnReferenceBranch(z[i], zref[i], box[2]) - zref[i];
      step_disp2 = std::max(step_disp2, static_cast<float>(dx * dx + dy * dy + dz * dz));
    }
    stats.max_disp2 = std::max(stats.max_disp2, step_disp2);
    if (DisplacementNeedsRebuild(step_disp2, trigger2)) {
      put_in_slot_order();
      list = BuildCellList(box, x, y, z, n, reach);
      xref.assign(x, x + n);
      yref.assign(y, y + n);
      zref.assign(z, z + n);
      ++stats.rebuilds;
    }
  }
  return stats;
}

}  // namespace tilemd
