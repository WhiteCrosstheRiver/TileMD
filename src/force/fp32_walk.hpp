#pragma once

#include <cmath>
#include <vector>

#include "force/lj_pair.hpp"
#include "force/shadow_delta.hpp"
#include "md/fused_atom.hpp"

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

}  // namespace tilemd
