#pragma once

#include <vector>

namespace tilemd {

// FP32 Lennard-Jones constants. Prepare them in double, then store float.
struct Fp32Lj {
  float sigma2 = 1.f;
  float eps4 = 4.f;
  float eps24 = 24.f;
  float rc = 2.5f;
  float rc2 = 6.25f;
  float u_shift = 0.f;
  float s_rc = 0.f;
  int shift = 0;
};

struct Fp32Force {
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  double potential_energy = 0.0;
  double virial[6] = {};
  double pair_abs = 0.0;
};

// `nn[i]` is the neighbor count. `nl[k * n + i]` is the neighbor slot.
// `threads_per_atom` is 1, 2, 4, or 8. Block size is 128.
Fp32Force FusedAtomForce(const double* box, const double* x, const double* y, const double* z,
                         const int* nn, const int* nl, int n, int max_neigh, const Fp32Lj& pair,
                         int threads_per_atom);

struct Fp32NveStats {
  double max_disp2 = 0.0;
};

// One launch per step. FP32 pair force, then symplectic Euler on the double
// master coordinates. The neighbor list is not rebuilt.
void FusedAtomNve(const double* box, double* x, double* y, double* z, double* vx, double* vy,
                  double* vz, int* image_x, int* image_y, int* image_z, const double* x_ref,
                  const double* y_ref, const double* z_ref, const int* nn, const int* nl, int n,
                  int max_neigh, const Fp32Lj& pair, double dt, double mass, int steps,
                  int threads_per_atom, Fp32NveStats* stats);

}  // namespace tilemd
