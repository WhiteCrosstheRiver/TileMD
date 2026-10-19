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
  int rebuilds = 0;
};

// Device state for the FP32 kick-drift. The constructor uploads once.
// `Steps` only launches kernels. `SkinSteps` launches a short batch. The step
// that crosses half the skin still runs; later kernels in that batch return
// without moving atoms, and the host then rebuilds the list.
class Fp32Session {
 public:
  Fp32Session(const double* box, const double* x, const double* y, const double* z,
              const double* vx, const double* vy, const double* vz, const int* image_x,
              const int* image_y, const int* image_z, const double* x_ref, const double* y_ref,
              const double* z_ref, const int* nn, const int* nl, int n, int max_neigh,
              const Fp32Lj& pair, double dt, double mass, int threads_per_atom);
  ~Fp32Session();
  Fp32Session(const Fp32Session&) = delete;
  Fp32Session& operator=(const Fp32Session&) = delete;

  Fp32NveStats Steps(int steps);
  // One launch per step. A rebuild reorders the device arrays into Morton slots.
  Fp32NveStats SkinSteps(int steps, double skin);
  void Download(double* x, double* y, double* z, double* vx, double* vy, double* vz, int* image_x,
                int* image_y, int* image_z) const;

 private:
  struct Impl;
  void LaunchOne();
  Impl* impl_;
};

// Uploads, runs, and downloads. `max_disp2` is the largest squared displacement
// from the reference over this call.
void FusedAtomNve(const double* box, double* x, double* y, double* z, double* vx, double* vy,
                  double* vz, int* image_x, int* image_y, int* image_z, const double* x_ref,
                  const double* y_ref, const double* z_ref, const int* nn, const int* nl, int n,
                  int max_neigh, const Fp32Lj& pair, double dt, double mass, int steps,
                  int threads_per_atom, Fp32NveStats* stats);

}  // namespace tilemd
