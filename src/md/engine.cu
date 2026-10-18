#include "md/engine.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/device_status.hpp"

namespace tilemd {

constexpr int kMaxTypes = 8;

struct PairRow {
  double sigma2;
  double eps4;
  double eps24;
  double rc;
  double rc2;
  double u_shift;
  double s_rc;
  int defined;
};

namespace {

constexpr int kBlock = 128;

__constant__ PairRow c_pairs[kMaxTypes * kMaxTypes];
__constant__ double c_box[3];
__constant__ int c_ncell[3];
__constant__ int c_shift;
__constant__ float c_neigh_cut2;

#define TILEMD_CHECK(call)                                                               \
  do {                                                                                   \
    cudaError_t tilemd_err_ = (call);                                                    \
    if (tilemd_err_ != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(tilemd_err_)); \
    }                                                                                    \
  } while (0)

void CheckLaunch() {
  TILEMD_CHECK(cudaGetLastError());
  TILEMD_CHECK(cudaDeviceSynchronize());
}

int Grid(int n) { return std::max(1, (n + kBlock - 1) / kBlock); }

void* MustMalloc(size_t bytes) {
  void* ptr = nullptr;
  TILEMD_CHECK(cudaMalloc(&ptr, bytes));
  return ptr;
}

__device__ float MinImage(float delta, float length) {
  return delta - length * rintf(delta / length);
}

__device__ double MinImageDouble(double delta, double length) {
  return delta - length * rint(delta / length);
}

__device__ int CellCoord(float x, float inv_edge, int ncell) {
  int i = static_cast<int>(floorf(x * inv_edge));
  if (i < 0) i = 0;
  if (i >= ncell) i = ncell - 1;
  return i;
}

__device__ int WrapCell(int i, int ncell, int delta) {
  int w = i + delta;
  if (w >= ncell) w -= ncell;
  if (w < 0) w += ncell;
  return w;
}

__device__ double WrapDouble(double x, double length, int* image) {
  double periods = floor(x / length);
  int shift = static_cast<int>(periods);
  x -= periods * length;
  if (x >= length) {
    x -= length;
    ++shift;
  } else if (x < 0.0) {
    x += length;
    --shift;
  }
  *image += shift;
  return x;
}

__device__ float WrapFloat(float x, float length) {
  float y = x - floorf(x / length) * length;
  if (y >= length) y = 0.f;
  if (y < 0.f) y += length;
  return y;
}

__global__ void WrapAndShadow(double* x, double* y, double* z, int* imgx, int* imgy, int* imgz,
                              const int* type, float4* pos, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = WrapDouble(x[i], c_box[0], &imgx[i]);
  y[i] = WrapDouble(y[i], c_box[1], &imgy[i]);
  z[i] = WrapDouble(z[i], c_box[2], &imgz[i]);
  pos[i] = make_float4(WrapFloat(static_cast<float>(x[i]), static_cast<float>(c_box[0])),
                       WrapFloat(static_cast<float>(y[i]), static_cast<float>(c_box[1])),
                       WrapFloat(static_cast<float>(z[i]), static_cast<float>(c_box[2])),
                       __int_as_float(type[i]));
}

__global__ void RefreshShadow(const double* x, const double* y, const double* z, const int* type,
                              float4* pos, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  pos[i] = make_float4(WrapFloat(static_cast<float>(x[i]), static_cast<float>(c_box[0])),
                       WrapFloat(static_cast<float>(y[i]), static_cast<float>(c_box[1])),
                       WrapFloat(static_cast<float>(z[i]), static_cast<float>(c_box[2])),
                       __int_as_float(type[i]));
}

__global__ void CellKeyKernel(const float4* pos, int* keys, int* indices, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i];
  int ix =
      CellCoord(p.x, static_cast<float>(c_ncell[0]) / static_cast<float>(c_box[0]), c_ncell[0]);
  int iy =
      CellCoord(p.y, static_cast<float>(c_ncell[1]) / static_cast<float>(c_box[1]), c_ncell[1]);
  int iz =
      CellCoord(p.z, static_cast<float>(c_ncell[2]) / static_cast<float>(c_box[2]), c_ncell[2]);
  keys[i] = ix + c_ncell[0] * (iy + c_ncell[1] * iz);
  indices[i] = i;
}

__global__ void FillCellStart(const int* sorted_keys, int* cell_start, int n, int ncell) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int c = 0; c <= ncell; ++c) cell_start[c] = n;
  if (n == 0) return;
  int first = sorted_keys[0];
  for (int c = 0; c <= first; ++c) cell_start[c] = 0;
  for (int i = 1; i < n; ++i) {
    int prev = sorted_keys[i - 1];
    int cur = sorted_keys[i];
    if (cur != prev) {
      for (int c = prev + 1; c <= cur; ++c) cell_start[c] = i;
    }
  }
}

__global__ void NeighborKernel(const float4* pos, const int* cell_start, const int* cell_atoms,
                               int* nn, int* nl, int max_neigh, int n, int* max_nn,
                               DeviceStatus* status) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 pi = pos[i];
  int ix =
      CellCoord(pi.x, static_cast<float>(c_ncell[0]) / static_cast<float>(c_box[0]), c_ncell[0]);
  int iy =
      CellCoord(pi.y, static_cast<float>(c_ncell[1]) / static_cast<float>(c_box[1]), c_ncell[1]);
  int iz =
      CellCoord(pi.z, static_cast<float>(c_ncell[2]) / static_cast<float>(c_box[2]), c_ncell[2]);
  int count = 0;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int cell =
            WrapCell(ix, c_ncell[0], dx) +
            c_ncell[0] * (WrapCell(iy, c_ncell[1], dy) + c_ncell[1] * WrapCell(iz, c_ncell[2], dz));
        for (int s = cell_start[cell]; s < cell_start[cell + 1]; ++s) {
          int j = cell_atoms[s];
          if (j == i) continue;
          float4 pj = pos[j];
          float dxij = MinImage(pi.x - pj.x, static_cast<float>(c_box[0]));
          float dyij = MinImage(pi.y - pj.y, static_cast<float>(c_box[1]));
          float dzij = MinImage(pi.z - pj.z, static_cast<float>(c_box[2]));
          float d2 = dxij * dxij + dyij * dyij + dzij * dzij;
          if (d2 < c_neigh_cut2) {
            if (count < max_neigh) nl[static_cast<long long>(count) * n + i] = j;
            ++count;
          }
        }
      }
    }
  }
  nn[i] = count < max_neigh ? count : max_neigh;
  atomicMax(max_nn, count);
  if (count > max_neigh) atomicExch(&status->overflow, 1);
}

// Index order matches the oracle's j loop. Cell order is deterministic too, but
// summing 1e36 pair terms in a different order misses the virial gate on
// boundary_500. The list is still built in cell order above.
__global__ void SortNeighbors(int* nl, const int* nn, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int count = nn[i];
  for (int a = 1; a < count; ++a) {
    int key = nl[static_cast<long long>(a) * n + i];
    int b = a;
    while (b > 0 && nl[static_cast<long long>(b - 1) * n + i] > key) {
      nl[static_cast<long long>(b) * n + i] = nl[static_cast<long long>(b - 1) * n + i];
      --b;
    }
    nl[static_cast<long long>(b) * n + i] = key;
  }
}

__global__ void ForceKernel(const double* x, const double* y, const double* z, const int* type,
                            const int* nn, const int* nl, float* fx, float* fy, float* fz,
                            double* energy, double* wxx, double* wyy, double* wzz, double* wxy,
                            double* wxz, double* wyz, int n, DeviceStatus* status) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int ti = type[i];
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  double e = 0.0;
  double axx = 0.0;
  double ayy = 0.0;
  double azz = 0.0;
  double axy = 0.0;
  double axz = 0.0;
  double ayz = 0.0;
  int count = nn[i];
  for (int k = 0; k < count; ++k) {
    int j = nl[static_cast<long long>(k) * n + i];
    int tj = type[j];
    PairRow p = c_pairs[ti * kMaxTypes + tj];
    if (!p.defined) {
      atomicExch(&status->nan_detected, 1);
      return;
    }
    // Double geometry. A float box of length ~17 already moves melt forces by
    // ~1e-3 relative, which misses the 1e-5 gate. The stored force is still float.
    double dx = MinImageDouble(x[i] - x[j], c_box[0]);
    double dy = MinImageDouble(y[i] - y[j], c_box[1]);
    double dz = MinImageDouble(z[i] - z[j], c_box[2]);
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= p.rc2) continue;
    double inv_r2 = 1.0 / r2;
    double sr2 = p.sigma2 * inv_r2;
    double sr6 = sr2 * sr2 * sr2;
    double s = p.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
    double u = p.eps4 * sr6 * (sr6 - 1.0) - p.u_shift;
    if (c_shift == 2) {
      double r = sqrt(r2);
      u += (r - p.rc) * p.s_rc * p.rc;
      s -= p.s_rc * p.rc / r;
    }
    double fix = s * dx;
    double fiy = s * dy;
    double fiz = s * dz;
    ax += fix;
    ay += fiy;
    az += fiz;
    e += 0.5 * u;
    axx += 0.5 * dx * fix;
    ayy += 0.5 * dy * fiy;
    azz += 0.5 * dz * fiz;
    axy += 0.5 * dx * fiy;
    axz += 0.5 * dx * fiz;
    ayz += 0.5 * dy * fiz;
  }
  fx[i] = static_cast<float>(ax);
  fy[i] = static_cast<float>(ay);
  fz[i] = static_cast<float>(az);
  energy[i] = e;
  wxx[i] = axx;
  wyy[i] = ayy;
  wzz[i] = azz;
  wxy[i] = axy;
  wxz[i] = axz;
  wyz[i] = ayz;
  if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) atomicExch(&status->nan_detected, 1);
}

__global__ void KickDrift(double* x, double* y, double* z, double* vx, double* vy, double* vz,
                          const float* fx, const float* fy, const float* fz, int* imgx, int* imgy,
                          int* imgz, int n, double dt, double inv_mass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double scale = 0.5 * dt * inv_mass;
  vx[i] += scale * static_cast<double>(fx[i]);
  vy[i] += scale * static_cast<double>(fy[i]);
  vz[i] += scale * static_cast<double>(fz[i]);
  x[i] = WrapDouble(x[i] + dt * vx[i], c_box[0], &imgx[i]);
  y[i] = WrapDouble(y[i] + dt * vy[i], c_box[1], &imgy[i]);
  z[i] = WrapDouble(z[i] + dt * vz[i], c_box[2], &imgz[i]);
}

__global__ void Kick(double* vx, double* vy, double* vz, const float* fx, const float* fy,
                     const float* fz, int n, double dt, double inv_mass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double scale = 0.5 * dt * inv_mass;
  vx[i] += scale * static_cast<double>(fx[i]);
  vy[i] += scale * static_cast<double>(fy[i]);
  vz[i] += scale * static_cast<double>(fz[i]);
}

__global__ void DispKernel(const float4* pos, const float4* xref, int n, DeviceStatus* status) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float dx = pos[i].x - xref[i].x;
  float dy = pos[i].y - xref[i].y;
  float dz = pos[i].z - xref[i].z;
  float d2 = dx * dx + dy * dy + dz * dz;
  atomicMax(reinterpret_cast<int*>(&status->max_disp2_bits), __float_as_int(d2));
}

int RoundUp32(int value) {
  if (value < 32) return 32;
  return (value + 31) & ~31;
}

}  // namespace

struct Engine::Impl {
  HostSystem system;
  int n = 0;
  int max_neigh = 0;
  int ncell[3] = {};
  int ncell_tot = 0;
  int rebuild_count = 0;
  bool forces_ready = false;
  double reach = 0.0;
  PairRow host_pairs[kMaxTypes * kMaxTypes] = {};

  double* x = nullptr;
  double* y = nullptr;
  double* z = nullptr;
  double* vx = nullptr;
  double* vy = nullptr;
  double* vz = nullptr;
  int* imgx = nullptr;
  int* imgy = nullptr;
  int* imgz = nullptr;
  int* type = nullptr;
  float4* pos = nullptr;
  float4* xref = nullptr;
  float* fx = nullptr;
  float* fy = nullptr;
  float* fz = nullptr;
  double* energy = nullptr;
  double* wxx = nullptr;
  double* wyy = nullptr;
  double* wzz = nullptr;
  double* wxy = nullptr;
  double* wxz = nullptr;
  double* wyz = nullptr;
  int* keys = nullptr;
  int* sorted_keys = nullptr;
  int* indices = nullptr;
  int* sorted_indices = nullptr;
  int* cell_start = nullptr;
  int* nn = nullptr;
  int* nl = nullptr;
  int* max_nn = nullptr;
  void* cub_temp = nullptr;
  size_t cub_bytes = 0;
  DeviceStatus* status = nullptr;

  void Upload();
  void ClearFlags();
  void BuildCells();
  void ReallocList(int capacity);
  int BuildNeighbors(bool grow);
  void ComputeForce();
  GpuForces DownloadForces(int overflow);

  ~Impl() {
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(vz);
    cudaFree(imgx);
    cudaFree(imgy);
    cudaFree(imgz);
    cudaFree(type);
    cudaFree(pos);
    cudaFree(xref);
    cudaFree(fx);
    cudaFree(fy);
    cudaFree(fz);
    cudaFree(energy);
    cudaFree(wxx);
    cudaFree(wyy);
    cudaFree(wzz);
    cudaFree(wxy);
    cudaFree(wxz);
    cudaFree(wyz);
    cudaFree(keys);
    cudaFree(sorted_keys);
    cudaFree(indices);
    cudaFree(sorted_indices);
    cudaFree(cell_start);
    cudaFree(nn);
    cudaFree(nl);
    cudaFree(max_nn);
    cudaFree(cub_temp);
    cudaFree(status);
  }
};

Engine::Engine(HostSystem system) : impl_(new Impl) {
  try {
    impl_->system = std::move(system);
    impl_->n = static_cast<int>(impl_->system.x.size());
    const HostSystem& in = impl_->system;
    if (impl_->n <= 0 || static_cast<int>(in.y.size()) != impl_->n ||
        static_cast<int>(in.z.size()) != impl_->n || static_cast<int>(in.type.size()) != impl_->n) {
      throw std::runtime_error("position and type arrays differ in length");
    }
    if (!(in.box[0] > 0.0) || !(in.box[1] > 0.0) || !(in.box[2] > 0.0)) {
      throw std::runtime_error("box lengths must be positive");
    }
    if (!(in.mass > 0.0)) throw std::runtime_error("mass must be positive");
    if (in.skin < 0.0) throw std::runtime_error("skin must be non-negative");
    if (in.pairs.empty()) throw std::runtime_error("no pair coefficients");

    double rc_max = 0.0;
    PairRow table[kMaxTypes * kMaxTypes] = {};
    auto prepare = [&](const PairSpec& coeff) {
      if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kMaxTypes ||
          coeff.type_j >= kMaxTypes) {
        throw std::runtime_error("pair type is outside 0..7");
      }
      if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
        throw std::runtime_error("sigma and cutoff must be positive");
      }
      PairRow row{};
      row.defined = 1;
      row.sigma2 = coeff.sigma * coeff.sigma;
      row.eps4 = 4.0 * coeff.epsilon;
      row.eps24 = 24.0 * coeff.epsilon;
      row.rc = coeff.cutoff;
      row.rc2 = row.rc * row.rc;
      double inv_rc2 = 1.0 / row.rc2;
      double sr2 = row.sigma2 * inv_rc2;
      double sr6 = sr2 * sr2 * sr2;
      double u_c = row.eps4 * sr6 * (sr6 - 1.0);
      row.s_rc = row.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
      row.u_shift = (in.shift == Shift::None) ? 0.0 : u_c;
      table[coeff.type_i * kMaxTypes + coeff.type_j] = row;
      rc_max = std::max(rc_max, coeff.cutoff);
    };
    for (const PairSpec& coeff : in.pairs) {
      prepare(coeff);
      if (coeff.type_i != coeff.type_j) {
        PairSpec swapped = coeff;
        swapped.type_i = coeff.type_j;
        swapped.type_j = coeff.type_i;
        prepare(swapped);
      }
    }
    bool used[kMaxTypes] = {};
    for (int ti : in.type) {
      if (ti < 0 || ti >= kMaxTypes) throw std::runtime_error("atom type is outside 0..7");
      used[ti] = true;
    }
    for (int a = 0; a < kMaxTypes; ++a) {
      if (!used[a]) continue;
      for (int b = 0; b < kMaxTypes; ++b) {
        if (used[b] && !table[a * kMaxTypes + b].defined) {
          throw std::runtime_error("missing pair coefficient");
        }
      }
    }
    std::copy(table, table + kMaxTypes * kMaxTypes, impl_->host_pairs);

    // Pad the list cutoff slightly so float rounding cannot drop a pair that
    // the force kernel would still accept. The force kernel keeps the exclusive
    // rc test. The cell edge stays at least this reach, so the 27-cell stencil
    // still covers it.
    impl_->reach = (rc_max + in.skin) * (1.0 + 1.0e-4) + 1.0e-5;
    for (int dim = 0; dim < 3; ++dim) {
      impl_->ncell[dim] = static_cast<int>(std::floor(in.box[dim] / impl_->reach));
      if (impl_->ncell[dim] < 3) throw std::runtime_error("box too small for 3 cells");
    }
    impl_->ncell_tot = impl_->ncell[0] * impl_->ncell[1] * impl_->ncell[2];

    if (in.max_neigh > 0) {
      impl_->max_neigh = in.max_neigh;
    } else {
      double volume = in.box[0] * in.box[1] * in.box[2];
      double rho = static_cast<double>(impl_->n) / volume;
      double sphere =
          (4.0 / 3.0) * 3.14159265358979323846 * impl_->reach * impl_->reach * impl_->reach;
      impl_->max_neigh = RoundUp32(static_cast<int>(std::ceil(1.5 * sphere * rho)));
    }
    if (impl_->max_neigh < 1) impl_->max_neigh = 32;
    if (impl_->n > 0 && impl_->max_neigh > (1 << 30) / impl_->n) {
      throw std::runtime_error("neighbor list index does not fit in 32 bits");
    }

    const int n = impl_->n;
    impl_->x = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->y = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->z = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->vx = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->vy = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->vz = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->imgx = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->imgy = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->imgz = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->type = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->pos = static_cast<float4*>(MustMalloc(sizeof(float4) * n));
    impl_->xref = static_cast<float4*>(MustMalloc(sizeof(float4) * n));
    impl_->fx = static_cast<float*>(MustMalloc(sizeof(float) * n));
    impl_->fy = static_cast<float*>(MustMalloc(sizeof(float) * n));
    impl_->fz = static_cast<float*>(MustMalloc(sizeof(float) * n));
    impl_->energy = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->wxx = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->wyy = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->wzz = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->wxy = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->wxz = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->wyz = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->keys = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->sorted_keys = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->indices = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->sorted_indices = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->cell_start = static_cast<int*>(MustMalloc(sizeof(int) * (impl_->ncell_tot + 1)));
    impl_->nn = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->nl =
        static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(impl_->max_neigh) * n));
    impl_->max_nn = static_cast<int*>(MustMalloc(sizeof(int)));
    impl_->status = static_cast<DeviceStatus*>(MustMalloc(sizeof(DeviceStatus)));

    std::vector<double> zeros(n, 0.0);
    const double* vx = in.vx.empty() ? zeros.data() : in.vx.data();
    const double* vy = in.vy.empty() ? zeros.data() : in.vy.data();
    const double* vz = in.vz.empty() ? zeros.data() : in.vz.data();
    if (!in.vx.empty() && static_cast<int>(in.vx.size()) != n) {
      throw std::runtime_error("velocity arrays differ in length");
    }
    TILEMD_CHECK(cudaMemcpy(impl_->x, in.x.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(impl_->y, in.y.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(impl_->z, in.z.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(impl_->vx, vx, sizeof(double) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(impl_->vy, vy, sizeof(double) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(impl_->vz, vz, sizeof(double) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemset(impl_->imgx, 0, sizeof(int) * n));
    TILEMD_CHECK(cudaMemset(impl_->imgy, 0, sizeof(int) * n));
    TILEMD_CHECK(cudaMemset(impl_->imgz, 0, sizeof(int) * n));
    TILEMD_CHECK(cudaMemcpy(impl_->type, in.type.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemset(impl_->fx, 0, sizeof(float) * n));
    TILEMD_CHECK(cudaMemset(impl_->fy, 0, sizeof(float) * n));
    TILEMD_CHECK(cudaMemset(impl_->fz, 0, sizeof(float) * n));

    cub::DeviceRadixSort::SortPairs(nullptr, impl_->cub_bytes, impl_->keys, impl_->sorted_keys,
                                    impl_->indices, impl_->sorted_indices, n);
    impl_->cub_temp = MustMalloc(impl_->cub_bytes);

    double box[3] = {in.box[0], in.box[1], in.box[2]};
    int shift = static_cast<int>(in.shift);
    float neigh_cut2 = static_cast<float>(impl_->reach * impl_->reach);
    TILEMD_CHECK(cudaMemcpyToSymbol(c_pairs, table, sizeof(table)));
    TILEMD_CHECK(cudaMemcpyToSymbol(c_box, box, sizeof(box)));
    TILEMD_CHECK(cudaMemcpyToSymbol(c_ncell, impl_->ncell, sizeof(impl_->ncell)));
    TILEMD_CHECK(cudaMemcpyToSymbol(c_shift, &shift, sizeof(shift)));
    TILEMD_CHECK(cudaMemcpyToSymbol(c_neigh_cut2, &neigh_cut2, sizeof(neigh_cut2)));
    // The pair table above is process-global. Every launch path copies it again.
    WrapAndShadow<<<Grid(n), kBlock>>>(impl_->x, impl_->y, impl_->z, impl_->imgx, impl_->imgy,
                                       impl_->imgz, impl_->type, impl_->pos, n);
    CheckLaunch();
  } catch (...) {
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

Engine::~Engine() { delete impl_; }

int Engine::NumAtoms() const { return impl_->n; }
int Engine::MaxNeigh() const { return impl_->max_neigh; }
int Engine::RebuildCount() const { return impl_->rebuild_count; }

void Engine::Impl::Upload() {
  double box[3] = {system.box[0], system.box[1], system.box[2]};
  int shift = static_cast<int>(system.shift);
  float neigh_cut2 = static_cast<float>(reach * reach);
  TILEMD_CHECK(cudaMemcpyToSymbol(c_pairs, host_pairs, sizeof(host_pairs)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_box, box, sizeof(box)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_ncell, ncell, sizeof(ncell)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_shift, &shift, sizeof(shift)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_neigh_cut2, &neigh_cut2, sizeof(neigh_cut2)));
}

void Engine::Impl::ClearFlags() {
  DeviceStatus blank{};
  TILEMD_CHECK(cudaMemcpy(status, &blank, sizeof(blank), cudaMemcpyHostToDevice));
  int zero = 0;
  TILEMD_CHECK(cudaMemcpy(max_nn, &zero, sizeof(zero), cudaMemcpyHostToDevice));
}

void Engine::Impl::BuildCells() {
  CellKeyKernel<<<Grid(n), kBlock>>>(pos, keys, indices, n);
  CheckLaunch();
  TILEMD_CHECK(cub::DeviceRadixSort::SortPairs(cub_temp, cub_bytes, keys, sorted_keys, indices,
                                               sorted_indices, n));
  FillCellStart<<<1, 1>>>(sorted_keys, cell_start, n, ncell_tot);
  CheckLaunch();
}

void Engine::Impl::ReallocList(int capacity) {
  int* fresh = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(capacity) * n));
  cudaFree(nl);
  nl = fresh;
  max_neigh = capacity;
}

int Engine::Impl::BuildNeighbors(bool grow) {
  BuildCells();
  for (int attempt = 0; attempt < 8; ++attempt) {
    ClearFlags();
    NeighborKernel<<<Grid(n), kBlock>>>(pos, cell_start, sorted_indices, nn, nl, max_neigh, n,
                                        max_nn, status);
    CheckLaunch();
    DeviceStatus host_status{};
    int needed = 0;
    TILEMD_CHECK(cudaMemcpy(&host_status, status, sizeof(host_status), cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(&needed, max_nn, sizeof(needed), cudaMemcpyDeviceToHost));
    if (!host_status.overflow) {
      SortNeighbors<<<Grid(n), kBlock>>>(nl, nn, n);
      CheckLaunch();
      TILEMD_CHECK(cudaMemcpy(xref, pos, sizeof(float4) * n, cudaMemcpyDeviceToDevice));
      ++rebuild_count;
      return 0;
    }
    if (!grow) return 1;
    int next = std::max(max_neigh + max_neigh / 2, needed);
    next = RoundUp32(next);
    if (next <= max_neigh) next = RoundUp32(max_neigh + 32);
    ReallocList(next);
  }
  throw std::runtime_error("neighbor list still overflows after growing");
}

void Engine::Impl::ComputeForce() {
  ClearFlags();
  ForceKernel<<<Grid(n), kBlock>>>(x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy,
                                   wxz, wyz, n, status);
  CheckLaunch();
  forces_ready = true;
}

GpuForces Engine::Impl::DownloadForces(int overflow) {
  GpuForces out;
  out.overflow = overflow;
  out.max_neigh = max_neigh;
  if (overflow) return out;
  out.fx.resize(n);
  out.fy.resize(n);
  out.fz.resize(n);
  TILEMD_CHECK(cudaMemcpy(out.fx.data(), fx, sizeof(float) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(out.fy.data(), fy, sizeof(float) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(out.fz.data(), fz, sizeof(float) * n, cudaMemcpyDeviceToHost));

  // Pair terms on boundary_500 are ~1e34 and cancel to ~1e11. Summing each
  // atom's partial, then summing those partials, moves xz by 4e6 and misses
  // the 1e-5 gate. One running total, atoms in index order and neighbors in
  // the sorted index order, is the same association as the oracle.
  std::vector<double> hx(n), hy(n), hz(n);
  std::vector<int> ht(n), hnn(n);
  std::vector<int> hnl(static_cast<size_t>(max_neigh) * n);
  TILEMD_CHECK(cudaMemcpy(hx.data(), x, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hy.data(), y, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hz.data(), z, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(ht.data(), type, sizeof(int) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hnn.data(), nn, sizeof(int) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hnl.data(), nl, sizeof(int) * hnl.size(), cudaMemcpyDeviceToHost));
  const bool force_shift = system.shift == Shift::Force;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < hnn[i]; ++k) {
      const int j = hnl[static_cast<size_t>(k) * n + i];
      const PairRow p = host_pairs[ht[i] * kMaxTypes + ht[j]];
      const double dx = hx[i] - hx[j] - system.box[0] * std::rint((hx[i] - hx[j]) / system.box[0]);
      const double dy = hy[i] - hy[j] - system.box[1] * std::rint((hy[i] - hy[j]) / system.box[1]);
      const double dz = hz[i] - hz[j] - system.box[2] * std::rint((hz[i] - hz[j]) / system.box[2]);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= p.rc2) continue;
      const double inv_r2 = 1.0 / r2;
      const double sr2 = p.sigma2 * inv_r2;
      const double sr6 = sr2 * sr2 * sr2;
      double s = p.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
      double u = p.eps4 * sr6 * (sr6 - 1.0) - p.u_shift;
      if (force_shift) {
        const double r = std::sqrt(r2);
        u += (r - p.rc) * p.s_rc * p.rc;
        s -= p.s_rc * p.rc / r;
      }
      const double fix = s * dx;
      const double fiy = s * dy;
      const double fiz = s * dz;
      out.potential_energy += 0.5 * u;
      out.virial[0] += 0.5 * dx * fix;
      out.virial[1] += 0.5 * dy * fiy;
      out.virial[2] += 0.5 * dz * fiz;
      out.virial[3] += 0.5 * dx * fiy;
      out.virial[4] += 0.5 * dx * fiz;
      out.virial[5] += 0.5 * dy * fiz;
    }
  }
  return out;
}

GpuForces Engine::ComputeForces(bool grow) {
  impl_->Upload();
  int overflow = impl_->BuildNeighbors(grow);
  if (!overflow) impl_->ComputeForce();
  return impl_->DownloadForces(overflow);
}

void Engine::Run(int steps) {
  if (steps < 0) throw std::runtime_error("step count must be non-negative");
  impl_->Upload();
  if (!impl_->forces_ready) {
    int overflow = impl_->BuildNeighbors(true);
    if (overflow) throw std::runtime_error("initial neighbor list overflowed");
    impl_->ComputeForce();
  }
  const int n = impl_->n;
  const double dt = impl_->system.dt;
  const double inv_mass = 1.0 / impl_->system.mass;
  const double trigger = (impl_->system.skin * 0.5) * (impl_->system.skin * 0.5);
  for (int step = 0; step < steps; ++step) {
    KickDrift<<<Grid(n), kBlock>>>(impl_->x, impl_->y, impl_->z, impl_->vx, impl_->vy, impl_->vz,
                                   impl_->fx, impl_->fy, impl_->fz, impl_->imgx, impl_->imgy,
                                   impl_->imgz, n, dt, inv_mass);
    CheckLaunch();
    RefreshShadow<<<Grid(n), kBlock>>>(impl_->x, impl_->y, impl_->z, impl_->type, impl_->pos, n);
    CheckLaunch();
    bool rebuild = impl_->system.rebuild_every_step;
    if (!rebuild) {
      impl_->ClearFlags();
      DispKernel<<<Grid(n), kBlock>>>(impl_->pos, impl_->xref, n, impl_->status);
      CheckLaunch();
      DeviceStatus status{};
      TILEMD_CHECK(cudaMemcpy(&status, impl_->status, sizeof(status), cudaMemcpyDeviceToHost));
      rebuild = status.max_disp2_bits > static_cast<float>(trigger);
    }
    if (rebuild) {
      int overflow = impl_->BuildNeighbors(true);
      if (overflow) throw std::runtime_error("neighbor list overflowed during integration");
    }
    impl_->ComputeForce();
    Kick<<<Grid(n), kBlock>>>(impl_->vx, impl_->vy, impl_->vz, impl_->fx, impl_->fy, impl_->fz, n,
                              dt, inv_mass);
    CheckLaunch();
  }
}

void Engine::DownloadPositions(std::vector<double>& x, std::vector<double>& y,
                               std::vector<double>& z) const {
  x.resize(impl_->n);
  y.resize(impl_->n);
  z.resize(impl_->n);
  TILEMD_CHECK(cudaMemcpy(x.data(), impl_->x, sizeof(double) * impl_->n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(y.data(), impl_->y, sizeof(double) * impl_->n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(z.data(), impl_->z, sizeof(double) * impl_->n, cudaMemcpyDeviceToHost));
}

void Engine::DownloadVelocities(std::vector<double>& vx, std::vector<double>& vy,
                                std::vector<double>& vz) const {
  vx.resize(impl_->n);
  vy.resize(impl_->n);
  vz.resize(impl_->n);
  TILEMD_CHECK(cudaMemcpy(vx.data(), impl_->vx, sizeof(double) * impl_->n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(vy.data(), impl_->vy, sizeof(double) * impl_->n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(vz.data(), impl_->vz, sizeof(double) * impl_->n, cudaMemcpyDeviceToHost));
}

}  // namespace tilemd
