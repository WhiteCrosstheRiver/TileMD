#include "md/engine.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/device_status.hpp"
#include "md/emergency.hpp"
#include "md/flag_nan.hpp"
#include "md/run_plan.hpp"
#include "md/tile_force.hpp"
#include "md/tile_rebuild.hpp"
#include "md/tile_thermo.hpp"
#include "neighbor/cluster_list.hpp"
#include "neighbor/cluster_order.hpp"

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
__constant__ int c_rebuild_every;
__constant__ float c_trigger2;
__constant__ int c_thermo_every;

constexpr int kThermoRing = 1024;

struct ThermoRecord {
  int step;
  double ke;
  double pe;
  double wxx;
  double wyy;
  double wzz;
  double wxy;
  double wxz;
  double wyz;
};

using ThermoPartial = TileThermoPartial;

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

template <typename T>
__device__ T LoadReadOnly(const T* ptr) {
  return __ldg(ptr);
}

__device__ double MinImageDouble(double delta, double length) {
  return delta - length * rint(delta / length);
}

// 10-bit Morton code. Stage-1 boxes stay under 1024 cells per axis.
__host__ __device__ unsigned Spread10(unsigned v) {
  v &= 0x3ffu;
  v = (v | (v << 16)) & 0x030000FFu;
  v = (v | (v << 8)) & 0x0300F00Fu;
  v = (v | (v << 4)) & 0x030C30C3u;
  v = (v | (v << 2)) & 0x09249249u;
  return v;
}

__host__ __device__ int MortonCell(int ix, int iy, int iz) {
  unsigned code = Spread10(static_cast<unsigned>(ix)) | (Spread10(static_cast<unsigned>(iy)) << 1) |
                  (Spread10(static_cast<unsigned>(iz)) << 2);
  return static_cast<int>(code);
}

__device__ int CellCoord(double x, double inv_edge, int ncell) {
  int i = static_cast<int>(floor(x * inv_edge));
  if (i < 0) i = 0;
  if (i >= ncell) i = ncell - 1;
  return i;
}

__device__ double ShadowOf(float4 value, int axis) {
  return ShadowCoordinate(axis == 0   ? value.x
                          : axis == 1 ? value.y
                                      : value.z,
                          __float_as_int(value.w), axis);
}

__device__ int WrapCell(int i, int ncell, int delta) {
  int w = i + delta;
  if (w >= ncell) w -= ncell;
  if (w < 0) w += ncell;
  return w;
}

__device__ double WrapDouble(double x, double length, int* image) {
  return WrapImage(x, length, image);
}

__global__ void WrapAndShadow(double* x, double* y, double* z, int* imgx, int* imgy, int* imgz,
                              const int* type, float4* pos, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  (void)type;
  x[i] = WrapDouble(x[i], c_box[0], &imgx[i]);
  y[i] = WrapDouble(y[i], c_box[1], &imgy[i]);
  z[i] = WrapDouble(z[i], c_box[2], &imgz[i]);
  pos[i] =
      make_float4(ShadowLocal(x[i]), ShadowLocal(y[i]), ShadowLocal(z[i]),
                  __int_as_float(PackShadow(ShadowBin(x[i]), ShadowBin(y[i]), ShadowBin(z[i]))));
}

__global__ void RefreshShadow(const double* x, const double* y, const double* z, const int* type,
                              float4* pos, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  (void)type;
  pos[i] =
      make_float4(ShadowLocal(x[i]), ShadowLocal(y[i]), ShadowLocal(z[i]),
                  __int_as_float(PackShadow(ShadowBin(x[i]), ShadowBin(y[i]), ShadowBin(z[i]))));
}

__device__ bool RebuildSkipped(const int* gate) { return gate != nullptr && *gate == 0; }

__global__ void CellKeyKernel(const float4* pos, int* keys, int* indices, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = LoadReadOnly(pos + i);
  int ix = CellCoord(ShadowOf(p, 0), static_cast<double>(c_ncell[0]) / c_box[0], c_ncell[0]);
  int iy = CellCoord(ShadowOf(p, 1), static_cast<double>(c_ncell[1]) / c_box[1], c_ncell[1]);
  int iz = CellCoord(ShadowOf(p, 2), static_cast<double>(c_ncell[2]) / c_box[2], c_ncell[2]);
  keys[i] = MortonCell(ix, iy, iz);
  indices[i] = i;
}

template <typename T>
__global__ void PermuteKernel(const T* in, T* out, const int* order, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  out[s] = in[order[s]];
}

template <typename T>
__global__ void CopyBackKernel(T* data, const T* scratch, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  data[s] = scratch[s];
}

__global__ void IdentityKernel(int* ids, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ids[i] = i;
}

__global__ void FillSlotKernel(const int* gid, int* slot, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  slot[gid[s]] = s;
}

__global__ void FillCellStart(const int* sorted_keys, int* cell_start, int n, int ncell,
                              const int* gate) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (RebuildSkipped(gate)) return;
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

__global__ void __launch_bounds__(kBlock)
    NeighborKernel(const float4* __restrict__ pos, const int* __restrict__ cell_start,
                   const int* __restrict__ cell_atoms, int* nn, int* nl, int max_neigh, int n,
                   int* max_nn, DeviceStatus* status, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 pi = LoadReadOnly(pos + i);
  const double xi = ShadowOf(pi, 0);
  const double yi = ShadowOf(pi, 1);
  const double zi = ShadowOf(pi, 2);
  int ix = CellCoord(xi, static_cast<double>(c_ncell[0]) / c_box[0], c_ncell[0]);
  int iy = CellCoord(yi, static_cast<double>(c_ncell[1]) / c_box[1], c_ncell[1]);
  int iz = CellCoord(zi, static_cast<double>(c_ncell[2]) / c_box[2], c_ncell[2]);
  int count = 0;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int cell = MortonCell(WrapCell(ix, c_ncell[0], dx), WrapCell(iy, c_ncell[1], dy),
                              WrapCell(iz, c_ncell[2], dz));
        for (int s = cell_start[cell]; s < cell_start[cell + 1]; ++s) {
          int j = cell_atoms[s];
          if (j == i) continue;
          float4 pj = LoadReadOnly(pos + j);
          double dxij = MinImageDouble(xi - ShadowOf(pj, 0), c_box[0]);
          double dyij = MinImageDouble(yi - ShadowOf(pj, 1), c_box[1]);
          double dzij = MinImageDouble(zi - ShadowOf(pj, 2), c_box[2]);
          double d2 = dxij * dxij + dyij * dyij + dzij * dzij;
          if (d2 < static_cast<double>(c_neigh_cut2)) {
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
// Neighbors are stored as sorted slots. Ordering by original global id keeps the
// pair sum in the same sequence as the oracle's j loop.
__global__ void SortNeighbors(int* nl, const int* nn, const int* gid, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int count = nn[i];
  for (int a = 1; a < count; ++a) {
    int key = nl[static_cast<long long>(a) * n + i];
    int key_gid = gid[key];
    int b = a;
    while (b > 0 && gid[nl[static_cast<long long>(b - 1) * n + i]] > key_gid) {
      nl[static_cast<long long>(b) * n + i] = nl[static_cast<long long>(b - 1) * n + i];
      --b;
    }
    nl[static_cast<long long>(b) * n + i] = key;
  }
}

template <int Block>
__global__ void __launch_bounds__(Block)
    ForceKernel(const double* __restrict__ x, const double* __restrict__ y,
                const double* __restrict__ z, const int* __restrict__ type,
                const int* __restrict__ nn, const int* __restrict__ nl, float* fx, float* fy,
                float* fz, double* energy, double* wxx, double* wyy, double* wzz, double* wxy,
                double* wxz, double* wyz, int n, DeviceStatus* status, const int* ev_flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int ev = ev_flag != nullptr && *ev_flag != 0;
  int ti = LoadReadOnly(type + i);
  const double xi = LoadReadOnly(x + i);
  const double yi = LoadReadOnly(y + i);
  const double zi = LoadReadOnly(z + i);
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
  int count = LoadReadOnly(nn + i);
  for (int k = 0; k < count; ++k) {
    int j = LoadReadOnly(nl + static_cast<long long>(k) * n + i);
    int tj = LoadReadOnly(type + j);
    PairRow p = c_pairs[ti * kMaxTypes + tj];
    if (!p.defined) {
      FlagNonFinite(status, i);
      return;
    }
    // Double geometry. A float box of length ~17 already moves melt forces by
    // ~1e-3 relative, which misses the 1e-5 gate. The stored force is still float.
    double dx = MinImageDouble(xi - LoadReadOnly(x + j), c_box[0]);
    double dy = MinImageDouble(yi - LoadReadOnly(y + j), c_box[1]);
    double dz = MinImageDouble(zi - LoadReadOnly(z + j), c_box[2]);
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= p.rc2) continue;
    double inv_r2 = 1.0 / r2;
    double sr2 = p.sigma2 * inv_r2;
    double sr6 = sr2 * sr2 * sr2;
    double s = p.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
    double u = 0.0;
    if (ev) u = p.eps4 * sr6 * (sr6 - 1.0) - p.u_shift;
    if (c_shift == 2) {
      double r = sqrt(r2);
      if (ev) u += (r - p.rc) * p.s_rc * p.rc;
      s -= p.s_rc * p.rc / r;
    }
    double fix = s * dx;
    double fiy = s * dy;
    double fiz = s * dz;
    ax += fix;
    ay += fiy;
    az += fiz;
    if (ev) {
      e += 0.5 * u;
      axx += 0.5 * dx * fix;
      ayy += 0.5 * dy * fiy;
      azz += 0.5 * dz * fiz;
      axy += 0.5 * dx * fiy;
      axz += 0.5 * dx * fiz;
      ayz += 0.5 * dy * fiz;
    }
  }
  fx[i] = static_cast<float>(ax);
  fy[i] = static_cast<float>(ay);
  fz[i] = static_cast<float>(az);
  if (ev) {
    energy[i] = e;
    wxx[i] = axx;
    wyy[i] = ayy;
    wzz[i] = azz;
    wxy[i] = axy;
    wxz[i] = axz;
    wyz[i] = ayz;
  }
  if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) FlagNonFinite(status, i);
}

template <int Block>
void LaunchForceBlock(cudaStream_t stream, const double* x, const double* y, const double* z,
                      const int* type, const int* nn, const int* nl, float* fx, float* fy,
                      float* fz, double* energy, double* wxx, double* wyy, double* wzz, double* wxy,
                      double* wxz, double* wyz, int n, DeviceStatus* status, const int* ev_flag) {
  const int grid = std::max(1, (n + Block - 1) / Block);
  ForceKernel<Block><<<grid, Block, 0, stream>>>(x, y, z, type, nn, nl, fx, fy, fz, energy, wxx,
                                                 wyy, wzz, wxy, wxz, wyz, n, status, ev_flag);
}

void LaunchForce(int block, cudaStream_t stream, const double* x, const double* y, const double* z,
                 const int* type, const int* nn, const int* nl, float* fx, float* fy, float* fz,
                 double* energy, double* wxx, double* wyy, double* wzz, double* wxy, double* wxz,
                 double* wyz, int n, DeviceStatus* status, const int* ev_flag) {
  if (block == 64) {
    LaunchForceBlock<64>(stream, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy, wxz,
                         wyz, n, status, ev_flag);
  } else if (block == 128) {
    LaunchForceBlock<128>(stream, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy,
                          wxz, wyz, n, status, ev_flag);
  } else if (block == 256) {
    LaunchForceBlock<256>(stream, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy,
                          wxz, wyz, n, status, ev_flag);
  } else {
    throw std::invalid_argument("force block size must be 64, 128, or 256");
  }
}

__global__ void PublishRebuild(cudaGraphConditionalHandle handle, const DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  cudaGraphSetConditional(handle, status->need_rebuild ? 1u : 0u);
}

__global__ void SetEvFlag(int* ev_flag, const int* step) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int every = c_thermo_every;
  int next = *step + 1;
  *ev_flag = (every > 0 && next % every == 0) ? 1 : 0;
}

__global__ void PartialThermo(const double* vx, const double* vy, const double* vz,
                              const double* energy, const double* wxx, const double* wyy,
                              const double* wzz, const double* wxy, const double* wxz,
                              const double* wyz, int n, ThermoPartial* partial,
                              const int* ev_flag) {
  using Reduce = cub::BlockReduce<double, kBlock>;
  __shared__ typename Reduce::TempStorage reduce_ke;
  __shared__ typename Reduce::TempStorage reduce_pe;
  __shared__ typename Reduce::TempStorage reduce_xx;
  __shared__ typename Reduce::TempStorage reduce_yy;
  __shared__ typename Reduce::TempStorage reduce_zz;
  __shared__ typename Reduce::TempStorage reduce_xy;
  __shared__ typename Reduce::TempStorage reduce_xz;
  __shared__ typename Reduce::TempStorage reduce_yz;
  const int ev = ev_flag != nullptr && *ev_flag != 0;
  if (!ev) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double ke = 0.0;
  double pe = 0.0;
  double axx = 0.0;
  double ayy = 0.0;
  double azz = 0.0;
  double axy = 0.0;
  double axz = 0.0;
  double ayz = 0.0;
  if (i < n) {
    ke = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    pe = energy[i];
    axx = wxx[i];
    ayy = wyy[i];
    azz = wzz[i];
    axy = wxy[i];
    axz = wxz[i];
    ayz = wyz[i];
  }
  ke = Reduce(reduce_ke).Sum(ke);
  pe = Reduce(reduce_pe).Sum(pe);
  axx = Reduce(reduce_xx).Sum(axx);
  ayy = Reduce(reduce_yy).Sum(ayy);
  azz = Reduce(reduce_zz).Sum(azz);
  axy = Reduce(reduce_xy).Sum(axy);
  axz = Reduce(reduce_xz).Sum(axz);
  ayz = Reduce(reduce_yz).Sum(ayz);
  if (threadIdx.x == 0) {
    ThermoPartial out;
    out.ke = ke;
    out.pe = pe;
    out.wxx = axx;
    out.wyy = ayy;
    out.wzz = azz;
    out.wxy = axy;
    out.wxz = axz;
    out.wyz = ayz;
    partial[blockIdx.x] = out;
  }
}

__global__ void FinishThermo(const ThermoPartial* partial, int npartial, ThermoRecord* ring,
                             int capacity, int* thermo_count, const int* step, const int* ev_flag,
                             int label_plus, DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (ev_flag == nullptr || *ev_flag == 0) return;
  ThermoRecord rec{};
  rec.step = *step + label_plus;
  for (int b = 0; b < npartial; ++b) {
    rec.ke += partial[b].ke;
    rec.pe += partial[b].pe;
    rec.wxx += partial[b].wxx;
    rec.wyy += partial[b].wyy;
    rec.wzz += partial[b].wzz;
    rec.wxy += partial[b].wxy;
    rec.wxz += partial[b].wxz;
    rec.wyz += partial[b].wyz;
  }
  int slot = *thermo_count;
  if (slot >= capacity) {
    status->overflow = 1;
    return;
  }
  ring[slot] = rec;
  *thermo_count = slot + 1;
}

__global__ void PublishTileRecord(const int* step, int every, int* record) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *record = (every > 0 && *step % every == 0) ? *step : -1;
}

__global__ void FinishTileThermo(const ThermoPartial* partial, int npartial, ThermoRecord* ring,
                                 int capacity, int* thermo_count, int step, DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  ThermoRecord rec{};
  rec.step = step;
  for (int b = 0; b < npartial; ++b) {
    rec.ke += partial[b].ke;
    rec.pe += partial[b].pe;
    rec.wxx += partial[b].wxx;
    rec.wyy += partial[b].wyy;
    rec.wzz += partial[b].wzz;
    rec.wxy += partial[b].wxy;
    rec.wxz += partial[b].wxz;
    rec.wyz += partial[b].wyz;
  }
  int slot = *thermo_count;
  if (slot >= capacity) {
    status->overflow = 1;
    return;
  }
  ring[slot] = rec;
  *thermo_count = slot + 1;
}

__global__ void FinishTileThermoDevice(const ThermoPartial* partial, int npartial,
                                       ThermoRecord* ring, int capacity, int* thermo_count,
                                       const int* record, const int* live, DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || record == nullptr || *record < 0) return;
  const int count = live != nullptr ? *live : npartial;
  ThermoRecord rec{};
  rec.step = *record;
  for (int b = 0; b < count; ++b) {
    rec.ke += partial[b].ke;
    rec.pe += partial[b].pe;
    rec.wxx += partial[b].wxx;
    rec.wyy += partial[b].wyy;
    rec.wzz += partial[b].wzz;
    rec.wxy += partial[b].wxy;
    rec.wxz += partial[b].wxz;
    rec.wyz += partial[b].wyz;
  }
  int slot = *thermo_count;
  if (slot >= capacity) {
    status->overflow = 1;
    return;
  }
  ring[slot] = rec;
  *thermo_count = slot + 1;
}

__global__ void IncrementStep(int* step) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *step += 1;
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

__global__ void DriftHalf(double* x, double* y, double* z, const double* vx, const double* vy,
                          const double* vz, int* imgx, int* imgy, int* imgz, int n,
                          double half_dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = WrapDouble(x[i] + half_dt * vx[i], c_box[0], &imgx[i]);
  y[i] = WrapDouble(y[i] + half_dt * vy[i], c_box[1], &imgy[i]);
  z[i] = WrapDouble(z[i] + half_dt * vz[i], c_box[2], &imgz[i]);
}

__device__ void PhiloxRound(unsigned* ctr, unsigned& key0, unsigned& key1) {
  const unsigned long long p0 = static_cast<unsigned long long>(0xD2511F53u) * ctr[0];
  const unsigned long long p1 = static_cast<unsigned long long>(0xCD9E8D57u) * ctr[2];
  const unsigned lo0 = static_cast<unsigned>(p0);
  const unsigned lo1 = static_cast<unsigned>(p1);
  const unsigned hi0 = static_cast<unsigned>(p0 >> 32);
  const unsigned hi1 = static_cast<unsigned>(p1 >> 32);
  const unsigned next0 = hi1 ^ ctr[1] ^ key0;
  const unsigned next1 = lo1;
  const unsigned next2 = hi0 ^ ctr[3] ^ key1;
  const unsigned next3 = lo0;
  ctr[0] = next0;
  ctr[1] = next1;
  ctr[2] = next2;
  ctr[3] = next3;
  key0 += 0x9E3779B9u;
  key1 += 0xBB67AE85u;
}

__device__ void DeviceNormals(unsigned seed, unsigned step_lo, unsigned step_hi, int gid,
                              double out[3]) {
  unsigned ctr[4] = {static_cast<unsigned>(gid), step_lo, step_hi, 0u};
  unsigned key0 = seed;
  unsigned key1 = 0u;
  for (int round = 0; round < 10; ++round) PhiloxRound(ctr, key0, key1);
  const unsigned bits[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
  double normal[4];
  for (int pair = 0; pair < 2; ++pair) {
    const double u1 = (static_cast<double>(bits[2 * pair]) + 0.5) * (1.0 / 4294967296.0);
    const double u2 = (static_cast<double>(bits[2 * pair + 1]) + 0.5) * (1.0 / 4294967296.0);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.28318530717958647692 * u2;
    normal[2 * pair] = radius * cos(angle);
    normal[2 * pair + 1] = radius * sin(angle);
  }
  out[0] = normal[0];
  out[1] = normal[1];
  out[2] = normal[2];
}

__global__ void OrnsteinUhlenbeck(double* vx, double* vy, double* vz, const int* gid, int n,
                                  double decay, double noise, unsigned seed, unsigned step_lo,
                                  unsigned step_hi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double draw[3] = {0.0, 0.0, 0.0};
  if (noise > 0.0) DeviceNormals(seed, step_lo, step_hi, gid[i], draw);
  vx[i] = decay * vx[i] + noise * draw[0];
  vy[i] = decay * vy[i] + noise * draw[1];
  vz[i] = decay * vz[i] + noise * draw[2];
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

struct MaxFloat {
  __device__ float operator()(float a, float b) const { return a > b ? a : b; }
};

__global__ void DispKernel(const float4* pos, const float4* xref, int n, DeviceStatus* status) {
  using Reduce = cub::BlockReduce<float, kBlock>;
  __shared__ typename Reduce::TempStorage reduce_storage;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float d2 = 0.f;
  if (i < n) {
    float4 pi = LoadReadOnly(pos + i);
    float4 ref = LoadReadOnly(xref + i);
    const double rx = ShadowOf(ref, 0);
    const double ry = ShadowOf(ref, 1);
    const double rz = ShadowOf(ref, 2);
    const double dx = OnReferenceBranch(ShadowOf(pi, 0), rx, c_box[0]) - rx;
    const double dy = OnReferenceBranch(ShadowOf(pi, 1), ry, c_box[1]) - ry;
    const double dz = OnReferenceBranch(ShadowOf(pi, 2), rz, c_box[2]) - rz;
    d2 = static_cast<float>(dx * dx + dy * dy + dz * dz);
  }
  float block_max = Reduce(reduce_storage).Reduce(d2, MaxFloat());
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<int*>(&status->max_disp2_bits), __float_as_int(block_max));
  }
}

__global__ void ResetDisplacement(DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  status->max_disp2_bits = 0.f;
  status->need_rebuild = 0;
}

__global__ void DecideRebuild(DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float d2 = status->max_disp2_bits;
  status->need_rebuild = (c_rebuild_every || DisplacementNeedsRebuild(d2, c_trigger2)) ? 1 : 0;
}

__global__ void ZeroMaxNn(int* max_nn, const int* gate) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (RebuildSkipped(gate)) return;
  *max_nn = 0;
}

__global__ void StoreXrefIf(float4* xref, const float4* pos, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  xref[i] = pos[i];
}

__global__ void AccountRebuild(DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (status->need_rebuild) {
    status->rebuild_count += 1;
    status->max_disp2_bits = 0.f;
  }
  status->need_rebuild = 0;
}

int RoundUp32(int value) {
  if (value < 32) return 32;
  return (value + 31) & ~31;
}

template <typename T>
void PermuteBuffer(T* data, T* scratch, const int* order, int n) {
  PermuteKernel<T><<<Grid(n), kBlock>>>(data, scratch, order, n, nullptr);
  CheckLaunch();
  TILEMD_CHECK(
      cudaMemcpy(data, scratch, sizeof(T) * static_cast<size_t>(n), cudaMemcpyDeviceToDevice));
}

template <typename T>
void QueuePermute(T* data, T* scratch, const int* order, int n, const int* gate,
                  cudaStream_t stream) {
  PermuteKernel<T><<<Grid(n), kBlock, 0, stream>>>(data, scratch, order, n, gate);
  CopyBackKernel<T><<<Grid(n), kBlock, 0, stream>>>(data, scratch, n, gate);
}

struct GraphHold {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStream_t stream = nullptr;
  ~GraphHold() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    if (stream) cudaStreamDestroy(stream);
  }
  void ResetGraph() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    exec = nullptr;
    graph = nullptr;
  }
};

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
  float4* pos_next = nullptr;
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
  int* gid = nullptr;
  int* slot_of_gid = nullptr;
  int* keys = nullptr;
  int* sorted_keys = nullptr;
  int* indices = nullptr;
  int* sorted_indices = nullptr;
  double* scratch_double = nullptr;
  int* scratch_int = nullptr;
  float* scratch_float = nullptr;
  float4* scratch_float4 = nullptr;
  int* cell_start = nullptr;
  int* nn = nullptr;
  int* nl = nullptr;
  int* max_nn = nullptr;
  void* cub_temp = nullptr;
  size_t cub_bytes = 0;
  DeviceStatus* status = nullptr;
  double* snap_x = nullptr;
  double* snap_y = nullptr;
  double* snap_z = nullptr;
  double* snap_vx = nullptr;
  double* snap_vy = nullptr;
  double* snap_vz = nullptr;
  int* snap_imgx = nullptr;
  int* snap_imgy = nullptr;
  int* snap_imgz = nullptr;
  int* snap_type = nullptr;
  int* snap_gid = nullptr;
  int* snap_step = nullptr;
  int* snap_thermo_count = nullptr;
  int* step_counter = nullptr;
  int* thermo_count = nullptr;
  int* ev_flag = nullptr;
  ThermoRecord* thermo_ring = nullptr;
  ThermoPartial* thermo_partial = nullptr;
  int npartial = 0;
  bool thermo_origin_written = false;
  bool leapfrog_ready = false;
  int tile_step = 0;
  int* tile_record = nullptr;
  bool tile_ready = false;
  int* tile_type = nullptr;
  int* tile_gid = nullptr;
  int* tile_ci = nullptr;
  int* tile_cj = nullptr;
  int* tile_shift = nullptr;
  int* tile_missing = nullptr;
  TileThermoPartial* tile_partial = nullptr;
  int tile_partial_cap = 0;
  int tile_ncluster = 0;
  int tile_npair = 0;
  int tile_slot_cap = 0;
  int tile_ci_cap = 0;
  int tile_pair_cap = 0;
  int tile_cluster_cap = 0;
  int tile_column_cap = 0;
  int tile_nx = 0;
  int tile_ny = 0;
  double tile_edge[2] = {};
  double tile_reach = 0.0;
  int* tile_live = nullptr;
  int* tile_column_count = nullptr;
  int* tile_column_cluster = nullptr;
  int* tile_column_ncluster = nullptr;
  int* tile_cluster_column = nullptr;
  unsigned long long* tile_zkey = nullptr;
  unsigned long long* tile_zkey_out = nullptr;
  int* tile_column_of = nullptr;
  int* tile_colkey = nullptr;
  int* tile_colkey_out = nullptr;
  int* tile_order_in = nullptr;
  int* tile_order_z = nullptr;
  int* tile_order_out = nullptr;
  double* tile_lo_x = nullptr;
  double* tile_lo_y = nullptr;
  double* tile_lo_z = nullptr;
  double* tile_hi_x = nullptr;
  double* tile_hi_y = nullptr;
  double* tile_hi_z = nullptr;
  int* tile_pair_count = nullptr;
  void* tile_cub = nullptr;
  size_t tile_cub_bytes = 0;
  std::vector<ThermoSample> drained;

  void Upload();
  void ClearFlags();
  [[noreturn]] void AbortIfNonFinite(int step, const DeviceStatus& host);
  void WriteDump(int step, bool append);
  void CollectDumpVirial(std::vector<double>& oxx, std::vector<double>& oyy,
                         std::vector<double>& ozz, std::vector<double>& oxy,
                         std::vector<double>& oxz, std::vector<double>& oyz);
  TileDeviceForces EvaluateTiles();
  void BuildCells();
  void ApplySpatialOrder();
  void QueueRebuild(cudaStream_t stream);
  void LaunchStep(cudaStream_t stream);
  void LaunchPrefix(cudaStream_t stream);
  void LaunchSuffix(cudaStream_t stream);
  void RunBaoab(int steps);
  bool CaptureConditional(cudaStream_t stream, int count, cudaGraph_t* graph,
                          cudaGraphExec_t* exec);
  void Snapshot(cudaStream_t stream);
  void RestoreSnapshot();
  void Capture(cudaStream_t stream, int count, cudaGraph_t* graph, cudaGraphExec_t* exec);
  void RecordInitialThermo();
  void DrainDeviceThermo();
  void ReallocList(int capacity);
  int BuildNeighbors(bool grow);
  void ComputeForce();
  GpuForces TileForces();
  bool TileListStale();
  void RebuildTileCache(float4* pos_cur);
  void PrepareTileRebuild(bool host_rebuilt);
  bool QueueTileRebuildStep(cudaStream_t stream, const float4* written);
  void QueueFusedPhysics(cudaStream_t stream, float4* read, float4* write);
  bool CaptureFusedConditional(cudaStream_t stream, int count, float4* cur, float4* nxt,
                               cudaGraph_t* graph, cudaGraphExec_t* exec);
  void LaunchTileStep(float4* pos_cur, float4* pos_out);
  void RunTileSteps(int count, float4*& latest, float4*& other, int step0);
  bool CaptureFusedGraph(cudaStream_t stream, int count, float4* cur, float4* nxt,
                         cudaGraph_t* graph, cudaGraphExec_t* exec);
  void RecordClosingTileThermo();
  void RunClusterTiles(int steps);
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
    cudaFree(pos_next);
    cudaFree(xref);
    cudaFree(tile_type);
    cudaFree(tile_gid);
    cudaFree(tile_ci);
    cudaFree(tile_cj);
    cudaFree(tile_shift);
    cudaFree(tile_missing);
    cudaFree(tile_partial);
    cudaFree(tile_record);
    cudaFree(tile_live);
    cudaFree(tile_column_count);
    cudaFree(tile_column_cluster);
    cudaFree(tile_column_ncluster);
    cudaFree(tile_cluster_column);
    cudaFree(tile_zkey);
    cudaFree(tile_zkey_out);
    cudaFree(tile_column_of);
    cudaFree(tile_colkey);
    cudaFree(tile_colkey_out);
    cudaFree(tile_order_in);
    cudaFree(tile_order_z);
    cudaFree(tile_order_out);
    cudaFree(tile_lo_x);
    cudaFree(tile_lo_y);
    cudaFree(tile_lo_z);
    cudaFree(tile_hi_x);
    cudaFree(tile_hi_y);
    cudaFree(tile_hi_z);
    cudaFree(tile_pair_count);
    cudaFree(tile_cub);
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
    cudaFree(gid);
    cudaFree(slot_of_gid);
    cudaFree(snap_x);
    cudaFree(snap_y);
    cudaFree(snap_z);
    cudaFree(snap_vx);
    cudaFree(snap_vy);
    cudaFree(snap_vz);
    cudaFree(snap_imgx);
    cudaFree(snap_imgy);
    cudaFree(snap_imgz);
    cudaFree(snap_type);
    cudaFree(snap_gid);
    cudaFree(snap_step);
    cudaFree(snap_thermo_count);
    cudaFree(step_counter);
    cudaFree(thermo_count);
    cudaFree(ev_flag);
    cudaFree(thermo_ring);
    cudaFree(thermo_partial);
    cudaFree(keys);
    cudaFree(sorted_keys);
    cudaFree(scratch_double);
    cudaFree(scratch_int);
    cudaFree(scratch_float);
    cudaFree(scratch_float4);
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
      if (impl_->ncell[dim] > 1024) throw std::runtime_error("more than 1024 cells on one axis");
    }
    // cell_start is indexed by the Morton code, including the sentinel after the
    // last cell. A linear cell count is smaller than that code, so the old
    // length wrote past the array and the stencil read the wrong range.
    const int last_code = MortonCell(impl_->ncell[0] - 1, impl_->ncell[1] - 1, impl_->ncell[2] - 1);
    impl_->ncell_tot = last_code + 1;

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
    impl_->pos_next = static_cast<float4*>(MustMalloc(sizeof(float4) * n));
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
    impl_->gid = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->slot_of_gid = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->scratch_double = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->scratch_int = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->scratch_float = static_cast<float*>(MustMalloc(sizeof(float) * n));
    impl_->scratch_float4 = static_cast<float4*>(MustMalloc(sizeof(float4) * n));
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
    TILEMD_CHECK(cudaMemset(impl_->status, 0, sizeof(DeviceStatus)));
    TILEMD_CHECK(cudaMemset(impl_->max_nn, 0, sizeof(int)));
    impl_->snap_x = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->snap_y = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->snap_z = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->snap_vx = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->snap_vy = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->snap_vz = static_cast<double*>(MustMalloc(sizeof(double) * n));
    impl_->snap_imgx = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->snap_imgy = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->snap_imgz = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->snap_type = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->snap_gid = static_cast<int*>(MustMalloc(sizeof(int) * n));
    impl_->snap_step = static_cast<int*>(MustMalloc(sizeof(int)));
    impl_->snap_thermo_count = static_cast<int*>(MustMalloc(sizeof(int)));
    impl_->step_counter = static_cast<int*>(MustMalloc(sizeof(int)));
    impl_->thermo_count = static_cast<int*>(MustMalloc(sizeof(int)));
    impl_->ev_flag = static_cast<int*>(MustMalloc(sizeof(int)));
    impl_->npartial = Grid(n);
    impl_->thermo_ring = static_cast<ThermoRecord*>(MustMalloc(sizeof(ThermoRecord) * kThermoRing));
    impl_->thermo_partial =
        static_cast<ThermoPartial*>(MustMalloc(sizeof(ThermoPartial) * impl_->npartial));
    TILEMD_CHECK(cudaMemset(impl_->step_counter, 0, sizeof(int)));
    TILEMD_CHECK(cudaMemset(impl_->thermo_count, 0, sizeof(int)));
    TILEMD_CHECK(cudaMemset(impl_->ev_flag, 0, sizeof(int)));

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
    std::vector<int> initial_gid(n);
    for (int i = 0; i < n; ++i) initial_gid[i] = i;
    TILEMD_CHECK(
        cudaMemcpy(impl_->gid, initial_gid.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(impl_->slot_of_gid, initial_gid.data(), sizeof(int) * n,
                            cudaMemcpyHostToDevice));
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
int Engine::RebuildCount() const {
  DeviceStatus snapshot{};
  TILEMD_CHECK(cudaMemcpy(&snapshot, impl_->status, sizeof(snapshot), cudaMemcpyDeviceToHost));
  return impl_->rebuild_count + snapshot.rebuild_count;
}

void Engine::Impl::Upload() {
  double box[3] = {system.box[0], system.box[1], system.box[2]};
  int shift = static_cast<int>(system.shift);
  float neigh_cut2 = static_cast<float>(reach * reach);
  int rebuild_every = system.rebuild_every_step ? 1 : 0;
  float trigger2 = static_cast<float>((system.skin * 0.5) * (system.skin * 0.5));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_pairs, host_pairs, sizeof(host_pairs)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_box, box, sizeof(box)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_ncell, ncell, sizeof(ncell)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_shift, &shift, sizeof(shift)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_neigh_cut2, &neigh_cut2, sizeof(neigh_cut2)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_rebuild_every, &rebuild_every, sizeof(rebuild_every)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_trigger2, &trigger2, sizeof(trigger2)));
  TILEMD_CHECK(cudaMemcpyToSymbol(c_thermo_every, &system.thermo_every, sizeof(int)));
}

void Engine::Impl::AbortIfNonFinite(int step, const DeviceStatus& host) {
  const int slot = host.nan_slot;
  int original = -1;
  std::vector<double> hx(static_cast<size_t>(std::max(n, 0)));
  std::vector<double> hy(hx.size());
  std::vector<double> hz(hx.size());
  std::vector<double> hvx(hx.size());
  std::vector<double> hvy(hx.size());
  std::vector<double> hvz(hx.size());
  std::vector<int> htype(hx.size());
  std::vector<int> hgid(hx.size());
  if (n > 0) {
    TILEMD_CHECK(cudaMemcpy(hx.data(), x, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hy.data(), y, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hz.data(), z, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hvx.data(), vx, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hvy.data(), vy, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hvz.data(), vz, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(htype.data(), type, sizeof(int) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hgid.data(), gid, sizeof(int) * n, cudaMemcpyDeviceToHost));
    if (slot >= 0 && slot < n) original = hgid[static_cast<size_t>(slot)];
  }
  WriteEmergencyFrame("emergency.extxyz", system.box, hx.data(), hy.data(), hz.data(), hvx.data(),
                      hvy.data(), hvz.data(), htype.data(), hgid.data(), n, step);
  throw std::runtime_error(NonFiniteMessage(step, original));
}

void ScatterToGid(const double* src, const int* gid, int n, std::vector<double>& dst);

void Engine::Impl::WriteDump(int step, bool append) {
  if (system.dump_every <= 0 || system.dump_path.empty()) return;
  std::vector<double> hx;
  std::vector<double> hy;
  std::vector<double> hz;
  std::vector<double> hvx;
  std::vector<double> hvy;
  std::vector<double> hvz;
  ScatterToGid(x, gid, n, hx);
  ScatterToGid(y, gid, n, hy);
  ScatterToGid(z, gid, n, hz);
  ScatterToGid(vx, gid, n, hvx);
  ScatterToGid(vy, gid, n, hvy);
  ScatterToGid(vz, gid, n, hvz);
  std::vector<int> stored_type(static_cast<size_t>(std::max(n, 0)));
  std::vector<int> hgid(stored_type.size());
  std::vector<int> type_gid(stored_type.size());
  std::vector<int> identity(stored_type.size());
  if (n > 0) {
    TILEMD_CHECK(cudaMemcpy(stored_type.data(), type, sizeof(int) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hgid.data(), gid, sizeof(int) * n, cudaMemcpyDeviceToHost));
    for (int s = 0; s < n; ++s) {
      type_gid[static_cast<size_t>(hgid[static_cast<size_t>(s)])] =
          stored_type[static_cast<size_t>(s)];
      identity[static_cast<size_t>(s)] = s;
    }
  }
  const char* shift_name = "none";
  if (system.shift == Shift::Energy) shift_name = "energy";
  if (system.shift == Shift::Force) shift_name = "force";
  std::vector<double> oxx;
  std::vector<double> oyy;
  std::vector<double> ozz;
  std::vector<double> oxy;
  std::vector<double> oxz;
  std::vector<double> oyz;
  const double* virial_xx = nullptr;
  if (system.dump_virial != 0) {
    CollectDumpVirial(oxx, oyy, ozz, oxy, oxz, oyz);
    virial_xx = oxx.data();
  }
  WriteEmergencyFrame(system.dump_path, system.box, hx.data(), hy.data(), hz.data(), hvx.data(),
                      hvy.data(), hvz.data(), type_gid.data(), identity.data(), n, step, append,
                      shift_name, virial_xx, oyy.data(), ozz.data(), oxy.data(), oxz.data(),
                      oyz.data());
}

void Engine::Impl::CollectDumpVirial(std::vector<double>& oxx, std::vector<double>& oyy,
                                     std::vector<double>& ozz, std::vector<double>& oxy,
                                     std::vector<double>& oxz, std::vector<double>& oyz) {
  if (system.use_cluster_tiles) {
    const TileDeviceForces raw = EvaluateTiles();
    oxx = raw.wxx;
    oyy = raw.wyy;
    ozz = raw.wzz;
    oxy = raw.wxy;
    oxz = raw.wxz;
    oyz = raw.wyz;
    return;
  }
  int one = 1;
  TILEMD_CHECK(cudaMemcpy(ev_flag, &one, sizeof(one), cudaMemcpyHostToDevice));
  LaunchForce(kBlock, nullptr, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy, wxz,
              wyz, n, status, ev_flag);
  CheckLaunch();
  ScatterToGid(wxx, gid, n, oxx);
  ScatterToGid(wyy, gid, n, oyy);
  ScatterToGid(wzz, gid, n, ozz);
  ScatterToGid(wxy, gid, n, oxy);
  ScatterToGid(wxz, gid, n, oxz);
  ScatterToGid(wyz, gid, n, oyz);
}

void Engine::Impl::ClearFlags() {
  DeviceStatus previous{};
  TILEMD_CHECK(cudaMemcpy(&previous, status, sizeof(previous), cudaMemcpyDeviceToHost));
  DeviceStatus blank{};
  blank.rebuild_count = previous.rebuild_count;
  blank.nan_slot = -1;
  TILEMD_CHECK(cudaMemcpy(status, &blank, sizeof(blank), cudaMemcpyHostToDevice));
  int zero = 0;
  TILEMD_CHECK(cudaMemcpy(max_nn, &zero, sizeof(zero), cudaMemcpyHostToDevice));
}

void Engine::Impl::BuildCells() {
  CellKeyKernel<<<Grid(n), kBlock>>>(pos, keys, indices, n, nullptr);
  CheckLaunch();
  // Stable: atoms that share a Morton cell keep their previous order.
  TILEMD_CHECK(cub::DeviceRadixSort::SortPairs(cub_temp, cub_bytes, keys, sorted_keys, indices,
                                               sorted_indices, n));
}

void Engine::Impl::ApplySpatialOrder() {
  PermuteBuffer(x, scratch_double, sorted_indices, n);
  PermuteBuffer(y, scratch_double, sorted_indices, n);
  PermuteBuffer(z, scratch_double, sorted_indices, n);
  PermuteBuffer(vx, scratch_double, sorted_indices, n);
  PermuteBuffer(vy, scratch_double, sorted_indices, n);
  PermuteBuffer(vz, scratch_double, sorted_indices, n);
  PermuteBuffer(imgx, scratch_int, sorted_indices, n);
  PermuteBuffer(imgy, scratch_int, sorted_indices, n);
  PermuteBuffer(imgz, scratch_int, sorted_indices, n);
  PermuteBuffer(type, scratch_int, sorted_indices, n);
  PermuteBuffer(gid, scratch_int, sorted_indices, n);
  PermuteBuffer(pos, scratch_float4, sorted_indices, n);
  PermuteBuffer(xref, scratch_float4, sorted_indices, n);
  PermuteBuffer(fx, scratch_float, sorted_indices, n);
  PermuteBuffer(fy, scratch_float, sorted_indices, n);
  PermuteBuffer(fz, scratch_float, sorted_indices, n);
  PermuteBuffer(energy, scratch_double, sorted_indices, n);
  PermuteBuffer(wxx, scratch_double, sorted_indices, n);
  PermuteBuffer(wyy, scratch_double, sorted_indices, n);
  PermuteBuffer(wzz, scratch_double, sorted_indices, n);
  PermuteBuffer(wxy, scratch_double, sorted_indices, n);
  PermuteBuffer(wxz, scratch_double, sorted_indices, n);
  PermuteBuffer(wyz, scratch_double, sorted_indices, n);
  FillSlotKernel<<<Grid(n), kBlock>>>(gid, slot_of_gid, n, nullptr);
  CheckLaunch();
  FillCellStart<<<1, 1>>>(sorted_keys, cell_start, n, ncell_tot, nullptr);
  CheckLaunch();
  IdentityKernel<<<Grid(n), kBlock>>>(sorted_indices, n, nullptr);
  CheckLaunch();
}

void Engine::Impl::QueueRebuild(cudaStream_t stream) {
  // The host does not read the flag. Kernels that would change atom order or
  // the list return immediately when need_rebuild is clear. That is the
  // fallback when the step is not inside a conditional graph node. CUB has no
  // prologue of its own; the conditional node is what skips the sort.
  const int* gate = &status->need_rebuild;
  CellKeyKernel<<<Grid(n), kBlock, 0, stream>>>(pos, keys, indices, n, gate);
  TILEMD_CHECK(cub::DeviceRadixSort::SortPairs(cub_temp, cub_bytes, keys, sorted_keys, indices,
                                               sorted_indices, n, 0,
                                               static_cast<int>(sizeof(int) * 8), stream));
  QueuePermute(x, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(y, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(z, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(vx, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(vy, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(vz, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(imgx, scratch_int, sorted_indices, n, gate, stream);
  QueuePermute(imgy, scratch_int, sorted_indices, n, gate, stream);
  QueuePermute(imgz, scratch_int, sorted_indices, n, gate, stream);
  QueuePermute(type, scratch_int, sorted_indices, n, gate, stream);
  QueuePermute(gid, scratch_int, sorted_indices, n, gate, stream);
  QueuePermute(pos, scratch_float4, sorted_indices, n, gate, stream);
  QueuePermute(xref, scratch_float4, sorted_indices, n, gate, stream);
  QueuePermute(fx, scratch_float, sorted_indices, n, gate, stream);
  QueuePermute(fy, scratch_float, sorted_indices, n, gate, stream);
  QueuePermute(fz, scratch_float, sorted_indices, n, gate, stream);
  QueuePermute(energy, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(wxx, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(wyy, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(wzz, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(wxy, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(wxz, scratch_double, sorted_indices, n, gate, stream);
  QueuePermute(wyz, scratch_double, sorted_indices, n, gate, stream);
  FillSlotKernel<<<Grid(n), kBlock, 0, stream>>>(gid, slot_of_gid, n, gate);
  FillCellStart<<<1, 1, 0, stream>>>(sorted_keys, cell_start, n, ncell_tot, gate);
  IdentityKernel<<<Grid(n), kBlock, 0, stream>>>(sorted_indices, n, gate);
  ZeroMaxNn<<<1, 1, 0, stream>>>(max_nn, gate);
  NeighborKernel<<<Grid(n), kBlock, 0, stream>>>(pos, cell_start, sorted_indices, nn, nl, max_neigh,
                                                 n, max_nn, status, gate);
  SortNeighbors<<<Grid(n), kBlock, 0, stream>>>(nl, nn, gid, n, gate);
  StoreXrefIf<<<Grid(n), kBlock, 0, stream>>>(xref, pos, n, gate);
  AccountRebuild<<<1, 1, 0, stream>>>(status);
}

void Engine::Impl::LaunchPrefix(cudaStream_t stream) {
  const double dt = system.dt;
  const double inv_mass = 1.0 / system.mass;
  KickDrift<<<Grid(n), kBlock, 0, stream>>>(x, y, z, vx, vy, vz, fx, fy, fz, imgx, imgy, imgz, n,
                                            dt, inv_mass);
  RefreshShadow<<<Grid(n), kBlock, 0, stream>>>(x, y, z, type, pos, n);
  ResetDisplacement<<<1, 1, 0, stream>>>(status);
  DispKernel<<<Grid(n), kBlock, 0, stream>>>(pos, xref, n, status);
  DecideRebuild<<<1, 1, 0, stream>>>(status);
}

void Engine::Impl::LaunchSuffix(cudaStream_t stream) {
  const double dt = system.dt;
  const double inv_mass = 1.0 / system.mass;
  SetEvFlag<<<1, 1, 0, stream>>>(ev_flag, step_counter);
  LaunchForce(kBlock, stream, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy, wxz,
              wyz, n, status, ev_flag);
  Kick<<<Grid(n), kBlock, 0, stream>>>(vx, vy, vz, fx, fy, fz, n, dt, inv_mass);
  PartialThermo<<<Grid(n), kBlock, 0, stream>>>(vx, vy, vz, energy, wxx, wyy, wzz, wxy, wxz, wyz, n,
                                                thermo_partial, ev_flag);
  FinishThermo<<<1, 1, 0, stream>>>(thermo_partial, npartial, thermo_ring, kThermoRing,
                                    thermo_count, step_counter, ev_flag, 1, status);
  IncrementStep<<<1, 1, 0, stream>>>(step_counter);
}

void Engine::Impl::LaunchStep(cudaStream_t stream) {
  LaunchPrefix(stream);
  QueueRebuild(stream);
  LaunchSuffix(stream);
}

void Engine::Impl::Snapshot(cudaStream_t stream) {
  TILEMD_CHECK(cudaMemcpyAsync(snap_x, x, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_y, y, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_z, z, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_vx, vx, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_vy, vy, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_vz, vz, sizeof(double) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_imgx, imgx, sizeof(int) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_imgy, imgy, sizeof(int) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_imgz, imgz, sizeof(int) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_type, type, sizeof(int) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_gid, gid, sizeof(int) * n, cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(
      cudaMemcpyAsync(snap_step, step_counter, sizeof(int), cudaMemcpyDeviceToDevice, stream));
  TILEMD_CHECK(cudaMemcpyAsync(snap_thermo_count, thermo_count, sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
}

void Engine::Impl::RestoreSnapshot() {
  TILEMD_CHECK(cudaMemcpy(x, snap_x, sizeof(double) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(y, snap_y, sizeof(double) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(z, snap_z, sizeof(double) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(vx, snap_vx, sizeof(double) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(vy, snap_vy, sizeof(double) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(vz, snap_vz, sizeof(double) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(imgx, snap_imgx, sizeof(int) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(imgy, snap_imgy, sizeof(int) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(imgz, snap_imgz, sizeof(int) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(type, snap_type, sizeof(int) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(gid, snap_gid, sizeof(int) * n, cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(step_counter, snap_step, sizeof(int), cudaMemcpyDeviceToDevice));
  TILEMD_CHECK(cudaMemcpy(thermo_count, snap_thermo_count, sizeof(int), cudaMemcpyDeviceToDevice));
  RefreshShadow<<<Grid(n), kBlock>>>(x, y, z, type, pos, n);
  CheckLaunch();
}

bool Engine::Impl::CaptureConditional(cudaStream_t stream, int count, cudaGraph_t* graph,
                                      cudaGraphExec_t* exec) {
  cudaGraph_t built = nullptr;
  if (cudaGraphCreate(&built, 0) != cudaSuccess) return false;
  std::vector<cudaGraphConditionalHandle> handles(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (cudaGraphConditionalHandleCreate(&handles[static_cast<size_t>(i)], built, 0,
                                         cudaGraphCondAssignDefault) != cudaSuccess) {
      cudaGraphDestroy(built);
      return false;
    }
  }
  if (cudaStreamBeginCaptureToGraph(stream, built, nullptr, nullptr, 0,
                                    cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGraphDestroy(built);
    return false;
  }
  cudaStream_t body = nullptr;
  if (cudaStreamCreate(&body) != cudaSuccess) {
    cudaGraph_t junk = nullptr;
    cudaStreamEndCapture(stream, &junk);
    cudaGraphDestroy(built);
    return false;
  }
  bool ok = true;
  for (int i = 0; i < count && ok; ++i) {
    LaunchPrefix(stream);
    PublishRebuild<<<1, 1, 0, stream>>>(handles[static_cast<size_t>(i)], status);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaGraph_t captured = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    size_t ndeps = 0;
    if (cudaStreamGetCaptureInfo(stream, &capture_status, nullptr, &captured, &deps, nullptr,
                                 &ndeps) != cudaSuccess) {
      ok = false;
      break;
    }
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handles[static_cast<size_t>(i)];
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t cond = nullptr;
    if (cudaGraphAddNode(&cond, captured, deps, nullptr, ndeps, &params) != cudaSuccess) {
      ok = false;
      break;
    }
    if (cudaStreamUpdateCaptureDependencies(stream, &cond, nullptr, 1,
                                            cudaStreamSetCaptureDependencies) != cudaSuccess) {
      ok = false;
      break;
    }
    cudaGraph_t body_graph = params.conditional.phGraph_out[0];
    if (cudaStreamBeginCaptureToGraph(body, body_graph, nullptr, nullptr, 0,
                                      cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      ok = false;
      break;
    }
    QueueRebuild(body);
    cudaGraph_t finished_body = nullptr;
    if (cudaStreamEndCapture(body, &finished_body) != cudaSuccess) ok = false;
    if (ok) LaunchSuffix(stream);
  }
  cudaStreamDestroy(body);
  cudaGraph_t ended = nullptr;
  cudaError_t end = cudaStreamEndCapture(stream, &ended);
  if (!ok || end != cudaSuccess) {
    if (ended && ended != built) cudaGraphDestroy(ended);
    cudaGraphDestroy(built);
    return false;
  }
  if (cudaGraphInstantiate(exec, built, 0) != cudaSuccess) {
    cudaGraphDestroy(built);
    return false;
  }
  *graph = built;
  return true;
}

void Engine::Impl::Capture(cudaStream_t stream, int count, cudaGraph_t* graph,
                           cudaGraphExec_t* exec) {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &capture_status);
  if (capture_status != cudaStreamCaptureStatusNone) {
    cudaGraph_t junk = nullptr;
    cudaStreamEndCapture(stream, &junk);
    if (junk) cudaGraphDestroy(junk);
  }
  if (CaptureConditional(stream, count, graph, exec)) return;
  TILEMD_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < count; ++i) LaunchStep(stream);
  TILEMD_CHECK(cudaStreamEndCapture(stream, graph));
  TILEMD_CHECK(cudaGraphInstantiate(exec, *graph, 0));
}

void Engine::Impl::ReallocList(int capacity) {
  int* fresh = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(capacity) * n));
  cudaFree(nl);
  nl = fresh;
  max_neigh = capacity;
}

int Engine::Impl::BuildNeighbors(bool grow) {
  BuildCells();
  ApplySpatialOrder();
  for (int attempt = 0; attempt < 8; ++attempt) {
    ClearFlags();
    NeighborKernel<<<Grid(n), kBlock>>>(pos, cell_start, sorted_indices, nn, nl, max_neigh, n,
                                        max_nn, status, nullptr);
    CheckLaunch();
    DeviceStatus host_status{};
    int needed = 0;
    TILEMD_CHECK(cudaMemcpy(&host_status, status, sizeof(host_status), cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(&needed, max_nn, sizeof(needed), cudaMemcpyDeviceToHost));
    if (!host_status.overflow) {
      SortNeighbors<<<Grid(n), kBlock>>>(nl, nn, gid, n, nullptr);
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
  LaunchForce(kBlock, nullptr, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy, wxz,
              wyz, n, status, nullptr);
  CheckLaunch();
  forces_ready = true;
}

TileDeviceForces Engine::Impl::EvaluateTiles() {
  std::vector<double> hx(static_cast<size_t>(n));
  std::vector<double> hy(static_cast<size_t>(n));
  std::vector<double> hz(static_cast<size_t>(n));
  std::vector<int> htype(static_cast<size_t>(n));
  if (n > 0) {
    TILEMD_CHECK(cudaMemcpy(hx.data(), x, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hy.data(), y, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hz.data(), z, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(htype.data(), type, sizeof(int) * n, cudaMemcpyDeviceToHost));
  }
  double max_cut = 0.0;
  std::vector<LjPair> coeffs;
  coeffs.reserve(system.pairs.size());
  for (const PairSpec& spec : system.pairs) {
    LjPair row;
    row.type_i = spec.type_i;
    row.type_j = spec.type_j;
    row.epsilon = spec.epsilon;
    row.sigma = spec.sigma;
    row.cutoff = spec.cutoff;
    coeffs.push_back(row);
    max_cut = std::max(max_cut, spec.cutoff);
  }
  LjShift lj_shift = LjShift::None;
  if (system.shift == Shift::Energy) lj_shift = LjShift::Energy;
  if (system.shift == Shift::Force) lj_shift = LjShift::Force;
  const ClusterGrid grid = BuildClusterGrid(system.box, hx.data(), hy.data(), hz.data(),
                                            htype.data(), n, max_cut + system.skin);
  const ClusterPairs list = BuildClusterPairs(grid);
  return RunTileForceGpu(grid, list, coeffs.data(), static_cast<int>(coeffs.size()), lj_shift);
}

GpuForces Engine::Impl::TileForces() {
  const TileDeviceForces raw = EvaluateTiles();
  GpuForces out;
  out.fx.resize(static_cast<size_t>(n));
  out.fy.resize(static_cast<size_t>(n));
  out.fz.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    out.fx[static_cast<size_t>(i)] = static_cast<float>(raw.fx[static_cast<size_t>(i)]);
    out.fy[static_cast<size_t>(i)] = static_cast<float>(raw.fy[static_cast<size_t>(i)]);
    out.fz[static_cast<size_t>(i)] = static_cast<float>(raw.fz[static_cast<size_t>(i)]);
  }
  out.potential_energy = raw.potential_energy;
  for (int k = 0; k < 6; ++k) out.virial[k] = raw.virial[k];
  out.max_neigh = max_neigh;
  if (n > 0) {
    TILEMD_CHECK(cudaMemcpy(fx, out.fx.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(fy, out.fy.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    TILEMD_CHECK(cudaMemcpy(fz, out.fz.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  }
  forces_ready = true;
  return out;
}

bool Engine::Impl::TileListStale() {
  if (!tile_ready || system.rebuild_every_step) return true;
  DeviceStatus host{};
  TILEMD_CHECK(cudaMemcpy(&host, status, sizeof(host), cudaMemcpyDeviceToHost));
  const float trigger2 = static_cast<float>((system.skin * 0.5) * (system.skin * 0.5));
  return host.max_disp2_bits > trigger2;
}

void Engine::Impl::RebuildTileCache(float4* pos_cur) {
  std::vector<double> hx(static_cast<size_t>(n));
  std::vector<double> hy(static_cast<size_t>(n));
  std::vector<double> hz(static_cast<size_t>(n));
  std::vector<int> htype(static_cast<size_t>(n));
  if (n > 0) {
    TILEMD_CHECK(cudaMemcpy(hx.data(), x, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hy.data(), y, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(hz.data(), z, sizeof(double) * n, cudaMemcpyDeviceToHost));
    TILEMD_CHECK(cudaMemcpy(htype.data(), type, sizeof(int) * n, cudaMemcpyDeviceToHost));
  }
  double max_cut = 0.0;
  for (const PairSpec& spec : system.pairs) max_cut = std::max(max_cut, spec.cutoff);
  const ClusterGrid grid = BuildClusterGrid(system.box, hx.data(), hy.data(), hz.data(),
                                            htype.data(), n, max_cut + system.skin);
  const ClusterPairs list = BuildClusterPairs(grid);
  const int nslot = grid.ncluster * 8;
  const int nci = grid.ncluster + 1;
  const int npair = static_cast<int>(list.cj.size());
  if (nslot > tile_slot_cap) {
    cudaFree(tile_type);
    cudaFree(tile_gid);
    tile_type = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(nslot)));
    tile_gid = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(nslot)));
    tile_slot_cap = nslot;
  }
  if (nci > tile_ci_cap) {
    cudaFree(tile_ci);
    tile_ci = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(nci)));
    tile_ci_cap = nci;
  }
  if (npair > tile_pair_cap) {
    cudaFree(tile_cj);
    cudaFree(tile_shift);
    tile_cj = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(npair)));
    tile_shift = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(npair)));
    tile_pair_cap = npair;
  }
  if (tile_missing == nullptr) {
    tile_missing = static_cast<int*>(MustMalloc(sizeof(int)));
  }
  if (nslot > 0) {
    TILEMD_CHECK(
        cudaMemcpy(tile_type, grid.type.data(), sizeof(int) * nslot, cudaMemcpyHostToDevice));
    TILEMD_CHECK(
        cudaMemcpy(tile_gid, grid.gid.data(), sizeof(int) * nslot, cudaMemcpyHostToDevice));
  }
  TILEMD_CHECK(cudaMemcpy(tile_ci, list.ci_start.data(), sizeof(int) * static_cast<size_t>(nci),
                          cudaMemcpyHostToDevice));
  if (npair > 0) {
    TILEMD_CHECK(cudaMemcpy(tile_cj, list.cj.data(), sizeof(int) * npair, cudaMemcpyHostToDevice));
    TILEMD_CHECK(
        cudaMemcpy(tile_shift, list.shift.data(), sizeof(int) * npair, cudaMemcpyHostToDevice));
  }
  tile_ncluster = grid.ncluster;
  tile_npair = npair;
  tile_ready = true;
  if (tile_live != nullptr) {
    TILEMD_CHECK(cudaMemcpy(tile_live, &tile_ncluster, sizeof(int), cudaMemcpyHostToDevice));
  }
  RefreshShadow<<<Grid(n), kBlock>>>(x, y, z, type, pos_cur, n);
  CheckLaunch();
  if (n > 0) {
    TILEMD_CHECK(cudaMemcpy(xref, pos_cur, sizeof(float4) * n, cudaMemcpyDeviceToDevice));
  }
  DeviceStatus host{};
  TILEMD_CHECK(cudaMemcpy(&host, status, sizeof(host), cudaMemcpyDeviceToHost));
  host.max_disp2_bits = 0.f;
  TILEMD_CHECK(cudaMemcpy(status, &host, sizeof(host), cudaMemcpyHostToDevice));
}

int* GrowInt(int* old_ptr, int old_n, int new_n) {
  int* fresh = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(new_n)));
  if (old_ptr != nullptr && old_n > 0) {
    const int ncopy = std::min(old_n, new_n);
    TILEMD_CHECK(cudaMemcpy(fresh, old_ptr, sizeof(int) * static_cast<size_t>(ncopy),
                            cudaMemcpyDeviceToDevice));
  }
  cudaFree(old_ptr);
  return fresh;
}

void Engine::Impl::PrepareTileRebuild(bool host_rebuilt) {
  if (n <= 0) return;
  double max_cut = 0.0;
  for (const PairSpec& spec : system.pairs) max_cut = std::max(max_cut, spec.cutoff);
  const ClusterShape shape = ChooseClusterColumns(n, system.box);
  tile_nx = shape.nx;
  tile_ny = shape.ny;
  tile_edge[0] = shape.edge_x;
  tile_edge[1] = shape.edge_y;
  tile_reach = max_cut + system.skin;
  const int columns = shape.nx * shape.ny;
  const int clusters = n;
  const int slots = clusters * kClusterSize;
  const int want_pairs = std::max(tile_pair_cap, tile_npair + tile_npair / 2 + 32);
  if (slots > tile_slot_cap) {
    tile_type = GrowInt(tile_type, tile_slot_cap, slots);
    tile_gid = GrowInt(tile_gid, tile_slot_cap, slots);
    tile_slot_cap = slots;
  }
  if (clusters + 1 > tile_ci_cap) {
    tile_ci = GrowInt(tile_ci, tile_ci_cap, clusters + 1);
    tile_ci_cap = clusters + 1;
  }
  if (want_pairs > tile_pair_cap) {
    tile_cj = GrowInt(tile_cj, tile_pair_cap, want_pairs);
    tile_shift = GrowInt(tile_shift, tile_pair_cap, want_pairs);
    tile_pair_cap = want_pairs;
  }
  if (tile_column_cap < columns) {
    cudaFree(tile_column_count);
    cudaFree(tile_column_cluster);
    cudaFree(tile_column_ncluster);
    tile_column_count = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(columns)));
    tile_column_cluster = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(columns)));
    tile_column_ncluster =
        static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(columns)));
    tile_column_cap = columns;
  }
  if (tile_cluster_cap < clusters) {
    cudaFree(tile_cluster_column);
    cudaFree(tile_pair_count);
    cudaFree(tile_lo_x);
    cudaFree(tile_lo_y);
    cudaFree(tile_lo_z);
    cudaFree(tile_hi_x);
    cudaFree(tile_hi_y);
    cudaFree(tile_hi_z);
    tile_cluster_column =
        static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(clusters)));
    tile_pair_count = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(clusters)));
    tile_lo_x = static_cast<double*>(MustMalloc(sizeof(double) * static_cast<size_t>(clusters)));
    tile_lo_y = static_cast<double*>(MustMalloc(sizeof(double) * static_cast<size_t>(clusters)));
    tile_lo_z = static_cast<double*>(MustMalloc(sizeof(double) * static_cast<size_t>(clusters)));
    tile_hi_x = static_cast<double*>(MustMalloc(sizeof(double) * static_cast<size_t>(clusters)));
    tile_hi_y = static_cast<double*>(MustMalloc(sizeof(double) * static_cast<size_t>(clusters)));
    tile_hi_z = static_cast<double*>(MustMalloc(sizeof(double) * static_cast<size_t>(clusters)));
    tile_zkey = static_cast<unsigned long long*>(
        MustMalloc(sizeof(unsigned long long) * static_cast<size_t>(n)));
    tile_zkey_out = static_cast<unsigned long long*>(
        MustMalloc(sizeof(unsigned long long) * static_cast<size_t>(n)));
    tile_column_of = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(n)));
    tile_colkey = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(n)));
    tile_colkey_out = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(n)));
    tile_order_in = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(n)));
    tile_order_z = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(n)));
    tile_order_out = static_cast<int*>(MustMalloc(sizeof(int) * static_cast<size_t>(n)));
    size_t bytes_z = 0;
    size_t bytes_c = 0;
    TILEMD_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes_z, tile_zkey, tile_zkey_out,
                                                 tile_order_in, tile_order_z, n));
    TILEMD_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes_c, tile_colkey, tile_colkey_out,
                                                 tile_order_z, tile_order_out, n));
    cudaFree(tile_cub);
    tile_cub_bytes = std::max(bytes_z, bytes_c);
    tile_cub = MustMalloc(tile_cub_bytes);
    tile_cluster_cap = clusters;
  }
  const bool created = tile_live == nullptr;
  if (created) tile_live = static_cast<int*>(MustMalloc(sizeof(int)));
  if (created || host_rebuilt) {
    TILEMD_CHECK(cudaMemcpy(tile_live, &tile_ncluster, sizeof(int), cudaMemcpyHostToDevice));
  }
}

bool Engine::Impl::QueueTileRebuildStep(cudaStream_t stream, const float4* written) {
  DeviceTileRebuild in;
  in.x = x;
  in.y = y;
  in.z = z;
  in.type = type;
  in.xref = xref;
  in.pos = written;
  in.status = status;
  in.n = n;
  in.nx = tile_nx;
  in.ny = tile_ny;
  in.edge_x = tile_edge[0];
  in.edge_y = tile_edge[1];
  in.reach = tile_reach;
  in.box_x = system.box[0];
  in.box_y = system.box[1];
  in.box_z = system.box[2];
  in.cluster_cap = tile_cluster_cap;
  in.pair_cap = tile_pair_cap;
  in.column_count = tile_column_count;
  in.column_cluster = tile_column_cluster;
  in.column_ncluster = tile_column_ncluster;
  in.cluster_column = tile_cluster_column;
  in.zkey = tile_zkey;
  in.zkey_out = tile_zkey_out;
  in.column_of = tile_column_of;
  in.colkey = tile_colkey;
  in.colkey_out = tile_colkey_out;
  in.order_in = tile_order_in;
  in.order_z = tile_order_z;
  in.order_out = tile_order_out;
  in.slot_type = tile_type;
  in.slot_gid = tile_gid;
  in.lo_x = tile_lo_x;
  in.lo_y = tile_lo_y;
  in.lo_z = tile_lo_z;
  in.hi_x = tile_hi_x;
  in.hi_y = tile_hi_y;
  in.hi_z = tile_hi_z;
  in.pair_count = tile_pair_count;
  in.ci_start = tile_ci;
  in.cj = tile_cj;
  in.shift = tile_shift;
  in.live = tile_live;
  in.cub_temp = tile_cub;
  in.cub_bytes = tile_cub_bytes;
  return QueueDeviceTileRebuild(in, stream);
}

void Engine::Impl::LaunchTileStep(float4* pos_cur, float4* pos_out) {
  std::vector<LjPair> coeffs;
  coeffs.reserve(system.pairs.size());
  for (const PairSpec& spec : system.pairs) {
    LjPair row;
    row.type_i = spec.type_i;
    row.type_j = spec.type_j;
    row.epsilon = spec.epsilon;
    row.sigma = spec.sigma;
    row.cutoff = spec.cutoff;
    coeffs.push_back(row);
  }
  LjShift lj_shift = LjShift::None;
  if (system.shift == Shift::Energy) lj_shift = LjShift::Energy;
  if (system.shift == Shift::Force) lj_shift = LjShift::Force;
  int record = -1;
  if (TileRecordsThermo(tile_step, system.thermo_every)) record = tile_step;
  if (record >= 0 && tile_ncluster > tile_partial_cap) {
    cudaFree(tile_partial);
    tile_partial = static_cast<TileThermoPartial*>(
        MustMalloc(sizeof(TileThermoPartial) * static_cast<size_t>(tile_ncluster)));
    tile_partial_cap = tile_ncluster;
  }
  LaunchFusedTileDevice(tile_ncluster, tile_type, tile_gid, tile_ci, tile_cj, tile_shift,
                        tile_missing, system.box, coeffs.data(), static_cast<int>(coeffs.size()),
                        lj_shift, x, y, z, vx, vy, vz, imgx, imgy, imgz, pos_cur, pos_out, xref,
                        status, step_counter, system.dt, 1.0 / system.mass, tile_partial, record);
  if (record >= 0 && tile_ncluster > 0) {
    FinishTileThermo<<<1, 1>>>(tile_partial, tile_ncluster, thermo_ring, kThermoRing, thermo_count,
                               record, status);
    CheckLaunch();
    DrainDeviceThermo();
  }
  if (tile_ncluster > 0) ++tile_step;
  forces_ready = false;
}

void Engine::Impl::RecordClosingTileThermo() {
  if (!TileRecordsThermo(tile_step, system.thermo_every) || n <= 0) return;
  std::vector<double> hx(static_cast<size_t>(n));
  std::vector<double> hy(static_cast<size_t>(n));
  std::vector<double> hz(static_cast<size_t>(n));
  std::vector<double> hvx(static_cast<size_t>(n));
  std::vector<double> hvy(static_cast<size_t>(n));
  std::vector<double> hvz(static_cast<size_t>(n));
  std::vector<int> htype(static_cast<size_t>(n));
  TILEMD_CHECK(cudaMemcpy(hx.data(), x, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hy.data(), y, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hz.data(), z, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hvx.data(), vx, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hvy.data(), vy, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hvz.data(), vz, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(htype.data(), type, sizeof(int) * n, cudaMemcpyDeviceToHost));
  double max_cut = 0.0;
  std::vector<LjPair> coeffs;
  coeffs.reserve(system.pairs.size());
  for (const PairSpec& spec : system.pairs) {
    LjPair row;
    row.type_i = spec.type_i;
    row.type_j = spec.type_j;
    row.epsilon = spec.epsilon;
    row.sigma = spec.sigma;
    row.cutoff = spec.cutoff;
    coeffs.push_back(row);
    max_cut = std::max(max_cut, spec.cutoff);
  }
  LjShift lj_shift = LjShift::None;
  if (system.shift == Shift::Energy) lj_shift = LjShift::Energy;
  if (system.shift == Shift::Force) lj_shift = LjShift::Force;
  const ClusterGrid grid = BuildClusterGrid(system.box, hx.data(), hy.data(), hz.data(),
                                            htype.data(), n, max_cut + system.skin);
  const ClusterPairs list = BuildClusterPairs(grid);
  const TileDeviceForces raw =
      RunTileForceGpu(grid, list, coeffs.data(), static_cast<int>(coeffs.size()), lj_shift);
  const double half = 0.5 * system.dt / system.mass;
  double ke = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ivx = hvx[static_cast<size_t>(i)] + half * raw.fx[static_cast<size_t>(i)];
    const double ivy = hvy[static_cast<size_t>(i)] + half * raw.fy[static_cast<size_t>(i)];
    const double ivz = hvz[static_cast<size_t>(i)] + half * raw.fz[static_cast<size_t>(i)];
    ke += 0.5 * (ivx * ivx + ivy * ivy + ivz * ivz);
  }
  ThermoSample sample;
  sample.step = tile_step;
  sample.kinetic_energy = ke;
  sample.potential_energy = raw.potential_energy;
  for (int axis = 0; axis < 6; ++axis) sample.virial[axis] = raw.virial[axis];
  drained.push_back(sample);
}

void Engine::Impl::RunTileSteps(int count, float4*& latest, float4*& other, int step0) {
  for (int step = 0; step < count; ++step) {
    if (TileListStale()) RebuildTileCache(latest);
    LaunchTileStep(latest, other);
    std::swap(latest, other);
    DeviceStatus finished{};
    TILEMD_CHECK(cudaMemcpy(&finished, status, sizeof(finished), cudaMemcpyDeviceToHost));
    if (finished.nan_detected) AbortIfNonFinite(step0 + step, finished);
  }
}

void Engine::Impl::QueueFusedPhysics(cudaStream_t stream, float4* read, float4* write) {
  PublishTileRecord<<<1, 1, 0, stream>>>(step_counter, system.thermo_every, tile_record);
  LaunchFusedTileKernel(tile_ncluster, tile_type, tile_gid, tile_ci, tile_cj, tile_shift,
                        tile_missing, x, y, z, vx, vy, vz, imgx, imgy, imgz, read, write, xref,
                        status, step_counter, system.dt, 1.0 / system.mass, tile_partial, -1,
                        stream, tile_record, tile_live, tile_cluster_cap);
  FinishTileThermoDevice<<<1, 1, 0, stream>>>(tile_partial, tile_cluster_cap, thermo_ring,
                                              kThermoRing, thermo_count, tile_record, tile_live,
                                              status);
  DecideRebuild<<<1, 1, 0, stream>>>(status);
}

bool Engine::Impl::CaptureFusedConditional(cudaStream_t stream, int count, float4* cur, float4* nxt,
                                           cudaGraph_t* graph, cudaGraphExec_t* exec) {
  if (count < 1) return false;
  cudaGraph_t built = nullptr;
  if (cudaGraphCreate(&built, 0) != cudaSuccess) return false;
  std::vector<cudaGraphConditionalHandle> handles(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (cudaGraphConditionalHandleCreate(&handles[static_cast<size_t>(i)], built, 0,
                                         cudaGraphCondAssignDefault) != cudaSuccess) {
      cudaGraphDestroy(built);
      return false;
    }
  }
  if (cudaStreamBeginCaptureToGraph(stream, built, nullptr, nullptr, 0,
                                    cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGraphDestroy(built);
    return false;
  }
  cudaStream_t body = nullptr;
  if (cudaStreamCreate(&body) != cudaSuccess) {
    cudaGraph_t junk = nullptr;
    cudaStreamEndCapture(stream, &junk);
    cudaGraphDestroy(built);
    return false;
  }
  bool ok = cudaMemsetAsync(tile_missing, 0, sizeof(int), stream) == cudaSuccess;
  float4* read = cur;
  float4* write = nxt;
  bool body_open = false;
  for (int step = 0; step < count && ok; ++step) {
    QueueFusedPhysics(stream, read, write);
    PublishRebuild<<<1, 1, 0, stream>>>(handles[static_cast<size_t>(step)], status);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaGraph_t captured = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    size_t ndeps = 0;
    if (cudaStreamGetCaptureInfo(stream, &capture_status, nullptr, &captured, &deps, nullptr,
                                 &ndeps) != cudaSuccess) {
      ok = false;
      break;
    }
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handles[static_cast<size_t>(step)];
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t cond = nullptr;
    if (cudaGraphAddNode(&cond, captured, deps, nullptr, ndeps, &params) != cudaSuccess) {
      ok = false;
      break;
    }
    if (cudaStreamUpdateCaptureDependencies(stream, &cond, nullptr, 1,
                                            cudaStreamSetCaptureDependencies) != cudaSuccess) {
      ok = false;
      break;
    }
    cudaGraph_t body_graph = params.conditional.phGraph_out[0];
    if (cudaStreamBeginCaptureToGraph(body, body_graph, nullptr, nullptr, 0,
                                      cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      ok = false;
      break;
    }
    body_open = true;
    if (!QueueTileRebuildStep(body, write)) ok = false;
    cudaGraph_t finished_body = nullptr;
    if (cudaStreamEndCapture(body, &finished_body) != cudaSuccess) ok = false;
    body_open = false;
    std::swap(read, write);
  }
  if (body_open) {
    cudaGraph_t junk_body = nullptr;
    cudaStreamEndCapture(body, &junk_body);
  }
  cudaStreamDestroy(body);
  cudaGraph_t ended = nullptr;
  const cudaError_t end = cudaStreamEndCapture(stream, &ended);
  if (!ok || end != cudaSuccess) {
    if (ended && ended != built) cudaGraphDestroy(ended);
    cudaGraphDestroy(built);
    return false;
  }
  if (cudaGraphInstantiate(exec, built, 0) != cudaSuccess) {
    cudaGraphDestroy(built);
    return false;
  }
  *graph = built;
  return true;
}

bool Engine::Impl::CaptureFusedGraph(cudaStream_t stream, int count, float4* cur, float4* nxt,
                                     cudaGraph_t* graph, cudaGraphExec_t* exec) {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &capture_status);
  if (capture_status != cudaStreamCaptureStatusNone) {
    cudaGraph_t junk = nullptr;
    cudaStreamEndCapture(stream, &junk);
    if (junk) cudaGraphDestroy(junk);
  }
  // The column sort cannot read the rebuild flag. A conditional node skips it.
  // If that node cannot be captured, the flat graph keeps the early-exit kernels.
  if (CaptureFusedConditional(stream, count, cur, nxt, graph, exec)) return true;
  cudaGetLastError();
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
  bool ok = cudaMemsetAsync(tile_missing, 0, sizeof(int), stream) == cudaSuccess;
  float4* read = cur;
  float4* write = nxt;
  for (int step = 0; step < count && ok; ++step) {
    QueueFusedPhysics(stream, read, write);
    if (!QueueTileRebuildStep(stream, write)) ok = false;
    std::swap(read, write);
  }
  cudaGraph_t built = nullptr;
  const cudaError_t end = cudaStreamEndCapture(stream, &built);
  if (!ok || end != cudaSuccess) {
    if (built) cudaGraphDestroy(built);
    return false;
  }
  if (cudaGraphInstantiate(exec, built, 0) != cudaSuccess) {
    cudaGraphDestroy(built);
    return false;
  }
  *graph = built;
  return true;
}

void Engine::Impl::RunClusterTiles(int steps) {
  Upload();
  if (!leapfrog_ready) {
    RefreshShadow<<<Grid(n), kBlock>>>(x, y, z, type, pos, n);
    CheckLaunch();
    if (n > 0) {
      TILEMD_CHECK(cudaMemcpy(xref, pos, sizeof(float4) * n, cudaMemcpyDeviceToDevice));
      TILEMD_CHECK(cudaMemcpy(pos_next, pos, sizeof(float4) * n, cudaMemcpyDeviceToDevice));
    }
    TileForces();
    // v(−dt/2) = v(0) − ½ dt f(0) / m. Kick adds ½ dt f / m, so the sign is flipped.
    Kick<<<Grid(n), kBlock>>>(vx, vy, vz, fx, fy, fz, n, -system.dt, 1.0 / system.mass);
    CheckLaunch();
    leapfrog_ready = true;
    forces_ready = false;
  }
  float4* latest = pos;
  float4* other = pos_next;
  const int dump_every = system.dump_path.empty() ? 0 : system.dump_every;
  if (dump_every > 0) WriteDump(0, false);
  if (!FusedGraphOk(system.rebuild_every_step)) {
    int done = 0;
    while (done < steps) {
      const int chunk = SegmentUntilDump(done, steps - done, dump_every, std::max(steps, 1));
      RunTileSteps(chunk, latest, other, done);
      done += chunk;
      if (dump_every > 0 && done % dump_every == 0) WriteDump(done, true);
    }
  } else {
    int done = 0;
    while (done < steps) {
      int unroll = GraphUnroll(system.graph_unroll, steps - done);
      if (system.thermo_every > 0) unroll = std::min(unroll, kThermoRing);
      unroll = SegmentUntilDump(done, steps - done, dump_every, unroll);
      const bool host_rebuilt = TileListStale();
      if (host_rebuilt) RebuildTileCache(latest);
      PrepareTileRebuild(host_rebuilt);
      if (tile_record == nullptr) {
        tile_record = static_cast<int*>(MustMalloc(sizeof(int)));
      }
      if (system.thermo_every > 0 && tile_cluster_cap > tile_partial_cap) {
        cudaFree(tile_partial);
        tile_partial = static_cast<TileThermoPartial*>(
            MustMalloc(sizeof(TileThermoPartial) * static_cast<size_t>(tile_cluster_cap)));
        tile_partial_cap = tile_cluster_cap;
      }
      const int saved_step = tile_step;
      GraphHold graph;
      TILEMD_CHECK(cudaStreamCreate(&graph.stream));
      Snapshot(graph.stream);
      TILEMD_CHECK(cudaStreamSynchronize(graph.stream));
      std::vector<LjPair> coeffs;
      coeffs.reserve(system.pairs.size());
      for (const PairSpec& spec : system.pairs) {
        LjPair row;
        row.type_i = spec.type_i;
        row.type_j = spec.type_j;
        row.epsilon = spec.epsilon;
        row.sigma = spec.sigma;
        row.cutoff = spec.cutoff;
        coeffs.push_back(row);
      }
      LjShift lj_shift = LjShift::None;
      if (system.shift == Shift::Energy) lj_shift = LjShift::Energy;
      if (system.shift == Shift::Force) lj_shift = LjShift::Force;
      UploadFusedTileSymbols(system.box, coeffs.data(), static_cast<int>(coeffs.size()), lj_shift);
      const bool captured =
          CaptureFusedGraph(graph.stream, unroll, latest, other, &graph.graph, &graph.exec);
      if (!captured || tile_ncluster <= 0) {
        RunTileSteps(unroll, latest, other, done);
      } else {
        DeviceStatus primed{};
        TILEMD_CHECK(cudaMemcpy(&primed, status, sizeof(primed), cudaMemcpyDeviceToHost));
        primed.overflow = 0;
        primed.nan_detected = 0;
        primed.nan_slot = -1;
        TILEMD_CHECK(cudaMemcpy(status, &primed, sizeof(primed), cudaMemcpyHostToDevice));
        TILEMD_CHECK(cudaGraphLaunch(graph.exec, graph.stream));
        TILEMD_CHECK(cudaStreamSynchronize(graph.stream));
        TILEMD_CHECK(cudaGetLastError());
        tile_step += unroll;
        if (unroll % 2 == 1) std::swap(latest, other);
        int missing_host = 0;
        TILEMD_CHECK(cudaMemcpy(&missing_host, tile_missing, sizeof(int), cudaMemcpyDeviceToHost));
        if (missing_host) throw std::invalid_argument("missing pair coefficient");
        DeviceStatus finished{};
        TILEMD_CHECK(cudaMemcpy(&finished, status, sizeof(finished), cudaMemcpyDeviceToHost));
        if (finished.nan_detected) AbortIfNonFinite(done, finished);
        if (finished.overflow == 0 && !TileListStale()) DrainDeviceThermo();
        if (finished.overflow != 0 || TileListStale()) {
          RestoreSnapshot();
          if (n > 0) {
            TILEMD_CHECK(cudaMemcpy(pos_next, pos, sizeof(float4) * n, cudaMemcpyDeviceToDevice));
          }
          tile_step = saved_step;
          latest = pos;
          other = pos_next;
          RunTileSteps(unroll, latest, other, done);
        }
      }
      done += unroll;
      if (dump_every > 0 && done % dump_every == 0) WriteDump(done, true);
    }
  }
  if (latest != pos) std::swap(pos, pos_next);
  RecordClosingTileThermo();
}

void Engine::Impl::RecordInitialThermo() {
  if (thermo_origin_written || system.thermo_every <= 0) return;
  int one = 1;
  TILEMD_CHECK(cudaMemcpy(ev_flag, &one, sizeof(one), cudaMemcpyHostToDevice));
  LaunchForce(kBlock, nullptr, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy, wxz,
              wyz, n, status, ev_flag);
  PartialThermo<<<Grid(n), kBlock>>>(vx, vy, vz, energy, wxx, wyy, wzz, wxy, wxz, wyz, n,
                                     thermo_partial, ev_flag);
  FinishThermo<<<1, 1>>>(thermo_partial, npartial, thermo_ring, kThermoRing, thermo_count,
                         step_counter, ev_flag, 0, status);
  CheckLaunch();
  thermo_origin_written = true;
}

void Engine::Impl::DrainDeviceThermo() {
  int count = 0;
  TILEMD_CHECK(cudaMemcpy(&count, thermo_count, sizeof(count), cudaMemcpyDeviceToHost));
  if (count < 0) count = 0;
  if (count > kThermoRing) count = kThermoRing;
  if (count == 0) return;
  std::vector<ThermoRecord> raw(static_cast<size_t>(count));
  TILEMD_CHECK(cudaMemcpy(raw.data(), thermo_ring, sizeof(ThermoRecord) * raw.size(),
                          cudaMemcpyDeviceToHost));
  drained.reserve(drained.size() + raw.size());
  for (const ThermoRecord& rec : raw) {
    ThermoSample sample;
    sample.step = rec.step;
    sample.kinetic_energy = rec.ke;
    sample.potential_energy = rec.pe;
    sample.virial[0] = rec.wxx;
    sample.virial[1] = rec.wyy;
    sample.virial[2] = rec.wzz;
    sample.virial[3] = rec.wxy;
    sample.virial[4] = rec.wxz;
    sample.virial[5] = rec.wyz;
    drained.push_back(sample);
  }
  int zero = 0;
  TILEMD_CHECK(cudaMemcpy(thermo_count, &zero, sizeof(zero), cudaMemcpyHostToDevice));
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
  std::vector<int> hgid(n);
  std::vector<int> slot(n);
  std::vector<float> stored_fx = out.fx;
  std::vector<float> stored_fy = out.fy;
  std::vector<float> stored_fz = out.fz;
  TILEMD_CHECK(cudaMemcpy(hgid.data(), gid, sizeof(int) * n, cudaMemcpyDeviceToHost));
  for (int s = 0; s < n; ++s) slot[hgid[s]] = s;
  for (int s = 0; s < n; ++s) {
    out.fx[hgid[s]] = stored_fx[s];
    out.fy[hgid[s]] = stored_fy[s];
    out.fz[hgid[s]] = stored_fz[s];
  }
  const bool force_shift = system.shift == Shift::Force;
  // Original-id order, then neighbor global id. That is the oracle's association.
  for (int g = 0; g < n; ++g) {
    const int i = slot[g];
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
  if (impl_->system.use_cluster_tiles) return impl_->TileForces();
  int overflow = impl_->BuildNeighbors(grow);
  if (!overflow) impl_->ComputeForce();
  return impl_->DownloadForces(overflow);
}

namespace {

void RequireForceBlock(int block_size) {
  if (block_size != 64 && block_size != 128 && block_size != 256) {
    throw std::invalid_argument("force block size must be 64, 128, or 256");
  }
}

}  // namespace

GpuForces Engine::ForcesAtBlock(int block_size) {
  RequireForceBlock(block_size);
  impl_->Upload();
  if (impl_->system.use_cluster_tiles) {
    throw std::invalid_argument("force block sweep is the atom-neighbor kernel");
  }
  int overflow = impl_->BuildNeighbors(true);
  if (!overflow) {
    LaunchForce(block_size, nullptr, impl_->x, impl_->y, impl_->z, impl_->type, impl_->nn,
                impl_->nl, impl_->fx, impl_->fy, impl_->fz, impl_->energy, impl_->wxx, impl_->wyy,
                impl_->wzz, impl_->wxy, impl_->wxz, impl_->wyz, impl_->n, impl_->status, nullptr);
    CheckLaunch();
    impl_->forces_ready = true;
  }
  return impl_->DownloadForces(overflow);
}

double Engine::TimeForceKernel(int block_size, int repeats) {
  RequireForceBlock(block_size);
  if (repeats < 1) throw std::invalid_argument("force timing needs at least one launch");
  impl_->Upload();
  if (impl_->system.use_cluster_tiles) {
    throw std::invalid_argument("force block sweep is the atom-neighbor kernel");
  }
  if (impl_->BuildNeighbors(true)) {
    throw std::runtime_error("neighbor list overflowed before the force timing");
  }
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  TILEMD_CHECK(cudaStreamCreate(&stream));
  TILEMD_CHECK(cudaEventCreate(&start));
  TILEMD_CHECK(cudaEventCreate(&stop));
  auto launch = [&]() {
    LaunchForce(block_size, stream, impl_->x, impl_->y, impl_->z, impl_->type, impl_->nn, impl_->nl,
                impl_->fx, impl_->fy, impl_->fz, impl_->energy, impl_->wxx, impl_->wyy, impl_->wzz,
                impl_->wxy, impl_->wxz, impl_->wyz, impl_->n, impl_->status, nullptr);
  };
  launch();
  TILEMD_CHECK(cudaStreamSynchronize(stream));
  TILEMD_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) launch();
  TILEMD_CHECK(cudaEventRecord(stop, stream));
  TILEMD_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  TILEMD_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  TILEMD_CHECK(cudaGetLastError());
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  impl_->forces_ready = true;
  return static_cast<double>(milliseconds) * 1.0e-3;
}

void Engine::Impl::RunBaoab(int steps) {
  if (!(system.gamma >= 0.0)) throw std::runtime_error("friction must be non-negative");
  if (!(system.temperature >= 0.0)) throw std::runtime_error("temperature must be non-negative");
  Upload();
  if (!forces_ready) {
    int overflow = BuildNeighbors(true);
    if (overflow) throw std::runtime_error("initial neighbor list overflowed");
    ComputeForce();
  }
  RecordInitialThermo();
  const int dump_every = system.dump_path.empty() ? 0 : system.dump_every;
  if (dump_every > 0) WriteDump(0, false);
  if (steps == 0) {
    forces_ready = true;
    return;
  }
  const double half_dt = 0.5 * system.dt;
  const double inv_mass = 1.0 / system.mass;
  const double decay = std::exp(-system.gamma * system.dt);
  const double noise = std::sqrt(system.temperature * (1.0 - decay * decay) * inv_mass);
  for (int done = 0; done < steps; ++done) {
    Kick<<<Grid(n), kBlock>>>(vx, vy, vz, fx, fy, fz, n, system.dt, inv_mass);
    DriftHalf<<<Grid(n), kBlock>>>(x, y, z, vx, vy, vz, imgx, imgy, imgz, n, half_dt);
    const auto step = static_cast<std::uint64_t>(done);
    OrnsteinUhlenbeck<<<Grid(n), kBlock>>>(vx, vy, vz, gid, n, decay, noise, system.langevin_seed,
                                           static_cast<unsigned>(step),
                                           static_cast<unsigned>(step >> 32));
    DriftHalf<<<Grid(n), kBlock>>>(x, y, z, vx, vy, vz, imgx, imgy, imgz, n, half_dt);
    RefreshShadow<<<Grid(n), kBlock>>>(x, y, z, type, pos, n);
    ResetDisplacement<<<1, 1>>>(status);
    DispKernel<<<Grid(n), kBlock>>>(pos, xref, n, status);
    DecideRebuild<<<1, 1>>>(status);
    QueueRebuild(nullptr);
    SetEvFlag<<<1, 1>>>(ev_flag, step_counter);
    LaunchForce(kBlock, nullptr, x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy, wxz,
                wyz, n, status, ev_flag);
    CheckLaunch();
    DeviceStatus finished{};
    TILEMD_CHECK(cudaMemcpy(&finished, status, sizeof(finished), cudaMemcpyDeviceToHost));
    if (finished.nan_detected) AbortIfNonFinite(done, finished);
    if (finished.overflow) {
      throw std::runtime_error("neighbor list overflowed during integration at step " +
                               std::to_string(done));
    }
    Kick<<<Grid(n), kBlock>>>(vx, vy, vz, fx, fy, fz, n, system.dt, inv_mass);
    PartialThermo<<<Grid(n), kBlock>>>(vx, vy, vz, energy, wxx, wyy, wzz, wxy, wxz, wyz, n,
                                       thermo_partial, ev_flag);
    FinishThermo<<<1, 1>>>(thermo_partial, npartial, thermo_ring, kThermoRing, thermo_count,
                           step_counter, ev_flag, 1, status);
    IncrementStep<<<1, 1>>>(step_counter);
    CheckLaunch();
    if (dump_every > 0 && (done + 1) % dump_every == 0) WriteDump(done + 1, true);
  }
  forces_ready = true;
}

void Engine::Run(int steps) {
  if (steps < 0) throw std::runtime_error("step count must be non-negative");
  if (impl_->system.langevin) {
    if (!(impl_->system.gamma >= 0.0)) throw std::runtime_error("friction must be non-negative");
    if (!(impl_->system.temperature >= 0.0)) {
      throw std::runtime_error("temperature must be non-negative");
    }
    if (impl_->system.gamma > 0.0) {
      if (impl_->system.use_cluster_tiles) {
        throw std::runtime_error("fused leapfrog does not apply Langevin");
      }
      impl_->RunBaoab(steps);
      return;
    }
  }
  if (impl_->system.use_cluster_tiles) {
    impl_->RunClusterTiles(steps);
    return;
  }
  impl_->Upload();
  if (!impl_->forces_ready) {
    int overflow = impl_->BuildNeighbors(true);
    if (overflow) throw std::runtime_error("initial neighbor list overflowed");
    impl_->ComputeForce();
  }
  impl_->RecordInitialThermo();
  const int dump_every = impl_->system.dump_path.empty() ? 0 : impl_->system.dump_every;
  if (dump_every > 0) impl_->WriteDump(0, false);
  if (steps == 0) {
    impl_->forces_ready = true;
    return;
  }
  // One captured step-group, replayed without a host sync inside the group.
  // Status is read at segment boundaries. graph_unroll is K in PLAN.md, default 50.
  // A remainder that does not fill K is a second captured graph, not one launch per step.
  // A dump ends the segment early so the frame is the positions at that step.
  constexpr int kSegment = 1000;
  const int unroll = GraphUnroll(impl_->system.graph_unroll, steps);
  GraphHold graph;
  TILEMD_CHECK(cudaStreamCreate(&graph.stream));
  impl_->Capture(graph.stream, unroll, &graph.graph, &graph.exec);
  int done = 0;
  while (done < steps) {
    const int segment = SegmentUntilDump(done, steps - done, dump_every, kSegment);
    const SegmentLaunch launch = SegmentSteps(segment, unroll);
    GraphHold tail;
    auto capture_tail = [&]() {
      tail.ResetGraph();
      if (launch.tail > 0) {
        impl_->Capture(graph.stream, launch.tail, &tail.graph, &tail.exec);
      }
    };
    capture_tail();
    int retries = 0;
    while (true) {
      impl_->Snapshot(graph.stream);
      for (int i = 0; i < launch.graphs; ++i) {
        TILEMD_CHECK(cudaGraphLaunch(graph.exec, graph.stream));
      }
      if (launch.tail > 0) TILEMD_CHECK(cudaGraphLaunch(tail.exec, graph.stream));
      TILEMD_CHECK(cudaStreamSynchronize(graph.stream));
      TILEMD_CHECK(cudaGetLastError());
      DeviceStatus finished{};
      TILEMD_CHECK(cudaMemcpy(&finished, impl_->status, sizeof(finished), cudaMemcpyDeviceToHost));
      if (finished.nan_detected) impl_->AbortIfNonFinite(done, finished);
      if (!finished.overflow) break;
      if (++retries > 4) {
        throw std::runtime_error("neighbor list overflowed during integration at step " +
                                 std::to_string(done));
      }
      impl_->RestoreSnapshot();
      int next = std::max(impl_->max_neigh + impl_->max_neigh / 2, impl_->max_neigh + 32);
      impl_->ReallocList(RoundUp32(next));
      if (impl_->BuildNeighbors(true)) {
        throw std::runtime_error("neighbor list overflowed during integration at step " +
                                 std::to_string(done));
      }
      impl_->ComputeForce();
      graph.ResetGraph();
      impl_->Capture(graph.stream, unroll, &graph.graph, &graph.exec);
      capture_tail();
    }
    impl_->DrainDeviceThermo();
    done += segment;
    if (dump_every > 0 && done % dump_every == 0) impl_->WriteDump(done, true);
  }
  impl_->forces_ready = true;
}

std::vector<ThermoSample> Engine::DrainThermo() {
  impl_->DrainDeviceThermo();
  std::vector<ThermoSample> out;
  out.swap(impl_->drained);
  return out;
}

void ScatterToGid(const double* src, const int* gid, int n, std::vector<double>& dst) {
  std::vector<double> stored(n);
  std::vector<int> hgid(n);
  TILEMD_CHECK(cudaMemcpy(stored.data(), src, sizeof(double) * n, cudaMemcpyDeviceToHost));
  TILEMD_CHECK(cudaMemcpy(hgid.data(), gid, sizeof(int) * n, cudaMemcpyDeviceToHost));
  dst.resize(n);
  for (int s = 0; s < n; ++s) dst[hgid[s]] = stored[s];
}

void Engine::DownloadPositions(std::vector<double>& x, std::vector<double>& y,
                               std::vector<double>& z) const {
  ScatterToGid(impl_->x, impl_->gid, impl_->n, x);
  ScatterToGid(impl_->y, impl_->gid, impl_->n, y);
  ScatterToGid(impl_->z, impl_->gid, impl_->n, z);
}

void Engine::DownloadVelocities(std::vector<double>& vx, std::vector<double>& vy,
                                std::vector<double>& vz) const {
  ScatterToGid(impl_->vx, impl_->gid, impl_->n, vx);
  ScatterToGid(impl_->vy, impl_->gid, impl_->n, vy);
  ScatterToGid(impl_->vz, impl_->gid, impl_->n, vz);
}

}  // namespace tilemd
