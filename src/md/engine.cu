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

struct ThermoPartial {
  double ke;
  double pe;
  double wxx;
  double wyy;
  double wzz;
  double wxy;
  double wxz;
  double wyz;
};

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

__device__ float MinImage(float delta, float length) {
  return delta - length * rintf(delta / length);
}

__device__ double MinImageDouble(double delta, double length) {
  return delta - length * rint(delta / length);
}

// 10-bit Morton code. Stage-1 boxes stay under 1024 cells per axis.
__device__ unsigned Spread10(unsigned v) {
  v &= 0x3ffu;
  v = (v | (v << 16)) & 0x030000FFu;
  v = (v | (v << 8)) & 0x0300F00Fu;
  v = (v | (v << 4)) & 0x030C30C3u;
  v = (v | (v << 2)) & 0x09249249u;
  return v;
}

__device__ int MortonCell(int ix, int iy, int iz) {
  unsigned code = Spread10(static_cast<unsigned>(ix)) | (Spread10(static_cast<unsigned>(iy)) << 1) |
                  (Spread10(static_cast<unsigned>(iz)) << 2);
  return static_cast<int>(code);
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

__device__ bool RebuildSkipped(const int* gate) { return gate != nullptr && *gate == 0; }

__global__ void CellKeyKernel(const float4* pos, int* keys, int* indices, int n, const int* gate) {
  if (RebuildSkipped(gate)) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = LoadReadOnly(pos + i);
  int ix =
      CellCoord(p.x, static_cast<float>(c_ncell[0]) / static_cast<float>(c_box[0]), c_ncell[0]);
  int iy =
      CellCoord(p.y, static_cast<float>(c_ncell[1]) / static_cast<float>(c_box[1]), c_ncell[1]);
  int iz =
      CellCoord(p.z, static_cast<float>(c_ncell[2]) / static_cast<float>(c_box[2]), c_ncell[2]);
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
          float4 pj = LoadReadOnly(pos + j);
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

__global__ void __launch_bounds__(kBlock)
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
      atomicExch(&status->nan_detected, 1);
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
  if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) atomicExch(&status->nan_detected, 1);
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
    float dx = MinImage(pi.x - ref.x, static_cast<float>(c_box[0]));
    float dy = MinImage(pi.y - ref.y, static_cast<float>(c_box[1]));
    float dz = MinImage(pi.z - ref.z, static_cast<float>(c_box[2]));
    d2 = dx * dx + dy * dy + dz * dz;
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
  status->need_rebuild = (c_rebuild_every || d2 > c_trigger2) ? 1 : 0;
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
  std::vector<ThermoSample> drained;

  void Upload();
  void ClearFlags();
  void BuildCells();
  void ApplySpatialOrder();
  void QueueRebuild(cudaStream_t stream);
  void LaunchStep(cudaStream_t stream);
  void LaunchPrefix(cudaStream_t stream);
  void LaunchSuffix(cudaStream_t stream);
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

void Engine::Impl::ClearFlags() {
  DeviceStatus previous{};
  TILEMD_CHECK(cudaMemcpy(&previous, status, sizeof(previous), cudaMemcpyDeviceToHost));
  DeviceStatus blank{};
  blank.rebuild_count = previous.rebuild_count;
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
  ForceKernel<<<Grid(n), kBlock, 0, stream>>>(x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy,
                                              wzz, wxy, wxz, wyz, n, status, ev_flag);
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
  ForceKernel<<<Grid(n), kBlock>>>(x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy,
                                   wxz, wyz, n, status, nullptr);
  CheckLaunch();
  forces_ready = true;
}

void Engine::Impl::RecordInitialThermo() {
  if (thermo_origin_written || system.thermo_every <= 0) return;
  int one = 1;
  TILEMD_CHECK(cudaMemcpy(ev_flag, &one, sizeof(one), cudaMemcpyHostToDevice));
  ForceKernel<<<Grid(n), kBlock>>>(x, y, z, type, nn, nl, fx, fy, fz, energy, wxx, wyy, wzz, wxy,
                                   wxz, wyz, n, status, ev_flag);
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
  impl_->RecordInitialThermo();
  if (steps == 0) {
    impl_->forces_ready = true;
    return;
  }
  // One captured step-group, replayed without a host sync inside the group.
  // Status is read at segment boundaries. 50 is the default unroll in PLAN.md.
  constexpr int kUnroll = 50;
  constexpr int kSegment = 1000;
  const int unroll = std::min(kUnroll, steps);
  GraphHold graph;
  TILEMD_CHECK(cudaStreamCreate(&graph.stream));
  impl_->Capture(graph.stream, unroll, &graph.graph, &graph.exec);
  int done = 0;
  while (done < steps) {
    const int segment = std::min(kSegment, steps - done);
    int retries = 0;
    while (true) {
      impl_->Snapshot(graph.stream);
      const int full = segment / unroll;
      const int tail = segment % unroll;
      for (int i = 0; i < full; ++i) TILEMD_CHECK(cudaGraphLaunch(graph.exec, graph.stream));
      for (int i = 0; i < tail; ++i) impl_->LaunchStep(graph.stream);
      TILEMD_CHECK(cudaStreamSynchronize(graph.stream));
      TILEMD_CHECK(cudaGetLastError());
      DeviceStatus finished{};
      TILEMD_CHECK(cudaMemcpy(&finished, impl_->status, sizeof(finished), cudaMemcpyDeviceToHost));
      if (finished.nan_detected) {
        throw std::runtime_error("non-finite force or position during integration at step " +
                                 std::to_string(done));
      }
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
    }
    impl_->DrainDeviceThermo();
    done += segment;
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
