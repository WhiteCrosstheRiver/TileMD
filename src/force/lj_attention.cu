#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "force/lj_attention.hpp"
#include "neighbor/cluster_list.hpp"

namespace tilemd {
namespace {

void CheckCuda(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

__device__ double ShflDouble(double value, int source) {
  const int lo = __shfl_sync(0xffffffffu, __double2loint(value), source);
  const int hi = __shfl_sync(0xffffffffu, __double2hiint(value), source);
  return __hiloint2double(hi, lo);
}

__device__ double ReduceLane(double value, int i_local) {
  double sum = ShflDouble(value, i_local);
  sum += ShflDouble(value, i_local + 8);
  sum += ShflDouble(value, i_local + 16);
  sum += ShflDouble(value, i_local + 24);
  return sum;
}

__device__ void DecodeShift(int shift, int* sx, int* sy, int* sz) {
  *sx = shift % 3 - 1;
  *sy = (shift / 3) % 3 - 1;
  *sz = shift / 9 - 1;
}

// One warp per i-cluster. Lane = i_local + 8 * j_sub, matching the M5 tile kernel.
// The four lanes that share an i atom reduce before one of them writes that atom.
__global__ void __launch_bounds__(32)
    LjAttentionTileKernel(const double* x, const double* y, const double* z, const int* type,
                          const int* gid, const int* ci_start, const int* cj, const int* shift,
                          const LjRow* table, int force_shift, double lx, double ly, double lz,
                          int n, int ncluster, double* fx, double* fy, double* fz, double* energy,
                          double* virial) {
  const int lane = threadIdx.x;
  const int ci = blockIdx.x;
  if (ci >= ncluster) return;
  const int i_local = lane % kClusterSize;
  const int j_sub = lane / kClusterSize;
  const int slot_i = ci * kClusterSize + i_local;
  const int ti = type[slot_i];
  const int gi = gid[slot_i];
  const double xi = x[slot_i];
  const double yi = y[slot_i];
  const double zi = z[slot_i];

  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  double a1 = 0.0;
  double ae = 0.0;
  double axx = 0.0;
  double ayy = 0.0;
  double azz = 0.0;
  double axy = 0.0;
  double axz = 0.0;
  double ayz = 0.0;

  const int begin = ci_start[ci];
  const int end = ci_start[ci + 1];
  for (int pair0 = begin; pair0 < end; pair0 += 4) {
    const int pair = pair0 + j_sub;
    const bool active = pair < end;
    const int other = active ? cj[pair] : 0;
    int sx = 0;
    int sy = 0;
    int sz = 0;
    if (active) DecodeShift(shift[pair], &sx, &sy, &sz);
    const int slot_j = other * kClusterSize + i_local;
    const int tj = active ? type[slot_j] : kMaskType;
    double xj = 0.0;
    double yj = 0.0;
    double zj = 0.0;
    if (active && tj != kMaskType) {
      xj = x[slot_j] + static_cast<double>(sx) * lx;
      yj = y[slot_j] + static_cast<double>(sy) * ly;
      zj = z[slot_j] + static_cast<double>(sz) * lz;
    }
    for (int atom = 0; atom < kClusterSize; ++atom) {
      const int source = atom + kClusterSize * j_sub;
      const double bx = ShflDouble(xj, source);
      const double by = ShflDouble(yj, source);
      const double bz = ShflDouble(zj, source);
      const int bt = __shfl_sync(0xffffffffu, tj, source);
      if (!active || ti == kMaskType || bt == kMaskType) continue;
      if (ci == other && atom == i_local && sx == 0 && sy == 0 && sz == 0) continue;
      const LjRow row = table[ti * LjAttention::kMaxTypes + bt];
      if (!row.defined) continue;
      const double dx = xi - bx;
      const double dy = yi - by;
      const double dz = zi - bz;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 < row.rc2) || !(r2 > 0.0)) continue;
      double s = 0.0;
      double u = 0.0;
      LjEvaluate(row, r2, force_shift, &s, &u);
      ax += s * (xi - dx);
      ay += s * (yi - dy);
      az += s * (zi - dz);
      a1 += s;
      ae += 0.5 * u;
      const double fix = s * dx;
      const double fiy = s * dy;
      const double fiz = s * dz;
      axx += 0.5 * dx * fix;
      ayy += 0.5 * dy * fiy;
      azz += 0.5 * dz * fiz;
      axy += 0.5 * dx * fiy;
      axz += 0.5 * dx * fiz;
      ayz += 0.5 * dy * fiz;
    }
  }

  const double rx = ReduceLane(ax, i_local);
  const double ry = ReduceLane(ay, i_local);
  const double rz = ReduceLane(az, i_local);
  const double r1 = ReduceLane(a1, i_local);
  const double re = ReduceLane(ae, i_local);
  const double rxx = ReduceLane(axx, i_local);
  const double ryy = ReduceLane(ayy, i_local);
  const double rzz = ReduceLane(azz, i_local);
  const double rxy = ReduceLane(axy, i_local);
  const double rxz = ReduceLane(axz, i_local);
  const double ryz = ReduceLane(ayz, i_local);
  if (j_sub == 0 && ti != kMaskType && gi >= 0) {
    fx[gi] = xi * r1 - rx;
    fy[gi] = yi * r1 - ry;
    fz[gi] = zi * r1 - rz;
    energy[gi] = re;
    virial[0 * n + gi] = rxx;
    virial[1 * n + gi] = ryy;
    virial[2 * n + gi] = rzz;
    virial[3 * n + gi] = rxy;
    virial[4 * n + gi] = rxz;
    virial[5 * n + gi] = ryz;
  }
}

template <typename T>
T* Upload(const std::vector<T>& host, const char* what) {
  T* device = nullptr;
  if (host.empty()) return nullptr;
  CheckCuda(cudaMalloc(&device, host.size() * sizeof(T)), what);
  CheckCuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what);
  return device;
}

}  // namespace

AttentionResult RunLjAttentionGpu(const double* box, const double* x, const double* y,
                                  const double* z, const int* type, int n,
                                  const AttentionPair* pairs, int npair, AttentionShift shift,
                                  double skin) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  if (skin < 0.0) throw std::invalid_argument("skin must be non-negative");
  LjAttention potential(pairs, npair, shift);
  bool seen[LjAttention::kMaxTypes] = {};
  for (int i = 0; i < n; ++i) {
    if (type[i] < 0 || type[i] >= LjAttention::kMaxTypes) {
      throw std::invalid_argument("atom type is outside 0..7");
    }
    seen[type[i]] = true;
  }
  LjRow table[LjAttention::kMaxTypes * LjAttention::kMaxTypes];
  potential.CopyTable(table);
  for (int ti = 0; ti < LjAttention::kMaxTypes; ++ti) {
    if (!seen[ti]) continue;
    for (int tj = 0; tj < LjAttention::kMaxTypes; ++tj) {
      if (seen[tj] && !table[ti * LjAttention::kMaxTypes + tj].defined) {
        throw std::invalid_argument("missing pair coefficient");
      }
    }
  }

  AttentionResult out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  out.energy.assign(static_cast<size_t>(n), 0.0);
  if (n == 0) return out;

  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, potential.Cutoff() + skin);
  const ClusterPairs list = BuildClusterPairs(grid);

  double* dx = Upload(grid.x, "copy x");
  double* dy = Upload(grid.y, "copy y");
  double* dz = Upload(grid.z, "copy z");
  int* dtype = Upload(grid.type, "copy type");
  int* dgid = Upload(grid.gid, "copy gid");
  int* dstart = Upload(list.ci_start, "copy ci_start");
  int* dcj = Upload(list.cj, "copy cj");
  int* dshift = Upload(list.shift, "copy shift");
  LjRow* dtable = nullptr;
  double* dfx = nullptr;
  double* dfy = nullptr;
  double* dfz = nullptr;
  double* denergy = nullptr;
  double* dvirial = nullptr;
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  CheckCuda(cudaMalloc(&dtable, sizeof(table)), "cudaMalloc table");
  CheckCuda(cudaMalloc(&dfx, bytes), "cudaMalloc fx");
  CheckCuda(cudaMalloc(&dfy, bytes), "cudaMalloc fy");
  CheckCuda(cudaMalloc(&dfz, bytes), "cudaMalloc fz");
  CheckCuda(cudaMalloc(&denergy, bytes), "cudaMalloc energy");
  CheckCuda(cudaMalloc(&dvirial, 6 * bytes), "cudaMalloc virial");
  CheckCuda(cudaMemcpy(dtable, table, sizeof(table), cudaMemcpyHostToDevice), "copy table");
  CheckCuda(cudaMemset(dfx, 0, bytes), "zero fx");
  CheckCuda(cudaMemset(dfy, 0, bytes), "zero fy");
  CheckCuda(cudaMemset(dfz, 0, bytes), "zero fz");
  CheckCuda(cudaMemset(denergy, 0, bytes), "zero energy");
  CheckCuda(cudaMemset(dvirial, 0, 6 * bytes), "zero virial");

  if (grid.ncluster > 0) {
    LjAttentionTileKernel<<<grid.ncluster, 32>>>(
        dx, dy, dz, dtype, dgid, dstart, dcj, dshift, dtable, potential.ForceShift(), grid.box[0],
        grid.box[1], grid.box[2], n, grid.ncluster, dfx, dfy, dfz, denergy, dvirial);
    CheckCuda(cudaGetLastError(), "LjAttentionTileKernel");
    CheckCuda(cudaDeviceSynchronize(), "LjAttentionTileKernel sync");
  }

  std::vector<double> partial(static_cast<size_t>(6 * n));
  CheckCuda(cudaMemcpy(out.fx.data(), dfx, bytes, cudaMemcpyDeviceToHost), "copy fx");
  CheckCuda(cudaMemcpy(out.fy.data(), dfy, bytes, cudaMemcpyDeviceToHost), "copy fy");
  CheckCuda(cudaMemcpy(out.fz.data(), dfz, bytes, cudaMemcpyDeviceToHost), "copy fz");
  CheckCuda(cudaMemcpy(out.energy.data(), denergy, bytes, cudaMemcpyDeviceToHost), "copy energy");
  CheckCuda(cudaMemcpy(partial.data(), dvirial, 6 * bytes, cudaMemcpyDeviceToHost), "copy virial");
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dz);
  cudaFree(dtype);
  cudaFree(dgid);
  cudaFree(dstart);
  cudaFree(dcj);
  cudaFree(dshift);
  cudaFree(dtable);
  cudaFree(dfx);
  cudaFree(dfy);
  cudaFree(dfz);
  cudaFree(denergy);
  cudaFree(dvirial);

  for (int i = 0; i < n; ++i) {
    out.potential_energy += out.energy[static_cast<size_t>(i)];
    for (int k = 0; k < 6; ++k) {
      out.virial[k] += partial[static_cast<size_t>(k * n + i)];
    }
  }
  return out;
}

double TimeLjAttentionKernel(const double* box, const double* x, const double* y, const double* z,
                             const int* type, int n, const AttentionPair* pairs, int npair,
                             AttentionShift shift, double skin, int repeats) {
  if (repeats < 1) throw std::invalid_argument("attention timing needs at least one launch");
  if (n < 1) throw std::invalid_argument("attention timing needs atoms");
  LjAttention potential(pairs, npair, shift);
  LjRow table[LjAttention::kMaxTypes * LjAttention::kMaxTypes];
  potential.CopyTable(table);
  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, potential.Cutoff() + skin);
  const ClusterPairs list = BuildClusterPairs(grid);
  if (grid.ncluster < 1) return 0.0;

  struct CudaFree {
    void* ptr = nullptr;
    ~CudaFree() { cudaFree(ptr); }
  };
  CudaFree dx{Upload(grid.x, "copy x")};
  CudaFree dy{Upload(grid.y, "copy y")};
  CudaFree dz{Upload(grid.z, "copy z")};
  CudaFree dtype{Upload(grid.type, "copy type")};
  CudaFree dgid{Upload(grid.gid, "copy gid")};
  CudaFree dstart{Upload(list.ci_start, "copy ci_start")};
  CudaFree dcj{Upload(list.cj, "copy cj")};
  CudaFree dshift{Upload(list.shift, "copy shift")};
  CudaFree dtable;
  CudaFree dfx;
  CudaFree dfy;
  CudaFree dfz;
  CudaFree denergy;
  CudaFree dvirial;
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  CheckCuda(cudaMalloc(&dtable.ptr, sizeof(table)), "cudaMalloc table");
  CheckCuda(cudaMalloc(&dfx.ptr, bytes), "cudaMalloc fx");
  CheckCuda(cudaMalloc(&dfy.ptr, bytes), "cudaMalloc fy");
  CheckCuda(cudaMalloc(&dfz.ptr, bytes), "cudaMalloc fz");
  CheckCuda(cudaMalloc(&denergy.ptr, bytes), "cudaMalloc energy");
  CheckCuda(cudaMalloc(&dvirial.ptr, 6 * bytes), "cudaMalloc virial");
  CheckCuda(cudaMemcpy(dtable.ptr, table, sizeof(table), cudaMemcpyHostToDevice), "copy table");

  auto launch = [&]() {
    LjAttentionTileKernel<<<grid.ncluster, 32>>>(
        static_cast<double*>(dx.ptr), static_cast<double*>(dy.ptr), static_cast<double*>(dz.ptr),
        static_cast<int*>(dtype.ptr), static_cast<int*>(dgid.ptr), static_cast<int*>(dstart.ptr),
        static_cast<int*>(dcj.ptr), static_cast<int*>(dshift.ptr), static_cast<LjRow*>(dtable.ptr),
        potential.ForceShift(), grid.box[0], grid.box[1], grid.box[2], n, grid.ncluster,
        static_cast<double*>(dfx.ptr), static_cast<double*>(dfy.ptr), static_cast<double*>(dfz.ptr),
        static_cast<double*>(denergy.ptr), static_cast<double*>(dvirial.ptr));
  };
  launch();
  CheckCuda(cudaDeviceSynchronize(), "attention warmup");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CheckCuda(cudaEventCreate(&start), "event");
  CheckCuda(cudaEventCreate(&stop), "event");
  CheckCuda(cudaEventRecord(start), "event");
  for (int i = 0; i < repeats; ++i) launch();
  CheckCuda(cudaEventRecord(stop), "event");
  CheckCuda(cudaEventSynchronize(stop), "event");
  float milliseconds = 0.0f;
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop), "event");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  CheckCuda(cudaGetLastError(), "LjAttentionTileKernel");
  return static_cast<double>(milliseconds) * 1.0e-3;
}

}  // namespace tilemd
