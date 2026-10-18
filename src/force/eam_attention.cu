#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "force/eam_attention.hpp"
#include "force/eam_eval.hpp"
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

struct FuncflEval {
  const EamKnot* rho = nullptr;
  const EamKnot* z2 = nullptr;
  const double* fp = nullptr;
  int nr = 0;
  double rdr = 0.0;
  double cut2 = 0.0;

  __device__ PairOutput operator()(int pass, double r2, int, int, int i, int j) const {
    return EamFuncflPair(rho, z2, fp, nr, rdr, cut2, pass, r2, i, j);
  }
};

struct SetflEval {
  const EamKnot* rho = nullptr;
  const EamKnot* z2 = nullptr;
  const double* fp = nullptr;
  int nr = 0;
  int stride = 0;
  double rdr = 0.0;
  double cut2 = 0.0;

  __device__ PairOutput operator()(int pass, double r2, int type_i, int type_j, int i,
                                   int j) const {
    const EamKnot* rho_j = rho + static_cast<size_t>(type_j) * static_cast<size_t>(stride);
    const EamKnot* rho_i = rho + static_cast<size_t>(type_i) * static_cast<size_t>(stride);
    const int pair = EamPairIndex(type_i, type_j);
    const EamKnot* table = z2 + static_cast<size_t>(pair) * static_cast<size_t>(stride);
    return EamSetflPair(rho_j, rho_i, table, fp, nr, rdr, cut2, pass, r2, i, j);
  }
};

// One warp per i-cluster. Lane = i_local + 8 * j_sub. Pass 0 stores density.
// Later passes add force, pair energy, and virial. The embedding update is a
// separate kernel between the passes.
template <class Eval>
__global__ void __launch_bounds__(32)
    EamAttentionKernel(const double* x, const double* y, const double* z, const int* type,
                       const int* gid, const int* ci_start, const int* cj, const int* shift,
                       Eval eval, int pass, double lx, double ly, double lz, int n, int ncluster,
                       double* density, double* fx, double* fy, double* fz, double* energy,
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
  double av = 0.0;
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
    const int gj = active ? gid[slot_j] : -1;
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
      const int bg = __shfl_sync(0xffffffffu, gj, source);
      if (!active || ti == kMaskType || bt == kMaskType) continue;
      if (ci == other && atom == i_local && sx == 0 && sy == 0 && sz == 0) continue;
      const double dx = xi - bx;
      const double dy = yi - by;
      const double dz = zi - bz;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0)) continue;
      const PairOutput term = eval(pass, r2, ti, bt, gi, bg);
      ax += term.force * (xi - dx);
      ay += term.force * (yi - dy);
      az += term.force * (zi - dz);
      a1 += term.force;
      av += term.value;
      ae += 0.5 * term.energy;
      const double fix = term.force * dx;
      const double fiy = term.force * dy;
      const double fiz = term.force * dz;
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
  const double rv = ReduceLane(av, i_local);
  const double re = ReduceLane(ae, i_local);
  const double rxx = ReduceLane(axx, i_local);
  const double ryy = ReduceLane(ayy, i_local);
  const double rzz = ReduceLane(azz, i_local);
  const double rxy = ReduceLane(axy, i_local);
  const double rxz = ReduceLane(axz, i_local);
  const double ryz = ReduceLane(ayz, i_local);
  if (j_sub == 0 && ti != kMaskType && gi >= 0) {
    if (pass == 0) density[gi] = rv;
    fx[gi] += xi * r1 - rx;
    fy[gi] += yi * r1 - ry;
    fz[gi] += zi * r1 - rz;
    energy[gi] += re;
    virial[0 * n + gi] += rxx;
    virial[1 * n + gi] += ryy;
    virial[2 * n + gi] += rzz;
    virial[3 * n + gi] += rxy;
    virial[4 * n + gi] += rxz;
    virial[5 * n + gi] += ryz;
  }
}

__global__ void EmbedFuncflKernel(const EamKnot* frho, int nrho, double rdrho, double rhomax,
                                  const double* density, double* fp, double* energy, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double derivative = 0.0;
  energy[i] += EamEmbed(frho, nrho, rdrho, rhomax, density[i], &derivative);
  fp[i] = derivative;
}

__global__ void EmbedSetflKernel(const EamKnot* frho, const int* type, int nrho, int stride,
                                 double rdrho, double rhomax, const double* density, double* fp,
                                 double* energy, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const EamKnot* table = frho + static_cast<size_t>(type[i]) * static_cast<size_t>(stride);
  double derivative = 0.0;
  energy[i] += EamEmbed(table, nrho, rdrho, rhomax, density[i], &derivative);
  fp[i] = derivative;
}

template <typename T>
T* Upload(const std::vector<T>& host, const char* what) {
  T* device = nullptr;
  if (host.empty()) return nullptr;
  CheckCuda(cudaMalloc(&device, host.size() * sizeof(T)), what);
  CheckCuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what);
  return device;
}

struct DeviceAtoms {
  double* x = nullptr;
  double* y = nullptr;
  double* z = nullptr;
  int* type = nullptr;
  int* gid = nullptr;
  int* ci_start = nullptr;
  int* cj = nullptr;
  int* shift = nullptr;
  double* density = nullptr;
  double* fp = nullptr;
  double* fx = nullptr;
  double* fy = nullptr;
  double* fz = nullptr;
  double* energy = nullptr;
  double* virial = nullptr;

  void Release() const {
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(type);
    cudaFree(gid);
    cudaFree(ci_start);
    cudaFree(cj);
    cudaFree(shift);
    cudaFree(density);
    cudaFree(fp);
    cudaFree(fx);
    cudaFree(fy);
    cudaFree(fz);
    cudaFree(energy);
    cudaFree(virial);
  }
};

void CheckArguments(const double* box, int n, double skin, double cut) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  if (skin < 0.0) throw std::invalid_argument("skin must be non-negative");
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) throw std::invalid_argument("box length must be positive");
  }
  if (!(cut > 0.0)) throw std::invalid_argument("cutoff must be positive");
}

DeviceAtoms UploadAtoms(const ClusterGrid& grid, const ClusterPairs& list, int n) {
  DeviceAtoms dev;
  dev.x = Upload(grid.x, "copy x");
  dev.y = Upload(grid.y, "copy y");
  dev.z = Upload(grid.z, "copy z");
  dev.type = Upload(grid.type, "copy type");
  dev.gid = Upload(grid.gid, "copy gid");
  dev.ci_start = Upload(list.ci_start, "copy ci_start");
  dev.cj = Upload(list.cj, "copy cj");
  dev.shift = Upload(list.shift, "copy shift");
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  CheckCuda(cudaMalloc(&dev.density, bytes), "cudaMalloc density");
  CheckCuda(cudaMalloc(&dev.fp, bytes), "cudaMalloc fp");
  CheckCuda(cudaMalloc(&dev.fx, bytes), "cudaMalloc fx");
  CheckCuda(cudaMalloc(&dev.fy, bytes), "cudaMalloc fy");
  CheckCuda(cudaMalloc(&dev.fz, bytes), "cudaMalloc fz");
  CheckCuda(cudaMalloc(&dev.energy, bytes), "cudaMalloc energy");
  CheckCuda(cudaMalloc(&dev.virial, 6 * bytes), "cudaMalloc virial");
  CheckCuda(cudaMemset(dev.density, 0, bytes), "zero density");
  CheckCuda(cudaMemset(dev.fp, 0, bytes), "zero fp");
  CheckCuda(cudaMemset(dev.fx, 0, bytes), "zero fx");
  CheckCuda(cudaMemset(dev.fy, 0, bytes), "zero fy");
  CheckCuda(cudaMemset(dev.fz, 0, bytes), "zero fz");
  CheckCuda(cudaMemset(dev.energy, 0, bytes), "zero energy");
  CheckCuda(cudaMemset(dev.virial, 0, 6 * bytes), "zero virial");
  return dev;
}

AttentionResult Download(DeviceAtoms dev, int n) {
  AttentionResult out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  out.energy.assign(static_cast<size_t>(n), 0.0);
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  std::vector<double> partial(static_cast<size_t>(6 * n));
  CheckCuda(cudaMemcpy(out.fx.data(), dev.fx, bytes, cudaMemcpyDeviceToHost), "copy fx");
  CheckCuda(cudaMemcpy(out.fy.data(), dev.fy, bytes, cudaMemcpyDeviceToHost), "copy fy");
  CheckCuda(cudaMemcpy(out.fz.data(), dev.fz, bytes, cudaMemcpyDeviceToHost), "copy fz");
  CheckCuda(cudaMemcpy(out.energy.data(), dev.energy, bytes, cudaMemcpyDeviceToHost),
            "copy energy");
  CheckCuda(cudaMemcpy(partial.data(), dev.virial, 6 * bytes, cudaMemcpyDeviceToHost),
            "copy virial");
  dev.Release();
  for (int i = 0; i < n; ++i) {
    out.potential_energy += out.energy[static_cast<size_t>(i)];
    for (int k = 0; k < 6; ++k) out.virial[k] += partial[static_cast<size_t>(k * n + i)];
  }
  return out;
}

template <class Eval>
void LaunchPass(Eval eval, int pass, const ClusterGrid& grid, DeviceAtoms dev) {
  if (grid.ncluster <= 0) return;
  EamAttentionKernel<Eval><<<grid.ncluster, 32>>>(
      dev.x, dev.y, dev.z, dev.type, dev.gid, dev.ci_start, dev.cj, dev.shift, eval, pass,
      grid.box[0], grid.box[1], grid.box[2], grid.n_real, grid.ncluster, dev.density, dev.fx,
      dev.fy, dev.fz, dev.energy, dev.virial);
  CheckCuda(cudaGetLastError(), "EamAttentionKernel");
}

}  // namespace

AttentionResult RunEamFuncflGpu(const EamFuncfl& potential, const double* box, const double* x,
                                const double* y, const double* z, const int* type, int n,
                                double skin) {
  CheckArguments(box, n, skin, potential.Cutoff());
  AttentionResult empty;
  empty.fx.assign(static_cast<size_t>(n), 0.0);
  empty.fy.assign(static_cast<size_t>(n), 0.0);
  empty.fz.assign(static_cast<size_t>(n), 0.0);
  empty.energy.assign(static_cast<size_t>(n), 0.0);
  if (n == 0) return empty;

  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, potential.Cutoff() + skin);
  const ClusterPairs list = BuildClusterPairs(grid);
  DeviceAtoms dev = UploadAtoms(grid, list, n);
  const int knots = potential.KnotCountR() + 1;
  std::vector<EamKnot> rho(potential.DensityKnots(), potential.DensityKnots() + knots);
  std::vector<EamKnot> z2(potential.Z2Knots(), potential.Z2Knots() + knots);
  const int embed_knots = potential.KnotCountRho() + 1;
  std::vector<EamKnot> frho(potential.EmbeddingKnots(), potential.EmbeddingKnots() + embed_knots);
  EamKnot* drho = Upload(rho, "copy rho");
  EamKnot* dz2 = Upload(z2, "copy z2");
  EamKnot* dfrho = Upload(frho, "copy frho");
  FuncflEval eval;
  eval.rho = drho;
  eval.z2 = dz2;
  eval.fp = dev.fp;
  eval.nr = potential.KnotCountR();
  eval.rdr = potential.InvDr();
  eval.cut2 = potential.Cutoff() * potential.Cutoff();
  LaunchPass(eval, 0, grid, dev);
  const int blocks = (n + 127) / 128;
  EmbedFuncflKernel<<<blocks, 128>>>(dfrho, potential.KnotCountRho(), potential.InvDrho(),
                                     potential.RhoMax(), dev.density, dev.fp, dev.energy, n);
  CheckCuda(cudaGetLastError(), "EmbedFuncflKernel");
  LaunchPass(eval, 1, grid, dev);
  CheckCuda(cudaDeviceSynchronize(), "EAM funcfl sync");
  cudaFree(drho);
  cudaFree(dz2);
  cudaFree(dfrho);
  return Download(dev, n);
}

AttentionResult RunEamSetflGpu(const EamSetfl& potential, const double* box, const double* x,
                               const double* y, const double* z, const int* type, int n,
                               double skin) {
  CheckArguments(box, n, skin, potential.Cutoff());
  for (int i = 0; i < n; ++i) {
    if (type[i] < 0 || type[i] >= potential.NumElements()) {
      throw std::invalid_argument("atom type is outside the setfl elements");
    }
  }
  AttentionResult empty;
  empty.fx.assign(static_cast<size_t>(n), 0.0);
  empty.fy.assign(static_cast<size_t>(n), 0.0);
  empty.fz.assign(static_cast<size_t>(n), 0.0);
  empty.energy.assign(static_cast<size_t>(n), 0.0);
  if (n == 0) return empty;

  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, potential.Cutoff() + skin);
  const ClusterPairs list = BuildClusterPairs(grid);
  DeviceAtoms dev = UploadAtoms(grid, list, n);
  const int nr = potential.KnotCountR();
  const int nrho = potential.KnotCountRho();
  const int stride = nr + 1;
  const int embed_stride = nrho + 1;
  std::vector<EamKnot> rho;
  std::vector<EamKnot> frho;
  rho.reserve(static_cast<size_t>(potential.NumElements()) * static_cast<size_t>(stride));
  frho.reserve(static_cast<size_t>(potential.NumElements()) * static_cast<size_t>(embed_stride));
  for (int element = 0; element < potential.NumElements(); ++element) {
    const EamKnot* density = potential.DensityKnots(element);
    const EamKnot* embedding = potential.EmbeddingKnots(element);
    rho.insert(rho.end(), density, density + stride);
    frho.insert(frho.end(), embedding, embedding + embed_stride);
  }
  std::vector<EamKnot> z2;
  z2.reserve(static_cast<size_t>(potential.PairTableCount()) * static_cast<size_t>(stride));
  for (int pair = 0; pair < potential.PairTableCount(); ++pair) {
    const EamKnot* table = potential.Z2Knots(pair);
    z2.insert(z2.end(), table, table + stride);
  }
  std::vector<int> original_type(type, type + n);
  EamKnot* drho = Upload(rho, "copy rho");
  EamKnot* dz2 = Upload(z2, "copy z2");
  EamKnot* dfrho = Upload(frho, "copy frho");
  int* dtype = Upload(original_type, "copy atom type");
  SetflEval eval;
  eval.rho = drho;
  eval.z2 = dz2;
  eval.fp = dev.fp;
  eval.nr = nr;
  eval.stride = stride;
  eval.rdr = potential.InvDr();
  eval.cut2 = potential.Cutoff() * potential.Cutoff();
  LaunchPass(eval, 0, grid, dev);
  const int blocks = (n + 127) / 128;
  EmbedSetflKernel<<<blocks, 128>>>(dfrho, dtype, nrho, embed_stride, potential.InvDrho(),
                                    potential.RhoMax(), dev.density, dev.fp, dev.energy, n);
  CheckCuda(cudaGetLastError(), "EmbedSetflKernel");
  LaunchPass(eval, 1, grid, dev);
  CheckCuda(cudaDeviceSynchronize(), "EAM setfl sync");
  cudaFree(drho);
  cudaFree(dz2);
  cudaFree(dfrho);
  cudaFree(dtype);
  return Download(dev, n);
}

}  // namespace tilemd
