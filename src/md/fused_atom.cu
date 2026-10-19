#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "force/lj_pair.hpp"
#include "force/shadow_delta.hpp"
#include "md/fused_atom.hpp"

namespace tilemd {
namespace {

struct AtomOut {
  float fx, fy, fz, energy;
  float wxx, wyy, wzz, wxy, wxz, wyz, pair_abs;
};

struct BoxCells {
  int nfull[3];
  float remainder[3];
};

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

BoxCells CellsOf(const double* box) {
  BoxCells cells{};
  for (int axis = 0; axis < 3; ++axis) {
    cells.nfull[axis] = static_cast<int>(std::floor(box[axis] / kShadowCell));
    const double remainder = box[axis] - static_cast<double>(cells.nfull[axis]) * kShadowCell;
    cells.remainder[axis] = static_cast<float>(remainder);
  }
  return cells;
}

__global__ void EncodeShadowKernel(const double* x, const double* y, const double* z, float4* pos,
                                   int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p;
  p.x = ShadowLocal(x[i]);
  p.y = ShadowLocal(y[i]);
  p.z = ShadowLocal(z[i]);
  p.w = __int_as_float(PackShadow(ShadowBin(x[i]), ShadowBin(y[i]), ShadowBin(z[i])));
  pos[i] = p;
}

template <int kThreadsPerAtom>
__global__ void __launch_bounds__(128)
    FusedAtomForceKernel(const float4* __restrict__ pos, const int* __restrict__ nn,
                         const int* __restrict__ nl, AtomOut* __restrict__ out, int n, Fp32Lj pair,
                         BoxCells cells) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int slot = tid / kThreadsPerAtom;
  const int sub = tid - slot * kThreadsPerAtom;
  if (slot >= n) return;

  const float4 pi = pos[slot];
  const int ix = ShadowAxis(__float_as_int(pi.w), 0);
  const int iy = ShadowAxis(__float_as_int(pi.w), 1);
  const int iz = ShadowAxis(__float_as_int(pi.w), 2);

  float fx = 0.f, fy = 0.f, fz = 0.f, energy = 0.f;
  float wxx = 0.f, wyy = 0.f, wzz = 0.f, wxy = 0.f, wxz = 0.f, wyz = 0.f, pair_abs = 0.f;
  const int count = nn[slot];
  for (int k = sub; k < count; k += kThreadsPerAtom) {
    const int j = nl[static_cast<long long>(k) * n + slot];
    const float4 pj = pos[j];
    const int jx = ShadowAxis(__float_as_int(pj.w), 0);
    const int jy = ShadowAxis(__float_as_int(pj.w), 1);
    const int jz = ShadowAxis(__float_as_int(pj.w), 2);
    const float dx = ShadowMinImage(ix, pi.x, jx, pj.x, cells.nfull[0], cells.remainder[0]);
    const float dy = ShadowMinImage(iy, pi.y, jy, pj.y, cells.nfull[1], cells.remainder[1]);
    const float dz = ShadowMinImage(iz, pi.z, jz, pj.z, cells.nfull[2], cells.remainder[2]);
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

#pragma unroll
  for (int mask = kThreadsPerAtom / 2; mask > 0; mask >>= 1) {
    fx += __shfl_xor_sync(0xffffffff, fx, mask);
    fy += __shfl_xor_sync(0xffffffff, fy, mask);
    fz += __shfl_xor_sync(0xffffffff, fz, mask);
    energy += __shfl_xor_sync(0xffffffff, energy, mask);
    wxx += __shfl_xor_sync(0xffffffff, wxx, mask);
    wyy += __shfl_xor_sync(0xffffffff, wyy, mask);
    wzz += __shfl_xor_sync(0xffffffff, wzz, mask);
    wxy += __shfl_xor_sync(0xffffffff, wxy, mask);
    wxz += __shfl_xor_sync(0xffffffff, wxz, mask);
    wyz += __shfl_xor_sync(0xffffffff, wyz, mask);
    pair_abs += __shfl_xor_sync(0xffffffff, pair_abs, mask);
  }

  if (sub == 0) {
    AtomOut row;
    row.fx = fx;
    row.fy = fy;
    row.fz = fz;
    row.energy = energy;
    row.wxx = wxx;
    row.wyy = wyy;
    row.wzz = wzz;
    row.wxy = wxy;
    row.wxz = wxz;
    row.wyz = wyz;
    row.pair_abs = pair_abs;
    out[slot] = row;
  }
}

template <int kThreadsPerAtom>
void LaunchForce(const float4* pos, const int* nn, const int* nl, AtomOut* out, int n, Fp32Lj pair,
                 BoxCells cells) {
  const int threads = n * kThreadsPerAtom;
  const int grid = (threads + 127) / 128;
  FusedAtomForceKernel<kThreadsPerAtom><<<grid, 128>>>(pos, nn, nl, out, n, pair, cells);
}

}  // namespace

Fp32Force FusedAtomForce(const double* box, const double* x, const double* y, const double* z,
                         const int* nn, const int* nl, int n, int max_neigh, const Fp32Lj& pair,
                         int threads_per_atom) {
  if (n <= 0) throw std::runtime_error("fused atom force needs at least one atom");
  if (threads_per_atom != 1 && threads_per_atom != 2 && threads_per_atom != 4 &&
      threads_per_atom != 8) {
    throw std::runtime_error("threads per atom must be 1, 2, 4, or 8");
  }

  double* dx = nullptr;
  double* dy = nullptr;
  double* dz = nullptr;
  float4* pos = nullptr;
  int* dnn = nullptr;
  int* dnl = nullptr;
  AtomOut* out = nullptr;
  const std::size_t atoms = static_cast<std::size_t>(n);
  const std::size_t list = static_cast<std::size_t>(max_neigh) * atoms;
  try {
    Check(cudaMalloc(&dx, atoms * sizeof(double)), "cudaMalloc x");
    Check(cudaMalloc(&dy, atoms * sizeof(double)), "cudaMalloc y");
    Check(cudaMalloc(&dz, atoms * sizeof(double)), "cudaMalloc z");
    Check(cudaMalloc(&pos, atoms * sizeof(float4)), "cudaMalloc pos");
    Check(cudaMalloc(&dnn, atoms * sizeof(int)), "cudaMalloc nn");
    Check(cudaMalloc(&dnl, list * sizeof(int)), "cudaMalloc nl");
    Check(cudaMalloc(&out, atoms * sizeof(AtomOut)), "cudaMalloc out");
    Check(cudaMemcpy(dx, x, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy x");
    Check(cudaMemcpy(dy, y, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy y");
    Check(cudaMemcpy(dz, z, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy z");
    Check(cudaMemcpy(dnn, nn, atoms * sizeof(int), cudaMemcpyHostToDevice), "copy nn");
    Check(cudaMemcpy(dnl, nl, list * sizeof(int), cudaMemcpyHostToDevice), "copy nl");

    const int encode_grid = (n + 127) / 128;
    EncodeShadowKernel<<<encode_grid, 128>>>(dx, dy, dz, pos, n);
    const BoxCells cells = CellsOf(box);
    if (threads_per_atom == 1) LaunchForce<1>(pos, dnn, dnl, out, n, pair, cells);
    if (threads_per_atom == 2) LaunchForce<2>(pos, dnn, dnl, out, n, pair, cells);
    if (threads_per_atom == 4) LaunchForce<4>(pos, dnn, dnl, out, n, pair, cells);
    if (threads_per_atom == 8) LaunchForce<8>(pos, dnn, dnl, out, n, pair, cells);
    Check(cudaGetLastError(), "fused atom force launch");
    Check(cudaDeviceSynchronize(), "fused atom force");

    std::vector<AtomOut> host(atoms);
    Check(cudaMemcpy(host.data(), out, atoms * sizeof(AtomOut), cudaMemcpyDeviceToHost),
          "copy forces");
    Fp32Force result;
    result.fx.resize(atoms);
    result.fy.resize(atoms);
    result.fz.resize(atoms);
    for (int i = 0; i < n; ++i) {
      result.fx[i] = host[i].fx;
      result.fy[i] = host[i].fy;
      result.fz[i] = host[i].fz;
      result.potential_energy += static_cast<double>(host[i].energy);
      result.virial[0] += static_cast<double>(host[i].wxx);
      result.virial[1] += static_cast<double>(host[i].wyy);
      result.virial[2] += static_cast<double>(host[i].wzz);
      result.virial[3] += static_cast<double>(host[i].wxy);
      result.virial[4] += static_cast<double>(host[i].wxz);
      result.virial[5] += static_cast<double>(host[i].wyz);
      result.pair_abs += static_cast<double>(host[i].pair_abs);
    }
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(pos);
    cudaFree(dnn);
    cudaFree(dnl);
    cudaFree(out);
    return result;
  } catch (...) {
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(pos);
    cudaFree(dnn);
    cudaFree(dnl);
    cudaFree(out);
    throw;
  }
}

}  // namespace tilemd
