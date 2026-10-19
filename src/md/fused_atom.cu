#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
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
  const bool live = slot < n;

  float fx = 0.f, fy = 0.f, fz = 0.f, energy = 0.f;
  float wxx = 0.f, wyy = 0.f, wzz = 0.f, wxy = 0.f, wxz = 0.f, wyz = 0.f, pair_abs = 0.f;
  if (live) {
    const float4 pi = pos[slot];
    const int packed = __float_as_int(pi.w);
    const int ix = ShadowAxis(packed, 0);
    const int iy = ShadowAxis(packed, 1);
    const int iz = ShadowAxis(packed, 2);
    const int count = nn[slot];
    for (int k = sub; k < count; k += kThreadsPerAtom) {
      const int j = nl[static_cast<long long>(k) * n + slot];
      const float4 pj = pos[j];
      const int jpacked = __float_as_int(pj.w);
      const float dx = ShadowMinImage(ix, pi.x, ShadowAxis(jpacked, 0), pj.x, cells.nfull[0],
                                      cells.remainder[0]);
      const float dy = ShadowMinImage(iy, pi.y, ShadowAxis(jpacked, 1), pj.y, cells.nfull[1],
                                      cells.remainder[1]);
      const float dz = ShadowMinImage(iz, pi.z, ShadowAxis(jpacked, 2), pj.z, cells.nfull[2],
                                      cells.remainder[2]);
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

  if (live && sub == 0) {
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

__device__ void KickAxis(double* coordinate, double* velocity, float force, double dt, double mass,
                         double length, int* image) {
  const double updated = *velocity + dt * static_cast<double>(force) / mass;
  *velocity = updated;
  *coordinate = WrapImage(*coordinate + dt * updated, length, image);
}

template <int kThreadsPerAtom>
__global__ void __launch_bounds__(128)
    FusedAtomStepKernel(const float4* __restrict__ cur, float4* __restrict__ next, double* x,
                        double* y, double* z, double* vx, double* vy, double* vz, int* image_x,
                        int* image_y, int* image_z, const double* xref, const double* yref,
                        const double* zref, const int* __restrict__ nn, const int* __restrict__ nl,
                        int n, Fp32Lj pair, BoxCells cells, double dt, double mass, double lx,
                        double ly, double lz, int* max_disp_bits) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int slot = tid / kThreadsPerAtom;
  const int sub = tid - slot * kThreadsPerAtom;
  const bool live = slot < n;

  float fx = 0.f, fy = 0.f, fz = 0.f;
  if (live) {
    const float4 pi = cur[slot];
    const int packed = __float_as_int(pi.w);
    const int ix = ShadowAxis(packed, 0);
    const int iy = ShadowAxis(packed, 1);
    const int iz = ShadowAxis(packed, 2);
    const int count = nn[slot];
    for (int k = sub; k < count; k += kThreadsPerAtom) {
      const int j = nl[static_cast<long long>(k) * n + slot];
      const float4 pj = cur[j];
      const int jpacked = __float_as_int(pj.w);
      const float dx = ShadowMinImage(ix, pi.x, ShadowAxis(jpacked, 0), pj.x, cells.nfull[0],
                                      cells.remainder[0]);
      const float dy = ShadowMinImage(iy, pi.y, ShadowAxis(jpacked, 1), pj.y, cells.nfull[1],
                                      cells.remainder[1]);
      const float dz = ShadowMinImage(iz, pi.z, ShadowAxis(jpacked, 2), pj.z, cells.nfull[2],
                                      cells.remainder[2]);
      const LjTerm<float> term =
          LjPair<float>(dx, dy, dz, pair.sigma2, pair.eps4, pair.eps24, pair.rc, pair.rc2,
                        pair.u_shift, pair.s_rc, pair.shift);
      fx += term.fx;
      fy += term.fy;
      fz += term.fz;
    }
  }

#pragma unroll
  for (int mask = kThreadsPerAtom / 2; mask > 0; mask >>= 1) {
    fx += __shfl_xor_sync(0xffffffff, fx, mask);
    fy += __shfl_xor_sync(0xffffffff, fy, mask);
    fz += __shfl_xor_sync(0xffffffff, fz, mask);
  }

  float disp2 = 0.f;
  if (live && sub == 0) {
    int ix = image_x[slot];
    int iy = image_y[slot];
    int iz = image_z[slot];
    KickAxis(x + slot, vx + slot, fx, dt, mass, lx, &ix);
    KickAxis(y + slot, vy + slot, fy, dt, mass, ly, &iy);
    KickAxis(z + slot, vz + slot, fz, dt, mass, lz, &iz);
    image_x[slot] = ix;
    image_y[slot] = iy;
    image_z[slot] = iz;
    float4 encoded;
    encoded.x = ShadowLocal(x[slot]);
    encoded.y = ShadowLocal(y[slot]);
    encoded.z = ShadowLocal(z[slot]);
    encoded.w =
        __int_as_float(PackShadow(ShadowBin(x[slot]), ShadowBin(y[slot]), ShadowBin(z[slot])));
    next[slot] = encoded;
    const double dx = OnReferenceBranch(x[slot], xref[slot], lx) - xref[slot];
    const double dy = OnReferenceBranch(y[slot], yref[slot], ly) - yref[slot];
    const double dz = OnReferenceBranch(z[slot], zref[slot], lz) - zref[slot];
    disp2 = static_cast<float>(dx * dx + dy * dy + dz * dz);
  }

  __shared__ float scratch[128];
  scratch[threadIdx.x] = disp2;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax(max_disp_bits, __float_as_int(scratch[0]));
}

template <int kThreadsPerAtom>
void LaunchStep(const float4* cur, float4* next, double* x, double* y, double* z, double* vx,
                double* vy, double* vz, int* image_x, int* image_y, int* image_z,
                const double* xref, const double* yref, const double* zref, const int* nn,
                const int* nl, int n, Fp32Lj pair, BoxCells cells, double dt, double mass,
                double lx, double ly, double lz, int* max_disp_bits) {
  const int threads = n * kThreadsPerAtom;
  const int grid = (threads + 127) / 128;
  FusedAtomStepKernel<kThreadsPerAtom>
      <<<grid, 128>>>(cur, next, x, y, z, vx, vy, vz, image_x, image_y, image_z, xref, yref, zref,
                      nn, nl, n, pair, cells, dt, mass, lx, ly, lz, max_disp_bits);
}

struct Hold {
  void* pointer[20] = {};
  int count = 0;
  void Add(void* item) { pointer[count++] = item; }
  ~Hold() {
    for (int i = 0; i < count; ++i) cudaFree(pointer[i]);
  }
};

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

void FusedAtomNve(const double* box, double* x, double* y, double* z, double* vx, double* vy,
                  double* vz, int* image_x, int* image_y, int* image_z, const double* x_ref,
                  const double* y_ref, const double* z_ref, const int* nn, const int* nl, int n,
                  int max_neigh, const Fp32Lj& pair, double dt, double mass, int steps,
                  int threads_per_atom, Fp32NveStats* stats) {
  if (n <= 0 || steps <= 0) throw std::runtime_error("fused atom step needs atoms and steps");
  if (!(mass > 0.0) || !(dt > 0.0)) throw std::runtime_error("mass and dt must be positive");
  if (threads_per_atom != 1 && threads_per_atom != 2 && threads_per_atom != 4 &&
      threads_per_atom != 8) {
    throw std::runtime_error("threads per atom must be 1, 2, 4, or 8");
  }
  if (stats == nullptr) throw std::runtime_error("missing nve stats");

  const std::size_t atoms = static_cast<std::size_t>(n);
  const std::size_t list = static_cast<std::size_t>(max_neigh) * atoms;
  Hold hold;
  double* dx = nullptr;
  double* dy = nullptr;
  double* dz = nullptr;
  double* dvx = nullptr;
  double* dvy = nullptr;
  double* dvz = nullptr;
  double* xref = nullptr;
  double* yref = nullptr;
  double* zref = nullptr;
  int* ix = nullptr;
  int* iy = nullptr;
  int* iz = nullptr;
  int* dnn = nullptr;
  int* dnl = nullptr;
  int* bits = nullptr;
  float4* pos0 = nullptr;
  float4* pos1 = nullptr;
  Check(cudaMalloc(&dx, atoms * sizeof(double)), "cudaMalloc x");
  hold.Add(dx);
  Check(cudaMalloc(&dy, atoms * sizeof(double)), "cudaMalloc y");
  hold.Add(dy);
  Check(cudaMalloc(&dz, atoms * sizeof(double)), "cudaMalloc z");
  hold.Add(dz);
  Check(cudaMalloc(&dvx, atoms * sizeof(double)), "cudaMalloc vx");
  hold.Add(dvx);
  Check(cudaMalloc(&dvy, atoms * sizeof(double)), "cudaMalloc vy");
  hold.Add(dvy);
  Check(cudaMalloc(&dvz, atoms * sizeof(double)), "cudaMalloc vz");
  hold.Add(dvz);
  Check(cudaMalloc(&xref, atoms * sizeof(double)), "cudaMalloc xref");
  hold.Add(xref);
  Check(cudaMalloc(&yref, atoms * sizeof(double)), "cudaMalloc yref");
  hold.Add(yref);
  Check(cudaMalloc(&zref, atoms * sizeof(double)), "cudaMalloc zref");
  hold.Add(zref);
  Check(cudaMalloc(&ix, atoms * sizeof(int)), "cudaMalloc image x");
  hold.Add(ix);
  Check(cudaMalloc(&iy, atoms * sizeof(int)), "cudaMalloc image y");
  hold.Add(iy);
  Check(cudaMalloc(&iz, atoms * sizeof(int)), "cudaMalloc image z");
  hold.Add(iz);
  Check(cudaMalloc(&dnn, atoms * sizeof(int)), "cudaMalloc nn");
  hold.Add(dnn);
  Check(cudaMalloc(&dnl, list * sizeof(int)), "cudaMalloc nl");
  hold.Add(dnl);
  Check(cudaMalloc(&bits, sizeof(int)), "cudaMalloc disp");
  hold.Add(bits);
  Check(cudaMalloc(&pos0, atoms * sizeof(float4)), "cudaMalloc pos0");
  hold.Add(pos0);
  Check(cudaMalloc(&pos1, atoms * sizeof(float4)), "cudaMalloc pos1");
  hold.Add(pos1);

  Check(cudaMemcpy(dx, x, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy x");
  Check(cudaMemcpy(dy, y, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy y");
  Check(cudaMemcpy(dz, z, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy z");
  Check(cudaMemcpy(dvx, vx, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy vx");
  Check(cudaMemcpy(dvy, vy, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy vy");
  Check(cudaMemcpy(dvz, vz, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy vz");
  Check(cudaMemcpy(xref, x_ref, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy xref");
  Check(cudaMemcpy(yref, y_ref, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy yref");
  Check(cudaMemcpy(zref, z_ref, atoms * sizeof(double), cudaMemcpyHostToDevice), "copy zref");
  Check(cudaMemcpy(ix, image_x, atoms * sizeof(int), cudaMemcpyHostToDevice), "copy image x");
  Check(cudaMemcpy(iy, image_y, atoms * sizeof(int), cudaMemcpyHostToDevice), "copy image y");
  Check(cudaMemcpy(iz, image_z, atoms * sizeof(int), cudaMemcpyHostToDevice), "copy image z");
  Check(cudaMemcpy(dnn, nn, atoms * sizeof(int), cudaMemcpyHostToDevice), "copy nn");
  Check(cudaMemcpy(dnl, nl, list * sizeof(int), cudaMemcpyHostToDevice), "copy nl");

  const int encode_grid = (n + 127) / 128;
  EncodeShadowKernel<<<encode_grid, 128>>>(dx, dy, dz, pos0, n);
  const BoxCells cells = CellsOf(box);
  float4* buffers[2] = {pos0, pos1};
  int current = 0;
  stats->max_disp2 = 0.0;
  for (int step = 0; step < steps; ++step) {
    Check(cudaMemset(bits, 0, sizeof(int)), "clear disp");
    const int next = 1 - current;
    if (threads_per_atom == 1) {
      LaunchStep<1>(buffers[current], buffers[next], dx, dy, dz, dvx, dvy, dvz, ix, iy, iz, xref,
                    yref, zref, dnn, dnl, n, pair, cells, dt, mass, box[0], box[1], box[2], bits);
    } else if (threads_per_atom == 2) {
      LaunchStep<2>(buffers[current], buffers[next], dx, dy, dz, dvx, dvy, dvz, ix, iy, iz, xref,
                    yref, zref, dnn, dnl, n, pair, cells, dt, mass, box[0], box[1], box[2], bits);
    } else if (threads_per_atom == 4) {
      LaunchStep<4>(buffers[current], buffers[next], dx, dy, dz, dvx, dvy, dvz, ix, iy, iz, xref,
                    yref, zref, dnn, dnl, n, pair, cells, dt, mass, box[0], box[1], box[2], bits);
    } else {
      LaunchStep<8>(buffers[current], buffers[next], dx, dy, dz, dvx, dvy, dvz, ix, iy, iz, xref,
                    yref, zref, dnn, dnl, n, pair, cells, dt, mass, box[0], box[1], box[2], bits);
    }
    Check(cudaGetLastError(), "fused atom step launch");
    Check(cudaDeviceSynchronize(), "fused atom step");
    int host_bits = 0;
    Check(cudaMemcpy(&host_bits, bits, sizeof(int), cudaMemcpyDeviceToHost), "copy disp");
    float disp_bits = 0.f;
    std::memcpy(&disp_bits, &host_bits, sizeof(disp_bits));
    const double disp2 = static_cast<double>(disp_bits);
    if (disp2 > stats->max_disp2) stats->max_disp2 = disp2;
    current = next;
  }

  Check(cudaMemcpy(x, dx, atoms * sizeof(double), cudaMemcpyDeviceToHost), "read x");
  Check(cudaMemcpy(y, dy, atoms * sizeof(double), cudaMemcpyDeviceToHost), "read y");
  Check(cudaMemcpy(z, dz, atoms * sizeof(double), cudaMemcpyDeviceToHost), "read z");
  Check(cudaMemcpy(vx, dvx, atoms * sizeof(double), cudaMemcpyDeviceToHost), "read vx");
  Check(cudaMemcpy(vy, dvy, atoms * sizeof(double), cudaMemcpyDeviceToHost), "read vy");
  Check(cudaMemcpy(vz, dvz, atoms * sizeof(double), cudaMemcpyDeviceToHost), "read vz");
  Check(cudaMemcpy(image_x, ix, atoms * sizeof(int), cudaMemcpyDeviceToHost), "read image x");
  Check(cudaMemcpy(image_y, iy, atoms * sizeof(int), cudaMemcpyDeviceToHost), "read image y");
  Check(cudaMemcpy(image_z, iz, atoms * sizeof(int), cudaMemcpyDeviceToHost), "read image z");
}

}  // namespace tilemd
