#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "force/lj_attention.hpp"

namespace tilemd {
namespace {

void CheckCuda(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

__global__ void LjAttentionKernel(int n, double lx, double ly, double lz, const double* x,
                                  const double* y, const double* z, const int* type,
                                  const LjRow* table, int force_shift, double* fx, double* fy,
                                  double* fz, double* energy, double* virial) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];
  const int ti = type[i];
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double sum_1 = 0.0;
  double pair_energy = 0.0;
  double vxx = 0.0;
  double vyy = 0.0;
  double vzz = 0.0;
  double vxy = 0.0;
  double vxz = 0.0;
  double vyz = 0.0;
  for (int j = 0; j < n; ++j) {
    if (i == j) continue;
    const double dx = xi - x[j] - lx * rint((xi - x[j]) / lx);
    const double dy = yi - y[j] - ly * rint((yi - y[j]) / ly);
    const double dz = zi - z[j] - lz * rint((zi - z[j]) / lz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    const LjRow row = table[ti * LjAttention::kMaxTypes + type[j]];
    if (!row.defined || !(r2 < row.rc2) || !(r2 > 0.0)) continue;
    double s = 0.0;
    double u = 0.0;
    LjEvaluate(row, r2, force_shift, &s, &u);
    sum_x += s * (xi - dx);
    sum_y += s * (yi - dy);
    sum_z += s * (zi - dz);
    sum_1 += s;
    pair_energy += 0.5 * u;
    const double fix = s * dx;
    const double fiy = s * dy;
    const double fiz = s * dz;
    vxx += 0.5 * dx * fix;
    vyy += 0.5 * dy * fiy;
    vzz += 0.5 * dz * fiz;
    vxy += 0.5 * dx * fiy;
    vxz += 0.5 * dx * fiz;
    vyz += 0.5 * dy * fiz;
  }
  fx[i] = xi * sum_1 - sum_x;
  fy[i] = yi * sum_1 - sum_y;
  fz[i] = zi * sum_1 - sum_z;
  energy[i] = pair_energy;
  virial[i] = vxx;
  virial[n + i] = vyy;
  virial[2 * n + i] = vzz;
  virial[3 * n + i] = vxy;
  virial[4 * n + i] = vxz;
  virial[5 * n + i] = vyz;
}

}  // namespace

AttentionResult RunLjAttentionGpu(const double* box, const double* x, const double* y,
                                  const double* z, const int* type, int n,
                                  const AttentionPair* pairs, int npair, AttentionShift shift) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
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

  double* dx = nullptr;
  double* dy = nullptr;
  double* dz = nullptr;
  int* dtype = nullptr;
  LjRow* dtable = nullptr;
  double* dfx = nullptr;
  double* dfy = nullptr;
  double* dfz = nullptr;
  double* denergy = nullptr;
  double* dvirial = nullptr;
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  CheckCuda(cudaMalloc(&dx, bytes), "cudaMalloc x");
  CheckCuda(cudaMalloc(&dy, bytes), "cudaMalloc y");
  CheckCuda(cudaMalloc(&dz, bytes), "cudaMalloc z");
  CheckCuda(cudaMalloc(&dtype, static_cast<size_t>(n) * sizeof(int)), "cudaMalloc type");
  CheckCuda(cudaMalloc(&dtable, sizeof(table)), "cudaMalloc table");
  CheckCuda(cudaMalloc(&dfx, bytes), "cudaMalloc fx");
  CheckCuda(cudaMalloc(&dfy, bytes), "cudaMalloc fy");
  CheckCuda(cudaMalloc(&dfz, bytes), "cudaMalloc fz");
  CheckCuda(cudaMalloc(&denergy, bytes), "cudaMalloc energy");
  CheckCuda(cudaMalloc(&dvirial, 6 * bytes), "cudaMalloc virial");
  CheckCuda(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice), "copy x");
  CheckCuda(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice), "copy y");
  CheckCuda(cudaMemcpy(dz, z, bytes, cudaMemcpyHostToDevice), "copy z");
  CheckCuda(cudaMemcpy(dtype, type, static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice),
            "copy type");
  CheckCuda(cudaMemcpy(dtable, table, sizeof(table), cudaMemcpyHostToDevice), "copy table");

  const int block = 128;
  const int grid = (n + block - 1) / block;
  LjAttentionKernel<<<grid, block>>>(n, box[0], box[1], box[2], dx, dy, dz, dtype, dtable,
                                     potential.ForceShift(), dfx, dfy, dfz, denergy, dvirial);
  CheckCuda(cudaGetLastError(), "LjAttentionKernel");
  CheckCuda(cudaDeviceSynchronize(), "LjAttentionKernel sync");

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

}  // namespace tilemd
