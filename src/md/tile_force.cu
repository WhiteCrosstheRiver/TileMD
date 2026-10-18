#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "core/device_status.hpp"
#include "md/tile_force.hpp"

namespace tilemd {
namespace {

constexpr int kTileTypes = 8;
constexpr int kMaskType = -1;

struct TileRow {
  double sigma2;
  double eps4;
  double eps24;
  double rc;
  double rc2;
  double u_shift;
  double s_rc;
  int defined;
};

__constant__ TileRow c_tile_rows[kTileTypes * kTileTypes];
__constant__ double c_tile_box[3];
__constant__ int c_tile_shift;

void Check(cudaError_t err) {
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t count) {
    if (count == 0) return;
    Check(cudaMalloc(&ptr, count * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

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

__device__ void DecodeTileShift(int shift, int* sx, int* sy, int* sz) {
  *sx = shift % 3 - 1;
  *sy = (shift / 3) % 3 - 1;
  *sz = shift / 9 - 1;
}

__device__ double WrapTile(double value, double length, int* image) {
  double periods = floor(value / length);
  int shift = static_cast<int>(periods);
  value -= periods * length;
  if (value >= length) {
    value -= length;
    ++shift;
  } else if (value < 0.0) {
    value += length;
    --shift;
  }
  *image += shift;
  return value;
}

__device__ float MinImageTile(float delta, float length) {
  return delta - length * rintf(delta / length);
}

// The cluster snapshot supplies neighbor coordinates. Only the lane that owns
// an atom writes that atom's position and velocity.
struct TileIntegrate {
  double* x = nullptr;
  double* y = nullptr;
  double* z = nullptr;
  double* vx = nullptr;
  double* vy = nullptr;
  double* vz = nullptr;
  int* imgx = nullptr;
  int* imgy = nullptr;
  int* imgz = nullptr;
  float4* pos_next = nullptr;
  float4* xref = nullptr;
  DeviceStatus* status = nullptr;
  int* step_counter = nullptr;
  double dt = 0.0;
  double inv_mass = 0.0;
};

template <bool kIntegrate>
__global__ void __launch_bounds__(32)
    TileForceKernel(const double* x, const double* y, const double* z, const int* type,
                    const int* gid, const int* ci_start, const int* cj, const int* shift,
                    double* fx, double* fy, double* fz, double* energy, double* wxx, double* wyy,
                    double* wzz, double* wxy, double* wxz, double* wyz, int ncluster, int* missing,
                    TileIntegrate integrate) {
  const int lane = threadIdx.x;
  const int ci = blockIdx.x;
  const int i_local = lane % 8;
  const int j_sub = lane / 8;
  if (ci >= ncluster) return;

  const int slot_i = ci * 8 + i_local;
  const int ti = type[slot_i];
  const double xi = x[slot_i];
  const double yi = y[slot_i];
  const double zi = z[slot_i];
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
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
    if (active) DecodeTileShift(shift[pair], &sx, &sy, &sz);
    const int slot_j = other * 8 + i_local;
    const double xj = active ? x[slot_j] + static_cast<double>(sx) * c_tile_box[0] : 0.0;
    const double yj = active ? y[slot_j] + static_cast<double>(sy) * c_tile_box[1] : 0.0;
    const double zj = active ? z[slot_j] + static_cast<double>(sz) * c_tile_box[2] : 0.0;
    const int tj = active ? type[slot_j] : kMaskType;

    for (int atom = 0; atom < 8; ++atom) {
      const int source = atom + 8 * j_sub;
      const double bx = ShflDouble(xj, source);
      const double by = ShflDouble(yj, source);
      const double bz = ShflDouble(zj, source);
      const int bt = __shfl_sync(0xffffffffu, tj, source);
      if (!active || ti == kMaskType || bt == kMaskType) continue;
      if (ci == other && atom == i_local && sx == 0 && sy == 0 && sz == 0) continue;
      const TileRow row = c_tile_rows[ti * kTileTypes + bt];
      if (!row.defined) {
        atomicExch(missing, 1);
        continue;
      }
      const double dx = xi - bx;
      const double dy = yi - by;
      const double dz = zi - bz;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 < row.rc2)) continue;
      const double inv_r2 = 1.0 / r2;
      const double sr2 = row.sigma2 * inv_r2;
      const double sr6 = sr2 * sr2 * sr2;
      double s = row.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
      double u = row.eps4 * sr6 * (sr6 - 1.0) - row.u_shift;
      if (c_tile_shift == 2) {
        const double r = sqrt(r2);
        u += (r - row.rc) * row.s_rc * row.rc;
        s -= row.s_rc * row.rc / r;
      }
      const double pfx = s * dx;
      const double pfy = s * dy;
      const double pfz = s * dz;
      ax += pfx;
      ay += pfy;
      az += pfz;
      ae += 0.5 * u;
      axx += 0.5 * dx * pfx;
      ayy += 0.5 * dy * pfy;
      azz += 0.5 * dz * pfz;
      axy += 0.5 * dx * pfy;
      axz += 0.5 * dx * pfz;
      ayz += 0.5 * dy * pfz;
    }
  }

  const double rx = ReduceLane(ax, i_local);
  const double ry = ReduceLane(ay, i_local);
  const double rz = ReduceLane(az, i_local);
  const double re = ReduceLane(ae, i_local);
  const double rxx = ReduceLane(axx, i_local);
  const double ryy = ReduceLane(ayy, i_local);
  const double rzz = ReduceLane(azz, i_local);
  const double rxy = ReduceLane(axy, i_local);
  const double rxz = ReduceLane(axz, i_local);
  const double ryz = ReduceLane(ayz, i_local);
  if constexpr (kIntegrate) {
    (void)re;
    (void)rxx;
    (void)ryy;
    (void)rzz;
    (void)rxy;
    (void)rxz;
    (void)ryz;
    double xn = 0.0;
    double yn = 0.0;
    double zn = 0.0;
    int atom_id = -1;
    if (j_sub == 0 && ti != kMaskType) {
      atom_id = gid[slot_i];
      const double scale = integrate.dt * integrate.inv_mass;
      const double vxn = integrate.vx[atom_id] + scale * rx;
      const double vyn = integrate.vy[atom_id] + scale * ry;
      const double vzn = integrate.vz[atom_id] + scale * rz;
      xn = WrapTile(integrate.x[atom_id] + integrate.dt * vxn, c_tile_box[0],
                    integrate.imgx + atom_id);
      yn = WrapTile(integrate.y[atom_id] + integrate.dt * vyn, c_tile_box[1],
                    integrate.imgy + atom_id);
      zn = WrapTile(integrate.z[atom_id] + integrate.dt * vzn, c_tile_box[2],
                    integrate.imgz + atom_id);
      integrate.vx[atom_id] = vxn;
      integrate.vy[atom_id] = vyn;
      integrate.vz[atom_id] = vzn;
      integrate.x[atom_id] = xn;
      integrate.y[atom_id] = yn;
      integrate.z[atom_id] = zn;
      integrate.pos_next[atom_id] = make_float4(static_cast<float>(xn), static_cast<float>(yn),
                                                static_cast<float>(zn), __int_as_float(ti));
    }
    float disp2 = 0.f;
    if (atom_id >= 0 && integrate.xref != nullptr) {
      const float4 ref = integrate.xref[atom_id];
      const float dx =
          MinImageTile(static_cast<float>(xn) - ref.x, static_cast<float>(c_tile_box[0]));
      const float dy =
          MinImageTile(static_cast<float>(yn) - ref.y, static_cast<float>(c_tile_box[1]));
      const float dz =
          MinImageTile(static_cast<float>(zn) - ref.z, static_cast<float>(c_tile_box[2]));
      disp2 = dx * dx + dy * dy + dz * dz;
    }
    float warp_max = disp2;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffffu, warp_max, offset));
    }
    if (lane == 0 && integrate.status != nullptr) {
      atomicMax(reinterpret_cast<int*>(&integrate.status->max_disp2_bits),
                __float_as_int(warp_max));
    }
    if (blockIdx.x == 0 && lane == 0 && integrate.step_counter != nullptr) {
      *integrate.step_counter += 1;
    }
  } else if (j_sub == 0 && ti != kMaskType) {
    const int atom_id = gid[slot_i];
    fx[atom_id] = rx;
    fy[atom_id] = ry;
    fz[atom_id] = rz;
    energy[atom_id] = re;
    wxx[atom_id] = rxx;
    wyy[atom_id] = ryy;
    wzz[atom_id] = rzz;
    wxy[atom_id] = rxy;
    wxz[atom_id] = rxz;
    wyz[atom_id] = ryz;
  }
}

}  // namespace

TileDeviceForces RunTileForceGpu(const ClusterGrid& grid, const ClusterPairs& list,
                                 const LjPair* pairs, int npair, LjShift shift) {
  if (npair < 1) throw std::invalid_argument("at least one pair coefficient is required");
  TileRow rows[kTileTypes * kTileTypes] = {};
  for (int p = 0; p < npair; ++p) {
    const LjPair& coeff = pairs[p];
    if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kTileTypes ||
        coeff.type_j >= kTileTypes) {
      throw std::invalid_argument("pair type is outside 0..7");
    }
    if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
      throw std::invalid_argument("sigma and cutoff must be positive");
    }
    auto fill = [&](int ti, int tj) {
      TileRow& row = rows[ti * kTileTypes + tj];
      row.defined = 1;
      row.sigma2 = coeff.sigma * coeff.sigma;
      row.eps4 = 4.0 * coeff.epsilon;
      row.eps24 = 24.0 * coeff.epsilon;
      row.rc = coeff.cutoff;
      row.rc2 = coeff.cutoff * coeff.cutoff;
      const double inv_rc2 = 1.0 / row.rc2;
      const double sr2 = row.sigma2 * inv_rc2;
      const double sr6 = sr2 * sr2 * sr2;
      const double u_c = row.eps4 * sr6 * (sr6 - 1.0);
      row.s_rc = row.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
      row.u_shift = (shift == LjShift::None) ? 0.0 : u_c;
    };
    fill(coeff.type_i, coeff.type_j);
    if (coeff.type_i != coeff.type_j) fill(coeff.type_j, coeff.type_i);
  }

  TileDeviceForces out;
  const int n = grid.n_real;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  if (n == 0 || grid.ncluster == 0) return out;

  const int shift_code = static_cast<int>(shift);
  Check(cudaMemcpyToSymbol(c_tile_rows, rows, sizeof(rows)));
  Check(cudaMemcpyToSymbol(c_tile_box, grid.box, sizeof(grid.box)));
  Check(cudaMemcpyToSymbol(c_tile_shift, &shift_code, sizeof(shift_code)));

  const int nslot = grid.ncluster * 8;
  DeviceBuffer<double> x(static_cast<size_t>(nslot));
  DeviceBuffer<double> y(static_cast<size_t>(nslot));
  DeviceBuffer<double> z(static_cast<size_t>(nslot));
  DeviceBuffer<int> type(static_cast<size_t>(nslot));
  DeviceBuffer<int> gid(static_cast<size_t>(nslot));
  DeviceBuffer<int> ci_start(list.ci_start.size());
  DeviceBuffer<int> cj(list.cj.size());
  DeviceBuffer<int> image(list.shift.size());
  DeviceBuffer<double> fx(static_cast<size_t>(n));
  DeviceBuffer<double> fy(static_cast<size_t>(n));
  DeviceBuffer<double> fz(static_cast<size_t>(n));
  DeviceBuffer<double> energy(static_cast<size_t>(n));
  DeviceBuffer<double> wxx(static_cast<size_t>(n));
  DeviceBuffer<double> wyy(static_cast<size_t>(n));
  DeviceBuffer<double> wzz(static_cast<size_t>(n));
  DeviceBuffer<double> wxy(static_cast<size_t>(n));
  DeviceBuffer<double> wxz(static_cast<size_t>(n));
  DeviceBuffer<double> wyz(static_cast<size_t>(n));
  DeviceBuffer<int> missing(1);

  Check(cudaMemcpy(x.ptr, grid.x.data(), sizeof(double) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(y.ptr, grid.y.data(), sizeof(double) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(z.ptr, grid.z.data(), sizeof(double) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(type.ptr, grid.type.data(), sizeof(int) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(gid.ptr, grid.gid.data(), sizeof(int) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(ci_start.ptr, list.ci_start.data(), sizeof(int) * list.ci_start.size(),
                   cudaMemcpyHostToDevice));
  if (!list.cj.empty()) {
    Check(cudaMemcpy(cj.ptr, list.cj.data(), sizeof(int) * list.cj.size(), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(image.ptr, list.shift.data(), sizeof(int) * list.shift.size(),
                     cudaMemcpyHostToDevice));
  }
  Check(cudaMemset(fx.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(fy.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(fz.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(energy.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(wxx.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(wyy.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(wzz.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(wxy.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(wxz.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(wyz.ptr, 0, sizeof(double) * n));
  Check(cudaMemset(missing.ptr, 0, sizeof(int)));

  TileForceKernel<false><<<grid.ncluster, 32>>>(
      x.ptr, y.ptr, z.ptr, type.ptr, gid.ptr, ci_start.ptr, cj.ptr, image.ptr, fx.ptr, fy.ptr,
      fz.ptr, energy.ptr, wxx.ptr, wyy.ptr, wzz.ptr, wxy.ptr, wxz.ptr, wyz.ptr, grid.ncluster,
      missing.ptr, TileIntegrate{});
  Check(cudaGetLastError());
  Check(cudaDeviceSynchronize());
  int missing_host = 0;
  Check(cudaMemcpy(&missing_host, missing.ptr, sizeof(int), cudaMemcpyDeviceToHost));
  if (missing_host) throw std::invalid_argument("missing pair coefficient");

  std::vector<double> atom_energy(static_cast<size_t>(n));
  std::vector<double> atom_wxx(static_cast<size_t>(n));
  std::vector<double> atom_wyy(static_cast<size_t>(n));
  std::vector<double> atom_wzz(static_cast<size_t>(n));
  std::vector<double> atom_wxy(static_cast<size_t>(n));
  std::vector<double> atom_wxz(static_cast<size_t>(n));
  std::vector<double> atom_wyz(static_cast<size_t>(n));
  Check(cudaMemcpy(out.fx.data(), fx.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(out.fy.data(), fy.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(out.fz.data(), fz.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_energy.data(), energy.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_wxx.data(), wxx.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_wyy.data(), wyy.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_wzz.data(), wzz.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_wxy.data(), wxy.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_wxz.data(), wxz.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(atom_wyz.data(), wyz.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    out.potential_energy += atom_energy[static_cast<size_t>(i)];
    out.virial[0] += atom_wxx[static_cast<size_t>(i)];
    out.virial[1] += atom_wyy[static_cast<size_t>(i)];
    out.virial[2] += atom_wzz[static_cast<size_t>(i)];
    out.virial[3] += atom_wxy[static_cast<size_t>(i)];
    out.virial[4] += atom_wxz[static_cast<size_t>(i)];
    out.virial[5] += atom_wyz[static_cast<size_t>(i)];
  }
  return out;
}

void RunFusedTileLeapfrog(const ClusterGrid& grid, const ClusterPairs& list, const LjPair* pairs,
                          int npair, LjShift shift, double* atom_x, double* atom_y, double* atom_z,
                          double* vx, double* vy, double* vz, int* imgx, int* imgy, int* imgz,
                          float4* pos_next, float4* xref, void* status, int* step_counter,
                          double dt, double inv_mass) {
  if (npair < 1) throw std::invalid_argument("at least one pair coefficient is required");
  if (!(dt > 0.0) || !(inv_mass > 0.0)) {
    throw std::invalid_argument("dt and inverse mass must be positive");
  }
  TileRow rows[kTileTypes * kTileTypes] = {};
  for (int p = 0; p < npair; ++p) {
    const LjPair& coeff = pairs[p];
    if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kTileTypes ||
        coeff.type_j >= kTileTypes) {
      throw std::invalid_argument("pair type is outside 0..7");
    }
    if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
      throw std::invalid_argument("sigma and cutoff must be positive");
    }
    auto fill = [&](int ti, int tj) {
      TileRow& row = rows[ti * kTileTypes + tj];
      row.defined = 1;
      row.sigma2 = coeff.sigma * coeff.sigma;
      row.eps4 = 4.0 * coeff.epsilon;
      row.eps24 = 24.0 * coeff.epsilon;
      row.rc = coeff.cutoff;
      row.rc2 = coeff.cutoff * coeff.cutoff;
      const double inv_rc2 = 1.0 / row.rc2;
      const double sr2 = row.sigma2 * inv_rc2;
      const double sr6 = sr2 * sr2 * sr2;
      const double u_c = row.eps4 * sr6 * (sr6 - 1.0);
      row.s_rc = row.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
      row.u_shift = (shift == LjShift::None) ? 0.0 : u_c;
    };
    fill(coeff.type_i, coeff.type_j);
    if (coeff.type_i != coeff.type_j) fill(coeff.type_j, coeff.type_i);
  }
  if (grid.n_real == 0 || grid.ncluster == 0) return;

  const int shift_code = static_cast<int>(shift);
  Check(cudaMemcpyToSymbol(c_tile_rows, rows, sizeof(rows)));
  Check(cudaMemcpyToSymbol(c_tile_box, grid.box, sizeof(grid.box)));
  Check(cudaMemcpyToSymbol(c_tile_shift, &shift_code, sizeof(shift_code)));

  const int nslot = grid.ncluster * 8;
  DeviceBuffer<double> x(static_cast<size_t>(nslot));
  DeviceBuffer<double> y(static_cast<size_t>(nslot));
  DeviceBuffer<double> z(static_cast<size_t>(nslot));
  DeviceBuffer<int> type(static_cast<size_t>(nslot));
  DeviceBuffer<int> gid(static_cast<size_t>(nslot));
  DeviceBuffer<int> ci_start(list.ci_start.size());
  DeviceBuffer<int> cj(list.cj.size());
  DeviceBuffer<int> image(list.shift.size());
  DeviceBuffer<int> missing(1);
  Check(cudaMemcpy(x.ptr, grid.x.data(), sizeof(double) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(y.ptr, grid.y.data(), sizeof(double) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(z.ptr, grid.z.data(), sizeof(double) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(type.ptr, grid.type.data(), sizeof(int) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(gid.ptr, grid.gid.data(), sizeof(int) * nslot, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(ci_start.ptr, list.ci_start.data(), sizeof(int) * list.ci_start.size(),
                   cudaMemcpyHostToDevice));
  if (!list.cj.empty()) {
    Check(cudaMemcpy(cj.ptr, list.cj.data(), sizeof(int) * list.cj.size(), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(image.ptr, list.shift.data(), sizeof(int) * list.shift.size(),
                     cudaMemcpyHostToDevice));
  }
  Check(cudaMemset(missing.ptr, 0, sizeof(int)));

  TileIntegrate step;
  step.x = atom_x;
  step.y = atom_y;
  step.z = atom_z;
  step.vx = vx;
  step.vy = vy;
  step.vz = vz;
  step.imgx = imgx;
  step.imgy = imgy;
  step.imgz = imgz;
  step.pos_next = pos_next;
  step.xref = xref;
  step.status = static_cast<DeviceStatus*>(status);
  step.step_counter = step_counter;
  step.dt = dt;
  step.inv_mass = inv_mass;
  TileForceKernel<true><<<grid.ncluster, 32>>>(x.ptr, y.ptr, z.ptr, type.ptr, gid.ptr, ci_start.ptr,
                                               cj.ptr, image.ptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, grid.ncluster, missing.ptr, step);
  Check(cudaGetLastError());
  Check(cudaDeviceSynchronize());
  int missing_host = 0;
  Check(cudaMemcpy(&missing_host, missing.ptr, sizeof(int), cudaMemcpyDeviceToHost));
  if (missing_host) throw std::invalid_argument("missing pair coefficient");
}

}  // namespace tilemd
