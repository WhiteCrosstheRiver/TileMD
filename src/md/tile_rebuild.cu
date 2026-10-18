#include <cuda_runtime.h>

#include <algorithm>
#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>
#include <string>

#include "md/tile_rebuild.hpp"
#include "neighbor/cluster_list.hpp"
#include "neighbor/cluster_order.hpp"

namespace tilemd {
namespace {

constexpr int kBlock = 128;

#define TILEMD_REBUILD_CHECK(call)                                                       \
  do {                                                                                   \
    cudaError_t tilemd_err_ = (call);                                                    \
    if (tilemd_err_ != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(tilemd_err_)); \
    }                                                                                    \
  } while (0)

void* MustMalloc(size_t bytes) {
  void* ptr = nullptr;
  TILEMD_REBUILD_CHECK(cudaMalloc(&ptr, bytes));
  return ptr;
}

int Grid(int n) { return std::max(1, (n + kBlock - 1) / kBlock); }

__global__ void CountTileColumns(const double* x, const double* y, const double* z, int n, int nx,
                                 int ny, double edge_x, double edge_y, double box_x, double box_y,
                                 double box_z, int* column_count, int* column_of,
                                 unsigned long long* zkey, int* order_in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double wx = WrapCoord(x[i], box_x);
  const double wy = WrapCoord(y[i], box_y);
  const double wz = WrapCoord(z[i], box_z);
  const int ix = ColumnIndex(wx, edge_x, nx);
  const int iy = ColumnIndex(wy, edge_y, ny);
  const int column = ix + nx * iy;
  atomicAdd(column_count + column, 1);
  column_of[i] = column;
  zkey[i] = OrderedZ(wz);
  order_in[i] = i;
}

__global__ void GatherColumnKey(const int* column_of, const int* order_z, int* colkey, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  colkey[i] = column_of[order_z[i]];
}

__global__ void WriteTileSlots(const double* x, const double* y, const double* z, const int* type,
                               const int* order, const int* column_count, int nx, int ny,
                               double box_x, double box_y, double box_z, int cluster_cap,
                               int* column_cluster, int* column_ncluster, int* cluster_column,
                               int* slot_type, int* slot_gid, double* lo_x, double* lo_y,
                               double* lo_z, double* hi_x, double* hi_y, double* hi_z, int* live,
                               DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int sorted_at = 0;
  int cluster_cursor = 0;
  const int ncolumn = nx * ny;
  for (int column = 0; column < ncolumn; ++column) {
    const int count = column_count[column];
    const int nclus = count == 0 ? 0 : (count + kClusterSize - 1) / kClusterSize;
    if (cluster_cursor + nclus > cluster_cap) {
      status->overflow = 1;
      *live = 0;
      return;
    }
    column_cluster[column] = cluster_cursor;
    column_ncluster[column] = nclus;
    const int base = cluster_cursor * kClusterSize;
    for (int a = 0; a < count; ++a) {
      const int atom = order[sorted_at++];
      slot_type[base + a] = type[atom];
      slot_gid[base + a] = atom;
    }
    for (int pad = count; pad < nclus * kClusterSize; ++pad) {
      slot_type[base + pad] = kMaskType;
      slot_gid[base + pad] = -1;
    }
    for (int local = 0; local < nclus; ++local) {
      const int cluster = cluster_cursor + local;
      cluster_column[cluster] = column;
      ClusterBox box;
      bool any = false;
      for (int s = 0; s < kClusterSize; ++s) {
        const int slot = cluster * kClusterSize + s;
        if (slot_type[slot] == kMaskType) continue;
        const int atom = slot_gid[slot];
        IncludeInBox(&box, &any, WrapCoord(x[atom], box_x), WrapCoord(y[atom], box_y),
                     WrapCoord(z[atom], box_z));
      }
      lo_x[cluster] = box.lo_x;
      lo_y[cluster] = box.lo_y;
      lo_z[cluster] = box.lo_z;
      hi_x[cluster] = box.hi_x;
      hi_y[cluster] = box.hi_y;
      hi_z[cluster] = box.hi_z;
    }
    cluster_cursor += nclus;
  }
  *live = cluster_cursor;
}

__global__ void CountTilePairs(const int* column_cluster, const int* column_ncluster,
                               const int* cluster_column, const double* lo_x, const double* lo_y,
                               const double* lo_z, const double* hi_x, const double* hi_y,
                               const double* hi_z, const int* live, int nx, int ny, double reach,
                               double edge_x, double edge_y, double box_x, double box_y,
                               double box_z, int* pair_count) {
  const int ci = blockIdx.x * blockDim.x + threadIdx.x;
  const int ncluster = *live;
  if (ci >= ncluster) return;
  ClusterPairView view;
  view.lo_x = lo_x;
  view.lo_y = lo_y;
  view.lo_z = lo_z;
  view.hi_x = hi_x;
  view.hi_y = hi_y;
  view.hi_z = hi_z;
  view.column_cluster = column_cluster;
  view.column_ncluster = column_ncluster;
  view.cluster_column = cluster_column;
  view.nx = nx;
  view.ny = ny;
  view.reach = reach;
  view.edge_x = edge_x;
  view.edge_y = edge_y;
  view.box_x = box_x;
  view.box_y = box_y;
  view.box_z = box_z;
  int kept = 0;
  VisitClusterPairs(view, ci, [&](int, int) { ++kept; }, nullptr);
  pair_count[ci] = kept;
}

__global__ void ScanTilePairs(const int* pair_count, const int* live, int pair_cap, int* ci_start,
                              int* live_out, DeviceStatus* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int ncluster = *live;
  long long total = 0;
  for (int c = 0; c < ncluster; ++c) {
    if (total > pair_cap || total > 0x7fffffffLL) {
      status->overflow = 1;
      *live_out = 0;
      ci_start[0] = 0;
      return;
    }
    ci_start[c] = static_cast<int>(total);
    total += pair_count[c];
  }
  if (total > pair_cap || total > 0x7fffffffLL) {
    status->overflow = 1;
    *live_out = 0;
    ci_start[0] = 0;
    return;
  }
  if (ncluster >= 0) ci_start[ncluster] = static_cast<int>(total);
  *live_out = ncluster;
}

__global__ void FillTilePairs(const int* column_cluster, const int* column_ncluster,
                              const int* cluster_column, const double* lo_x, const double* lo_y,
                              const double* lo_z, const double* hi_x, const double* hi_y,
                              const double* hi_z, const int* live, const int* ci_start, int nx,
                              int ny, double reach, double edge_x, double edge_y, double box_x,
                              double box_y, double box_z, int* cj_out, int* shift_out) {
  const int ci = blockIdx.x * blockDim.x + threadIdx.x;
  const int ncluster = *live;
  if (ci >= ncluster) return;
  ClusterPairView view;
  view.lo_x = lo_x;
  view.lo_y = lo_y;
  view.lo_z = lo_z;
  view.hi_x = hi_x;
  view.hi_y = hi_y;
  view.hi_z = hi_z;
  view.column_cluster = column_cluster;
  view.column_ncluster = column_ncluster;
  view.cluster_column = cluster_column;
  view.nx = nx;
  view.ny = ny;
  view.reach = reach;
  view.edge_x = edge_x;
  view.edge_y = edge_y;
  view.box_x = box_x;
  view.box_y = box_y;
  view.box_z = box_z;
  int cursor = ci_start[ci];
  VisitClusterPairs(
      view, ci,
      [&](int cj, int shift) {
        cj_out[cursor] = cj;
        shift_out[cursor] = shift;
        ++cursor;
      },
      nullptr);
}

__global__ void PublishTileXref(float4* xref, const float4* pos, int n, DeviceStatus* status) {
  if (status->overflow) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) xref[i] = pos[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) status->max_disp2_bits = 0.f;
}

}  // namespace

bool QueueDeviceTileRebuild(const DeviceTileRebuild& in, cudaStream_t stream) {
  if (in.n <= 0 || in.nx <= 0 || in.ny <= 0) return true;
  const int ncolumn = in.nx * in.ny;
  if (cudaMemsetAsync(in.column_count, 0, sizeof(int) * static_cast<size_t>(ncolumn), stream) !=
      cudaSuccess) {
    return false;
  }
  if (cudaMemsetAsync(in.pair_count, 0, sizeof(int) * static_cast<size_t>(in.cluster_cap),
                      stream) != cudaSuccess) {
    return false;
  }
  CountTileColumns<<<Grid(in.n), kBlock, 0, stream>>>(
      in.x, in.y, in.z, in.n, in.nx, in.ny, in.edge_x, in.edge_y, in.box_x, in.box_y, in.box_z,
      in.column_count, in.column_of, in.zkey, in.order_in);
  size_t sort_bytes = in.cub_bytes;
  if (cub::DeviceRadixSort::SortPairs(
          in.cub_temp, sort_bytes, in.zkey, in.zkey_out, in.order_in, in.order_z, in.n, 0,
          static_cast<int>(sizeof(unsigned long long) * 8), stream) != cudaSuccess) {
    return false;
  }
  GatherColumnKey<<<Grid(in.n), kBlock, 0, stream>>>(in.column_of, in.order_z, in.colkey, in.n);
  sort_bytes = in.cub_bytes;
  if (cub::DeviceRadixSort::SortPairs(in.cub_temp, sort_bytes, in.colkey, in.colkey_out, in.order_z,
                                      in.order_out, in.n, 0, static_cast<int>(sizeof(int) * 8),
                                      stream) != cudaSuccess) {
    return false;
  }
  WriteTileSlots<<<1, 1, 0, stream>>>(in.x, in.y, in.z, in.type, in.order_out, in.column_count,
                                      in.nx, in.ny, in.box_x, in.box_y, in.box_z, in.cluster_cap,
                                      in.column_cluster, in.column_ncluster, in.cluster_column,
                                      in.slot_type, in.slot_gid, in.lo_x, in.lo_y, in.lo_z, in.hi_x,
                                      in.hi_y, in.hi_z, in.live, in.status);
  CountTilePairs<<<Grid(in.cluster_cap), kBlock, 0, stream>>>(
      in.column_cluster, in.column_ncluster, in.cluster_column, in.lo_x, in.lo_y, in.lo_z, in.hi_x,
      in.hi_y, in.hi_z, in.live, in.nx, in.ny, in.reach, in.edge_x, in.edge_y, in.box_x, in.box_y,
      in.box_z, in.pair_count);
  // Scan writes the published live count. Pair fill reads that count, so an
  // overflow leaves the next fused step with nothing to walk.
  ScanTilePairs<<<1, 1, 0, stream>>>(in.pair_count, in.live, in.pair_cap, in.ci_start, in.live,
                                     in.status);
  FillTilePairs<<<Grid(in.cluster_cap), kBlock, 0, stream>>>(
      in.column_cluster, in.column_ncluster, in.cluster_column, in.lo_x, in.lo_y, in.lo_z, in.hi_x,
      in.hi_y, in.hi_z, in.live, in.ci_start, in.nx, in.ny, in.reach, in.edge_x, in.edge_y,
      in.box_x, in.box_y, in.box_z, in.cj, in.shift);
  PublishTileXref<<<Grid(in.n), kBlock, 0, stream>>>(in.xref, in.pos, in.n, in.status);
  return cudaGetLastError() == cudaSuccess;
}

HostClusterList RebuildClusterListOnDevice(const double* box, const double* x, const double* y,
                                           const double* z, const int* type, int n, double reach) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  HostClusterList out;
  if (n == 0) return out;
  const ClusterShape shape = ChooseClusterColumns(n, box);
  const int clusters = n;
  const int slots = clusters * kClusterSize;
  const int columns = shape.nx * shape.ny;
  const int pair_cap = clusters * clusters * 3 + 8;
  DeviceTileRebuild in;
  in.n = n;
  in.nx = shape.nx;
  in.ny = shape.ny;
  in.edge_x = shape.edge_x;
  in.edge_y = shape.edge_y;
  in.reach = reach;
  in.box_x = box[0];
  in.box_y = box[1];
  in.box_z = box[2];
  in.cluster_cap = clusters;
  in.pair_cap = pair_cap;
  in.x = static_cast<double*>(MustMalloc(sizeof(double) * n));
  in.y = static_cast<double*>(MustMalloc(sizeof(double) * n));
  in.z = static_cast<double*>(MustMalloc(sizeof(double) * n));
  in.type = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.pos = static_cast<float4*>(MustMalloc(sizeof(float4) * n));
  in.xref = static_cast<float4*>(MustMalloc(sizeof(float4) * n));
  in.status = static_cast<DeviceStatus*>(MustMalloc(sizeof(DeviceStatus)));
  in.column_count = static_cast<int*>(MustMalloc(sizeof(int) * columns));
  in.column_cluster = static_cast<int*>(MustMalloc(sizeof(int) * columns));
  in.column_ncluster = static_cast<int*>(MustMalloc(sizeof(int) * columns));
  in.cluster_column = static_cast<int*>(MustMalloc(sizeof(int) * clusters));
  in.zkey = static_cast<unsigned long long*>(MustMalloc(sizeof(unsigned long long) * n));
  in.zkey_out = static_cast<unsigned long long*>(MustMalloc(sizeof(unsigned long long) * n));
  in.column_of = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.colkey = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.colkey_out = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.order_in = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.order_z = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.order_out = static_cast<int*>(MustMalloc(sizeof(int) * n));
  in.slot_type = static_cast<int*>(MustMalloc(sizeof(int) * slots));
  in.slot_gid = static_cast<int*>(MustMalloc(sizeof(int) * slots));
  in.lo_x = static_cast<double*>(MustMalloc(sizeof(double) * clusters));
  in.lo_y = static_cast<double*>(MustMalloc(sizeof(double) * clusters));
  in.lo_z = static_cast<double*>(MustMalloc(sizeof(double) * clusters));
  in.hi_x = static_cast<double*>(MustMalloc(sizeof(double) * clusters));
  in.hi_y = static_cast<double*>(MustMalloc(sizeof(double) * clusters));
  in.hi_z = static_cast<double*>(MustMalloc(sizeof(double) * clusters));
  in.pair_count = static_cast<int*>(MustMalloc(sizeof(int) * clusters));
  in.ci_start = static_cast<int*>(MustMalloc(sizeof(int) * (clusters + 1)));
  in.cj = static_cast<int*>(MustMalloc(sizeof(int) * pair_cap));
  in.shift = static_cast<int*>(MustMalloc(sizeof(int) * pair_cap));
  in.live = static_cast<int*>(MustMalloc(sizeof(int)));
  size_t bytes_z = 0;
  size_t bytes_c = 0;
  TILEMD_REBUILD_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes_z, in.zkey, in.zkey_out,
                                                       in.order_in, in.order_z, n));
  TILEMD_REBUILD_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes_c, in.colkey, in.colkey_out,
                                                       in.order_z, in.order_out, n));
  in.cub_bytes = std::max(bytes_z, bytes_c);
  in.cub_temp = MustMalloc(in.cub_bytes);
  TILEMD_REBUILD_CHECK(
      cudaMemcpy(const_cast<double*>(in.x), x, sizeof(double) * n, cudaMemcpyHostToDevice));
  TILEMD_REBUILD_CHECK(
      cudaMemcpy(const_cast<double*>(in.y), y, sizeof(double) * n, cudaMemcpyHostToDevice));
  TILEMD_REBUILD_CHECK(
      cudaMemcpy(const_cast<double*>(in.z), z, sizeof(double) * n, cudaMemcpyHostToDevice));
  TILEMD_REBUILD_CHECK(
      cudaMemcpy(const_cast<int*>(in.type), type, sizeof(int) * n, cudaMemcpyHostToDevice));
  TILEMD_REBUILD_CHECK(cudaMemset(in.status, 0, sizeof(DeviceStatus)));
  TILEMD_REBUILD_CHECK(cudaMemset(in.live, 0, sizeof(int)));
  if (!QueueDeviceTileRebuild(in, nullptr)) {
    throw std::runtime_error("device cluster rebuild failed");
  }
  TILEMD_REBUILD_CHECK(cudaDeviceSynchronize());
  DeviceStatus status{};
  TILEMD_REBUILD_CHECK(cudaMemcpy(&status, in.status, sizeof(status), cudaMemcpyDeviceToHost));
  TILEMD_REBUILD_CHECK(cudaMemcpy(&out.ncluster, in.live, sizeof(int), cudaMemcpyDeviceToHost));
  if (status.overflow) throw std::runtime_error("device cluster list overflowed");
  const int nslot = out.ncluster * kClusterSize;
  out.type.resize(static_cast<size_t>(nslot));
  out.gid.resize(static_cast<size_t>(nslot));
  out.ci_start.resize(static_cast<size_t>(out.ncluster) + 1);
  if (nslot > 0) {
    TILEMD_REBUILD_CHECK(
        cudaMemcpy(out.type.data(), in.slot_type, sizeof(int) * nslot, cudaMemcpyDeviceToHost));
    TILEMD_REBUILD_CHECK(
        cudaMemcpy(out.gid.data(), in.slot_gid, sizeof(int) * nslot, cudaMemcpyDeviceToHost));
  }
  TILEMD_REBUILD_CHECK(cudaMemcpy(out.ci_start.data(), in.ci_start,
                                  sizeof(int) * out.ci_start.size(), cudaMemcpyDeviceToHost));
  const int npair = out.ci_start.back();
  out.cj.resize(static_cast<size_t>(npair));
  out.shift.resize(static_cast<size_t>(npair));
  if (npair > 0) {
    TILEMD_REBUILD_CHECK(
        cudaMemcpy(out.cj.data(), in.cj, sizeof(int) * npair, cudaMemcpyDeviceToHost));
    TILEMD_REBUILD_CHECK(
        cudaMemcpy(out.shift.data(), in.shift, sizeof(int) * npair, cudaMemcpyDeviceToHost));
  }
  cudaFree(const_cast<double*>(in.x));
  cudaFree(const_cast<double*>(in.y));
  cudaFree(const_cast<double*>(in.z));
  cudaFree(const_cast<int*>(in.type));
  cudaFree(const_cast<float4*>(in.pos));
  cudaFree(in.xref);
  cudaFree(in.status);
  cudaFree(in.column_count);
  cudaFree(in.column_cluster);
  cudaFree(in.column_ncluster);
  cudaFree(in.cluster_column);
  cudaFree(in.zkey);
  cudaFree(in.zkey_out);
  cudaFree(in.column_of);
  cudaFree(in.colkey);
  cudaFree(in.colkey_out);
  cudaFree(in.order_in);
  cudaFree(in.order_z);
  cudaFree(in.order_out);
  cudaFree(in.slot_type);
  cudaFree(in.slot_gid);
  cudaFree(in.lo_x);
  cudaFree(in.lo_y);
  cudaFree(in.lo_z);
  cudaFree(in.hi_x);
  cudaFree(in.hi_y);
  cudaFree(in.hi_z);
  cudaFree(in.pair_count);
  cudaFree(in.ci_start);
  cudaFree(in.cj);
  cudaFree(in.shift);
  cudaFree(in.live);
  cudaFree(in.cub_temp);
  return out;
}

}  // namespace tilemd
