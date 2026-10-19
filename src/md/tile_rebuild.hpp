#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "core/device_status.hpp"

namespace tilemd {

// Buffers for one device rebuild of the cluster list. Sizes stay fixed for a
// captured graph: `cluster_cap` is the launch bound, `pair_cap` is the CSR
// capacity. A rebuild that does not fit sets `status->overflow` and a live
// cluster count of 0.
struct DeviceTileRebuild {
  const double* x = nullptr;
  const double* y = nullptr;
  const double* z = nullptr;
  const int* type = nullptr;
  float4* xref = nullptr;
  const float4* pos = nullptr;
  DeviceStatus* status = nullptr;
  int n = 0;
  int nx = 1;
  int ny = 1;
  double edge_x = 0.0;
  double edge_y = 0.0;
  double reach = 0.0;
  double box_x = 0.0;
  double box_y = 0.0;
  double box_z = 0.0;
  int cluster_cap = 0;
  int pair_cap = 0;
  int* column_count = nullptr;
  int* column_cluster = nullptr;
  int* column_ncluster = nullptr;
  int* cluster_column = nullptr;
  unsigned long long* zkey = nullptr;
  unsigned long long* zkey_out = nullptr;
  int* column_of = nullptr;
  int* colkey = nullptr;
  int* colkey_out = nullptr;
  int* order_in = nullptr;
  int* order_z = nullptr;
  int* order_out = nullptr;
  int* slot_type = nullptr;
  int* slot_gid = nullptr;
  double* lo_x = nullptr;
  double* lo_y = nullptr;
  double* lo_z = nullptr;
  double* hi_x = nullptr;
  double* hi_y = nullptr;
  double* hi_z = nullptr;
  int* pair_count = nullptr;
  int* ci_start = nullptr;
  int* cj = nullptr;
  int* shift = nullptr;
  int* live = nullptr;
  void* cub_temp = nullptr;
  size_t cub_bytes = 0;
};

// Stable column grouping, then the same pair walk as `BuildClusterPairs`.
// `cub_temp` must already be large enough. Safe to record on a capturing stream.
bool QueueDeviceTileRebuild(const DeviceTileRebuild& in, cudaStream_t stream);

struct HostClusterList {
  int ncluster = 0;
  std::vector<int> type;
  std::vector<int> gid;
  std::vector<int> ci_start;
  std::vector<int> cj;
  std::vector<int> shift;
};

// Allocates, rebuilds, and downloads. Used by the GPU list test.
HostClusterList RebuildClusterListOnDevice(const double* box, const double* x, const double* y,
                                           const double* z, const int* type, int n, double reach);

}  // namespace tilemd
