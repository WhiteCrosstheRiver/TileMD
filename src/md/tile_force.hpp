#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "neighbor/cluster_list.hpp"

namespace tilemd {

struct TileDeviceForces {
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
  double potential_energy = 0.0;
  double virial[6] = {};
};

// One warp per i-cluster. Lane lane = i_local + 8 * j_sub, with i_local in 0..7
// and j_sub in 0..3. The host builds the cluster list; the kernel applies the
// stored image shift and writes one double accumulator per real atom.
TileDeviceForces RunTileForceGpu(const ClusterGrid& grid, const ClusterPairs& list,
                                 const LjPair* pairs, int npair, LjShift shift);

// One leapfrog step. Pair coordinates are read from `pos_cur`. The owning lane
// updates that atom's double position and velocity and writes `pos_next`.
void RunFusedTileLeapfrog(const ClusterGrid& grid, const ClusterPairs& list, const LjPair* pairs,
                          int npair, LjShift shift, double* x, double* y, double* z, double* vx,
                          double* vy, double* vz, int* imgx, int* imgy, int* imgz, float4* pos_cur,
                          float4* pos_next, float4* xref, void* status, int* step_counter,
                          double dt, double inv_mass);

// The cluster list is already on the device. Pair coordinates still come from pos_cur.
void LaunchFusedTileDevice(int ncluster, const int* type, const int* gid, const int* ci_start,
                           const int* cj, const int* image, int* missing, const double* box,
                           const LjPair* pairs, int npair, LjShift shift, double* x, double* y,
                           double* z, double* vx, double* vy, double* vz, int* imgx, int* imgy,
                           int* imgz, float4* pos_cur, float4* pos_next, float4* xref, void* status,
                           int* step_counter, double dt, double inv_mass);

}  // namespace tilemd
