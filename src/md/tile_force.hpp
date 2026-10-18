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

// One cluster's contribution on a thermo step. The host sums these after the
// fused kernel. Kinetic energy uses the integer-time velocity.
struct TileThermoPartial {
  double ke = 0.0;
  double pe = 0.0;
  double wxx = 0.0;
  double wyy = 0.0;
  double wzz = 0.0;
  double wxy = 0.0;
  double wxz = 0.0;
  double wyz = 0.0;
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
// `record_step < 0` skips energy, virial, and the partial write. Otherwise that
// step index is the integer time of the positions the force was computed from,
// and each cluster writes one partial.
// Pair rows and the box. Called before a captured graph; the kernel itself
// does not copy them and does not synchronize.
void UploadFusedTileSymbols(const double* box, const LjPair* pairs, int npair, LjShift shift);

// One fused kernel on `stream`. Symbols must already be uploaded. No host sync.
void LaunchFusedTileKernel(int ncluster, const int* type, const int* gid, const int* ci_start,
                           const int* cj, const int* image, int* missing, double* x, double* y,
                           double* z, double* vx, double* vy, double* vz, int* imgx, int* imgy,
                           int* imgz, float4* pos_cur, float4* pos_next, float4* xref, void* status,
                           int* step_counter, double dt, double inv_mass,
                           TileThermoPartial* partial, int record_step, cudaStream_t stream,
                           const int* record_ptr, const int* live_clusters, int launch_blocks);

void LaunchFusedTileDevice(int ncluster, const int* type, const int* gid, const int* ci_start,
                           const int* cj, const int* image, int* missing, const double* box,
                           const LjPair* pairs, int npair, LjShift shift, double* x, double* y,
                           double* z, double* vx, double* vy, double* vz, int* imgx, int* imgy,
                           int* imgz, float4* pos_cur, float4* pos_next, float4* xref, void* status,
                           int* step_counter, double dt, double inv_mass,
                           TileThermoPartial* partial, int record_step);

}  // namespace tilemd
