#pragma once

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

}  // namespace tilemd
