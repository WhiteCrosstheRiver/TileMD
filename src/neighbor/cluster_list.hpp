#pragma once

#include <vector>

namespace tilemd {

// One cluster is 8 consecutive slots in a column. Padding slots use this type
// and are left out of the bounding box. The force loop masks them by type.
constexpr int kClusterSize = 8;
constexpr int kMaskType = -1;

// Columns in x–y. Atoms in a column are stable-sorted by z and padded so the
// column length is a multiple of 8. `nx` and `ny` of 0 choose a grid whose
// mean occupancy is about 32 atoms.
struct ClusterGrid {
  double box[3] = {};
  double edge[2] = {};
  int ncolumn[2] = {};
  double reach = 0.0;
  int n_real = 0;
  int ncluster = 0;
  // Slot order. `gid` is the input index, or -1 for a padding slot.
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<int> type;
  std::vector<int> gid;
  // Column (ix + nx * iy) begins at this cluster and owns this many clusters.
  std::vector<int> column_cluster;
  std::vector<int> column_ncluster;
  // Real atoms only. An empty cluster has lo > hi on every axis.
  std::vector<double> lo_x;
  std::vector<double> lo_y;
  std::vector<double> lo_z;
  std::vector<double> hi_x;
  std::vector<double> hi_y;
  std::vector<double> hi_z;
};

// CSR of cluster pairs. `shift` is sx + 1 + 3 * (sy + 1 + 3 * (sz + 1)), each
// component in {-1, 0, 1}.
struct ClusterPairs {
  std::vector<int> ci_start;
  std::vector<int> cj;
  std::vector<int> shift;
  int examined = 0;
  int kept = 0;
};

int ShiftIndex(int sx, int sy, int sz);

ClusterGrid BuildClusterGrid(const double* box, const double* x, const double* y, const double* z,
                             const int* type, int n, double reach, int nx = 0, int ny = 0);

// A j-cluster is kept when the periodic bounding-box distance is below `reach`.
ClusterPairs BuildClusterPairs(const ClusterGrid& grid);

enum class LjShift { None, Energy, Force };

struct LjPair {
  int type_i = 0;
  int type_j = 0;
  double epsilon = 1.0;
  double sigma = 1.0;
  double cutoff = 2.5;
};

// Full-list Lennard-Jones on the cluster pairs. Forces are stored in the input
// atom order. `skin` widens the list; the pair cutoff is unchanged.
struct TileForces {
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
  double potential_energy = 0.0;
  double virial[6] = {};
  int pairs_visited = 0;
  int pairs_inside = 0;
};

// `nx` and `ny` of 0 use the default column grid, about 32 atoms per column.
TileForces ComputeTileForces(const double* box, const double* x, const double* y, const double* z,
                             const int* type, int n, const LjPair* pairs, int npair, double skin,
                             LjShift shift, int nx = 0, int ny = 0);

}  // namespace tilemd
