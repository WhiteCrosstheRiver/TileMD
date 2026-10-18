#include "neighbor/cluster_list.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tilemd {
namespace {

struct Slot {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int type = 0;
  int gid = -1;
};

double Wrap(double x, double length) {
  double y = x - std::floor(x / length) * length;
  if (y >= length) y = 0.0;
  if (y < 0.0) y += length;
  return y;
}

int ColumnIndex(double x, double edge, int ncolumn) {
  int i = static_cast<int>(std::floor(x / edge));
  if (i < 0) i = 0;
  if (i >= ncolumn) i = ncolumn - 1;
  return i;
}

void ChooseColumns(int n, const double* box, int requested_nx, int requested_ny, int* nx, int* ny) {
  if (requested_nx > 0 && requested_ny > 0) {
    *nx = requested_nx;
    *ny = requested_ny;
    return;
  }
  const double columns = std::max(1.0, static_cast<double>(n) / 32.0);
  const double aspect = box[0] / box[1];
  int x = std::max(1, static_cast<int>(std::lround(std::sqrt(columns * aspect))));
  int y = std::max(1, static_cast<int>(std::lround(columns / static_cast<double>(x))));
  *nx = x;
  *ny = y;
}

double AxisGap(double a_lo, double a_hi, double b_lo, double b_hi) {
  if (a_hi < b_lo) return b_lo - a_hi;
  if (b_hi < a_lo) return a_lo - b_hi;
  return 0.0;
}

bool EmptyBox(const ClusterGrid& grid, int cluster) {
  return grid.lo_x[cluster] > grid.hi_x[cluster];
}

}  // namespace

int ShiftIndex(int sx, int sy, int sz) { return (sx + 1) + 3 * ((sy + 1) + 3 * (sz + 1)); }

ClusterGrid BuildClusterGrid(const double* box, const double* x, const double* y, const double* z,
                             const int* type, int n, double reach, int nx, int ny) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  if (reach < 0.0) throw std::invalid_argument("reach must be non-negative");
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) throw std::invalid_argument("box length must be positive");
  }
  ClusterGrid grid;
  grid.box[0] = box[0];
  grid.box[1] = box[1];
  grid.box[2] = box[2];
  grid.reach = reach;
  grid.n_real = n;
  ChooseColumns(n, box, nx, ny, &grid.ncolumn[0], &grid.ncolumn[1]);
  grid.edge[0] = box[0] / static_cast<double>(grid.ncolumn[0]);
  grid.edge[1] = box[1] / static_cast<double>(grid.ncolumn[1]);

  const int ncolumn = grid.ncolumn[0] * grid.ncolumn[1];
  std::vector<std::vector<Slot>> columns(static_cast<size_t>(ncolumn));
  for (int i = 0; i < n; ++i) {
    Slot slot;
    slot.x = Wrap(x[i], box[0]);
    slot.y = Wrap(y[i], box[1]);
    slot.z = Wrap(z[i], box[2]);
    slot.type = type[i];
    slot.gid = i;
    const int ix = ColumnIndex(slot.x, grid.edge[0], grid.ncolumn[0]);
    const int iy = ColumnIndex(slot.y, grid.edge[1], grid.ncolumn[1]);
    columns[static_cast<size_t>(ix + grid.ncolumn[0] * iy)].push_back(slot);
  }

  grid.column_cluster.assign(static_cast<size_t>(ncolumn), 0);
  grid.column_ncluster.assign(static_cast<size_t>(ncolumn), 0);
  int cluster_cursor = 0;
  for (int column = 0; column < ncolumn; ++column) {
    std::vector<Slot>& slots = columns[static_cast<size_t>(column)];
    std::stable_sort(slots.begin(), slots.end(),
                     [](const Slot& a, const Slot& b) { return a.z < b.z; });
    if (!slots.empty()) {
      while (static_cast<int>(slots.size()) % kClusterSize != 0) {
        Slot pad;
        pad.type = kMaskType;
        pad.gid = -1;
        slots.push_back(pad);
      }
    }
    grid.column_cluster[static_cast<size_t>(column)] = cluster_cursor;
    const int ncluster = static_cast<int>(slots.size()) / kClusterSize;
    grid.column_ncluster[static_cast<size_t>(column)] = ncluster;
    cluster_cursor += ncluster;
    for (const Slot& slot : slots) {
      grid.x.push_back(slot.x);
      grid.y.push_back(slot.y);
      grid.z.push_back(slot.z);
      grid.type.push_back(slot.type);
      grid.gid.push_back(slot.gid);
    }
  }
  grid.ncluster = cluster_cursor;
  grid.lo_x.assign(static_cast<size_t>(grid.ncluster), 1.0);
  grid.lo_y.assign(static_cast<size_t>(grid.ncluster), 1.0);
  grid.lo_z.assign(static_cast<size_t>(grid.ncluster), 1.0);
  grid.hi_x.assign(static_cast<size_t>(grid.ncluster), 0.0);
  grid.hi_y.assign(static_cast<size_t>(grid.ncluster), 0.0);
  grid.hi_z.assign(static_cast<size_t>(grid.ncluster), 0.0);
  for (int cluster = 0; cluster < grid.ncluster; ++cluster) {
    bool any = false;
    double lo_x = 0.0;
    double lo_y = 0.0;
    double lo_z = 0.0;
    double hi_x = 0.0;
    double hi_y = 0.0;
    double hi_z = 0.0;
    for (int local = 0; local < kClusterSize; ++local) {
      const int slot = cluster * kClusterSize + local;
      if (grid.type[static_cast<size_t>(slot)] == kMaskType) continue;
      const double sx = grid.x[static_cast<size_t>(slot)];
      const double sy = grid.y[static_cast<size_t>(slot)];
      const double sz = grid.z[static_cast<size_t>(slot)];
      if (!any) {
        lo_x = hi_x = sx;
        lo_y = hi_y = sy;
        lo_z = hi_z = sz;
        any = true;
      } else {
        lo_x = std::min(lo_x, sx);
        lo_y = std::min(lo_y, sy);
        lo_z = std::min(lo_z, sz);
        hi_x = std::max(hi_x, sx);
        hi_y = std::max(hi_y, sy);
        hi_z = std::max(hi_z, sz);
      }
    }
    if (any) {
      grid.lo_x[static_cast<size_t>(cluster)] = lo_x;
      grid.lo_y[static_cast<size_t>(cluster)] = lo_y;
      grid.lo_z[static_cast<size_t>(cluster)] = lo_z;
      grid.hi_x[static_cast<size_t>(cluster)] = hi_x;
      grid.hi_y[static_cast<size_t>(cluster)] = hi_y;
      grid.hi_z[static_cast<size_t>(cluster)] = hi_z;
    }
  }
  return grid;
}

ClusterPairs BuildClusterPairs(const ClusterGrid& grid) {
  ClusterPairs pairs;
  pairs.ci_start.assign(static_cast<size_t>(grid.ncluster) + 1, 0);
  const int nx = grid.ncolumn[0];
  const int ny = grid.ncolumn[1];
  const int radius_x = std::min(nx, static_cast<int>(std::ceil(grid.reach / grid.edge[0])));
  const int radius_y = std::min(ny, static_cast<int>(std::ceil(grid.reach / grid.edge[1])));
  const double reach2 = grid.reach * grid.reach;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      const int icolumn = ix + nx * iy;
      const int i_begin = grid.column_cluster[static_cast<size_t>(icolumn)];
      const int i_count = grid.column_ncluster[static_cast<size_t>(icolumn)];
      for (int local_i = 0; local_i < i_count; ++local_i) {
        const int ci = i_begin + local_i;
        if (EmptyBox(grid, ci)) {
          pairs.ci_start[static_cast<size_t>(ci) + 1] = static_cast<int>(pairs.cj.size());
          continue;
        }
        for (int dy = -radius_y; dy <= radius_y; ++dy) {
          for (int dx = -radius_x; dx <= radius_x; ++dx) {
            int jx = ix + dx;
            int sx = 0;
            if (jx >= nx) {
              jx -= nx;
              sx = 1;
            } else if (jx < 0) {
              jx += nx;
              sx = -1;
            }
            int jy = iy + dy;
            int sy = 0;
            if (jy >= ny) {
              jy -= ny;
              sy = 1;
            } else if (jy < 0) {
              jy += ny;
              sy = -1;
            }
            const int jcolumn = jx + nx * jy;
            const int j_begin = grid.column_cluster[static_cast<size_t>(jcolumn)];
            const int j_count = grid.column_ncluster[static_cast<size_t>(jcolumn)];
            for (int local_j = 0; local_j < j_count; ++local_j) {
              const int cj = j_begin + local_j;
              ++pairs.examined;
              if (EmptyBox(grid, cj)) continue;
              for (int sz = -1; sz <= 1; ++sz) {
                const double j_lo_z = grid.lo_z[static_cast<size_t>(cj)] + sz * grid.box[2];
                const double j_hi_z = grid.hi_z[static_cast<size_t>(cj)] + sz * grid.box[2];
                if (grid.hi_z[static_cast<size_t>(ci)] + grid.reach < j_lo_z) continue;
                if (j_hi_z + grid.reach < grid.lo_z[static_cast<size_t>(ci)]) continue;
                const double gap_x =
                    AxisGap(grid.lo_x[static_cast<size_t>(ci)], grid.hi_x[static_cast<size_t>(ci)],
                            grid.lo_x[static_cast<size_t>(cj)] + sx * grid.box[0],
                            grid.hi_x[static_cast<size_t>(cj)] + sx * grid.box[0]);
                const double gap_y =
                    AxisGap(grid.lo_y[static_cast<size_t>(ci)], grid.hi_y[static_cast<size_t>(ci)],
                            grid.lo_y[static_cast<size_t>(cj)] + sy * grid.box[1],
                            grid.hi_y[static_cast<size_t>(cj)] + sy * grid.box[1]);
                const double gap_z = AxisGap(grid.lo_z[static_cast<size_t>(ci)],
                                             grid.hi_z[static_cast<size_t>(ci)], j_lo_z, j_hi_z);
                if (gap_x * gap_x + gap_y * gap_y + gap_z * gap_z < reach2) {
                  pairs.cj.push_back(cj);
                  pairs.shift.push_back(ShiftIndex(sx, sy, sz));
                  ++pairs.kept;
                }
              }
            }
          }
        }
        pairs.ci_start[static_cast<size_t>(ci) + 1] = static_cast<int>(pairs.cj.size());
      }
    }
  }
  return pairs;
}

}  // namespace tilemd
