#include "neighbor/cluster_list.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tilemd {
namespace {

struct Slot {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int type = 0;
  int gid = -1;
};

}  // namespace

ClusterShape ChooseClusterColumns(int n, const double* box, int nx, int ny) {
  ClusterShape shape;
  if (nx > 0 && ny > 0) {
    shape.nx = nx;
    shape.ny = ny;
  } else {
    const double columns = std::max(1.0, static_cast<double>(n) / 32.0);
    const double aspect = box[0] / box[1];
    shape.nx = std::max(1, static_cast<int>(std::lround(std::sqrt(columns * aspect))));
    shape.ny = std::max(1, static_cast<int>(std::lround(columns / static_cast<double>(shape.nx))));
  }
  shape.edge_x = box[0] / static_cast<double>(shape.nx);
  shape.edge_y = box[1] / static_cast<double>(shape.ny);
  return shape;
}

ClusterSlots SlotsInDeviceOrder(const double* box, const double* x, const double* y,
                                const double* z, const int* type, int n, int nx, int ny) {
  ClusterSlots out;
  if (n <= 0) return out;
  const ClusterShape shape = ChooseClusterColumns(n, box, nx, ny);
  struct Item {
    unsigned long long zkey = 0;
    int column = 0;
    int index = 0;
  };
  std::vector<Item> items(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double wx = WrapCoord(x[i], box[0]);
    const double wy = WrapCoord(y[i], box[1]);
    const double wz = WrapCoord(z[i], box[2]);
    items[static_cast<size_t>(i)].zkey = OrderedZ(wz);
    const int ix = ColumnIndex(wx, shape.edge_x, shape.nx);
    const int iy = ColumnIndex(wy, shape.edge_y, shape.ny);
    items[static_cast<size_t>(i)].column = ix + shape.nx * iy;
    items[static_cast<size_t>(i)].index = i;
  }
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) { return a.zkey < b.zkey; });
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) { return a.column < b.column; });
  const int ncolumn = shape.nx * shape.ny;
  std::vector<int> counts(static_cast<size_t>(ncolumn), 0);
  for (const Item& item : items) ++counts[static_cast<size_t>(item.column)];
  int sorted_at = 0;
  int cluster_cursor = 0;
  for (int column = 0; column < ncolumn; ++column) {
    const int count = counts[static_cast<size_t>(column)];
    const int nclus = count == 0 ? 0 : (count + kClusterSize - 1) / kClusterSize;
    for (int a = 0; a < count; ++a) {
      const int atom = items[static_cast<size_t>(sorted_at++)].index;
      out.type.push_back(type[atom]);
      out.gid.push_back(atom);
    }
    for (int pad = count; pad < nclus * kClusterSize; ++pad) {
      out.type.push_back(kMaskType);
      out.gid.push_back(-1);
    }
    cluster_cursor += nclus;
  }
  out.ncluster = cluster_cursor;
  return out;
}

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
  const ClusterShape shape = ChooseClusterColumns(n, box, nx, ny);
  grid.ncolumn[0] = shape.nx;
  grid.ncolumn[1] = shape.ny;
  grid.edge[0] = shape.edge_x;
  grid.edge[1] = shape.edge_y;

  const int ncolumn = grid.ncolumn[0] * grid.ncolumn[1];
  std::vector<std::vector<Slot>> columns(static_cast<size_t>(ncolumn));
  for (int i = 0; i < n; ++i) {
    Slot slot;
    slot.x = WrapCoord(x[i], box[0]);
    slot.y = WrapCoord(y[i], box[1]);
    slot.z = WrapCoord(z[i], box[2]);
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
    ClusterBox box;
    bool any = false;
    for (int local = 0; local < kClusterSize; ++local) {
      const int slot = cluster * kClusterSize + local;
      if (grid.type[static_cast<size_t>(slot)] == kMaskType) continue;
      IncludeInBox(&box, &any, grid.x[static_cast<size_t>(slot)], grid.y[static_cast<size_t>(slot)],
                   grid.z[static_cast<size_t>(slot)]);
    }
    grid.lo_x[static_cast<size_t>(cluster)] = box.lo_x;
    grid.lo_y[static_cast<size_t>(cluster)] = box.lo_y;
    grid.lo_z[static_cast<size_t>(cluster)] = box.lo_z;
    grid.hi_x[static_cast<size_t>(cluster)] = box.hi_x;
    grid.hi_y[static_cast<size_t>(cluster)] = box.hi_y;
    grid.hi_z[static_cast<size_t>(cluster)] = box.hi_z;
  }
  return grid;
}

ClusterPairs BuildClusterPairs(const ClusterGrid& grid) {
  ClusterPairs pairs;
  pairs.ci_start.assign(static_cast<size_t>(grid.ncluster) + 1, 0);
  const int nx = grid.ncolumn[0];
  const int ny = grid.ncolumn[1];
  // Column index is ix + nx * iy, and clusters were numbered in that same
  // linear order. The pair rows have to be appended in cluster order so each
  // CSR range stays contiguous.
  std::vector<int> cluster_column(static_cast<size_t>(grid.ncluster), 0);
  for (int column = 0; column < nx * ny; ++column) {
    const int begin = grid.column_cluster[static_cast<size_t>(column)];
    const int count = grid.column_ncluster[static_cast<size_t>(column)];
    for (int local = 0; local < count; ++local) {
      cluster_column[static_cast<size_t>(begin + local)] = column;
    }
  }
  ClusterPairView view;
  view.lo_x = grid.lo_x.data();
  view.lo_y = grid.lo_y.data();
  view.lo_z = grid.lo_z.data();
  view.hi_x = grid.hi_x.data();
  view.hi_y = grid.hi_y.data();
  view.hi_z = grid.hi_z.data();
  view.column_cluster = grid.column_cluster.data();
  view.column_ncluster = grid.column_ncluster.data();
  view.cluster_column = cluster_column.data();
  view.nx = nx;
  view.ny = ny;
  view.reach = grid.reach;
  view.edge_x = grid.edge[0];
  view.edge_y = grid.edge[1];
  view.box_x = grid.box[0];
  view.box_y = grid.box[1];
  view.box_z = grid.box[2];
  for (int icolumn = 0; icolumn < nx * ny; ++icolumn) {
    const int i_begin = grid.column_cluster[static_cast<size_t>(icolumn)];
    const int i_count = grid.column_ncluster[static_cast<size_t>(icolumn)];
    for (int local_i = 0; local_i < i_count; ++local_i) {
      const int ci = i_begin + local_i;
      pairs.ci_start[static_cast<size_t>(ci)] = static_cast<int>(pairs.cj.size());
      VisitClusterPairs(
          view, ci,
          [&](int cj, int shift) {
            pairs.cj.push_back(cj);
            pairs.shift.push_back(shift);
            ++pairs.kept;
          },
          &pairs.examined);
    }
  }
  pairs.ci_start[static_cast<size_t>(grid.ncluster)] = static_cast<int>(pairs.cj.size());
  return pairs;
}

namespace {

constexpr int kMaxTypes = 8;

struct Prepared {
  bool defined = false;
  double sigma2 = 0.0;
  double eps4 = 0.0;
  double eps24 = 0.0;
  double rc = 0.0;
  double rc2 = 0.0;
  double u_shift = 0.0;
  double s_rc = 0.0;
};

void DecodeShift(int shift, int* sx, int* sy, int* sz) {
  *sx = shift % 3 - 1;
  *sy = (shift / 3) % 3 - 1;
  *sz = shift / 9 - 1;
}

// Wrapped `now` on the same branch as `ref`. Motion below half a box crosses
// at most one boundary, so the rounded quotient is that crossing.
double OnReferenceBranch(double now, double ref, double length) {
  return now + length * std::rint((ref - now) / length);
}

void FillPairs(Prepared table[kMaxTypes][kMaxTypes], const LjPair* pairs, int npair, LjShift shift,
               double* max_cut) {
  if (npair < 1) throw std::invalid_argument("at least one pair coefficient is required");
  for (int p = 0; p < npair; ++p) {
    const LjPair& coeff = pairs[p];
    if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kMaxTypes ||
        coeff.type_j >= kMaxTypes) {
      throw std::invalid_argument("pair type is outside 0..7");
    }
    if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
      throw std::invalid_argument("sigma and cutoff must be positive");
    }
    auto fill = [&](int ti, int tj) {
      Prepared& row = table[ti][tj];
      row.defined = true;
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
    *max_cut = std::max(*max_cut, coeff.cutoff);
  }
}

TileForces WalkTilePairs(const ClusterGrid& grid, const ClusterPairs& list, const double* x,
                         const double* y, const double* z, const double* xref, const double* yref,
                         const double* zref, const Prepared table[kMaxTypes][kMaxTypes],
                         LjShift shift, bool hold_reference) {
  const int n = grid.n_real;
  TileForces out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  for (int ci = 0; ci < grid.ncluster; ++ci) {
    for (int k = list.ci_start[static_cast<size_t>(ci)];
         k < list.ci_start[static_cast<size_t>(ci) + 1]; ++k) {
      const int cj = list.cj[static_cast<size_t>(k)];
      int sx = 0;
      int sy = 0;
      int sz = 0;
      DecodeShift(list.shift[static_cast<size_t>(k)], &sx, &sy, &sz);
      for (int local_i = 0; local_i < kClusterSize; ++local_i) {
        const int slot_i = ci * kClusterSize + local_i;
        if (grid.type[static_cast<size_t>(slot_i)] == kMaskType) continue;
        const int gid_i = grid.gid[static_cast<size_t>(slot_i)];
        const int ti = grid.type[static_cast<size_t>(slot_i)];
        const double xi =
            hold_reference ? OnReferenceBranch(x[gid_i], xref[gid_i], grid.box[0]) : x[gid_i];
        const double yi =
            hold_reference ? OnReferenceBranch(y[gid_i], yref[gid_i], grid.box[1]) : y[gid_i];
        const double zi =
            hold_reference ? OnReferenceBranch(z[gid_i], zref[gid_i], grid.box[2]) : z[gid_i];
        for (int local_j = 0; local_j < kClusterSize; ++local_j) {
          const int slot_j = cj * kClusterSize + local_j;
          if (grid.type[static_cast<size_t>(slot_j)] == kMaskType) continue;
          if (ci == cj && local_i == local_j && sx == 0 && sy == 0 && sz == 0) continue;
          ++out.pairs_visited;
          const int gid_j = grid.gid[static_cast<size_t>(slot_j)];
          const int tj = grid.type[static_cast<size_t>(slot_j)];
          const Prepared& row = table[ti][tj];
          if (!row.defined) throw std::invalid_argument("missing pair coefficient");
          const double xj =
              hold_reference ? OnReferenceBranch(x[gid_j], xref[gid_j], grid.box[0]) : x[gid_j];
          const double yj =
              hold_reference ? OnReferenceBranch(y[gid_j], yref[gid_j], grid.box[1]) : y[gid_j];
          const double zj =
              hold_reference ? OnReferenceBranch(z[gid_j], zref[gid_j], grid.box[2]) : z[gid_j];
          const double dx = xi - (xj + static_cast<double>(sx) * grid.box[0]);
          const double dy = yi - (yj + static_cast<double>(sy) * grid.box[1]);
          const double dz = zi - (zj + static_cast<double>(sz) * grid.box[2]);
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= row.rc2) continue;
          ++out.pairs_inside;
          const double inv_r2 = 1.0 / r2;
          const double sr2 = row.sigma2 * inv_r2;
          const double sr6 = sr2 * sr2 * sr2;
          double s = row.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
          double u = row.eps4 * sr6 * (sr6 - 1.0) - row.u_shift;
          if (shift == LjShift::Force) {
            const double r = std::sqrt(r2);
            u += (r - row.rc) * row.s_rc * row.rc;
            s -= row.s_rc * row.rc / r;
          }
          const double fx = s * dx;
          const double fy = s * dy;
          const double fz = s * dz;
          out.fx[static_cast<size_t>(gid_i)] += fx;
          out.fy[static_cast<size_t>(gid_i)] += fy;
          out.fz[static_cast<size_t>(gid_i)] += fz;
          out.potential_energy += 0.5 * u;
          out.virial[0] += 0.5 * dx * fx;
          out.virial[1] += 0.5 * dy * fy;
          out.virial[2] += 0.5 * dz * fz;
          out.virial[3] += 0.5 * dx * fy;
          out.virial[4] += 0.5 * dx * fz;
          out.virial[5] += 0.5 * dy * fz;
        }
      }
    }
  }
  return out;
}

}  // namespace

TileForces ComputeTileForces(const double* box, const double* x, const double* y, const double* z,
                             const int* type, int n, const LjPair* pairs, int npair, double skin,
                             LjShift shift, int nx, int ny) {
  if (skin < 0.0) throw std::invalid_argument("skin must be non-negative");
  Prepared table[kMaxTypes][kMaxTypes] = {};
  double max_cut = 0.0;
  FillPairs(table, pairs, npair, shift, &max_cut);

  const ClusterGrid grid = BuildClusterGrid(box, x, y, z, type, n, max_cut + skin, nx, ny);
  const ClusterPairs list = BuildClusterPairs(grid);
  std::vector<double> wrapped_x(static_cast<size_t>(n));
  std::vector<double> wrapped_y(static_cast<size_t>(n));
  std::vector<double> wrapped_z(static_cast<size_t>(n));
  for (int slot = 0; slot < static_cast<int>(grid.gid.size()); ++slot) {
    const int gid = grid.gid[static_cast<size_t>(slot)];
    if (gid < 0) continue;
    wrapped_x[static_cast<size_t>(gid)] = grid.x[static_cast<size_t>(slot)];
    wrapped_y[static_cast<size_t>(gid)] = grid.y[static_cast<size_t>(slot)];
    wrapped_z[static_cast<size_t>(gid)] = grid.z[static_cast<size_t>(slot)];
  }
  return WalkTilePairs(grid, list, wrapped_x.data(), wrapped_y.data(), wrapped_z.data(), nullptr,
                       nullptr, nullptr, table, shift, false);
}

TileForces ComputeTileForcesOnList(const ClusterGrid& grid, const ClusterPairs& list,
                                   const double* x, const double* y, const double* z,
                                   const double* xref, const double* yref, const double* zref,
                                   const LjPair* pairs, int npair, LjShift shift) {
  Prepared table[kMaxTypes][kMaxTypes] = {};
  double max_cut = 0.0;
  FillPairs(table, pairs, npair, shift, &max_cut);
  (void)max_cut;
  return WalkTilePairs(grid, list, x, y, z, xref, yref, zref, table, shift, true);
}

}  // namespace tilemd
