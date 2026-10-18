#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define TILEMD_HD __host__ __device__
#else
#define TILEMD_HD
#endif

namespace tilemd {

// True when the farthest atom has moved past half the skin. Equality does not rebuild.
TILEMD_HD inline bool DisplacementNeedsRebuild(float disp2, float trigger2) {
  return disp2 > trigger2;
}

// Shared by the host cluster builder and the device rebuild. The device sort
// keys z with OrderedZ after wrapping, then groups by column. std::stable_sort
// on that key matches a stable radix sort.

TILEMD_HD inline double WrapCoord(double x, double length) {
#if defined(__CUDA_ARCH__)
  double y = x - ::floor(x / length) * length;
#else
  double y = x - std::floor(x / length) * length;
#endif
  if (y >= length) y = 0.0;
  if (y < 0.0) y += length;
  return y;
}

TILEMD_HD inline int ColumnIndex(double x, double edge, int ncolumn) {
#if defined(__CUDA_ARCH__)
  int i = static_cast<int>(::floor(x / edge));
#else
  int i = static_cast<int>(std::floor(x / edge));
#endif
  if (i < 0) i = 0;
  if (i >= ncolumn) i = ncolumn - 1;
  return i;
}

TILEMD_HD inline unsigned long long OrderedZ(double z) {
  if (z == 0.0) z = 0.0;
  unsigned long long bits = 0;
#if defined(__CUDA_ARCH__)
  bits = static_cast<unsigned long long>(__double_as_longlong(z));
#else
  std::memcpy(&bits, &z, sizeof(bits));
#endif
  const unsigned long long mask = (bits >> 63) ? ~0ull : (1ull << 63);
  return bits ^ mask;
}

TILEMD_HD inline int ShiftIndex(int sx, int sy, int sz) {
  return (sx + 1) + 3 * ((sy + 1) + 3 * (sz + 1));
}

// Empty clusters keep lo > hi. A wrapped coordinate is included only after `WrapCoord`.
struct ClusterBox {
  double lo_x = 1.0;
  double lo_y = 1.0;
  double lo_z = 1.0;
  double hi_x = 0.0;
  double hi_y = 0.0;
  double hi_z = 0.0;
};

TILEMD_HD inline void IncludeInBox(ClusterBox* box, bool* any, double x, double y, double z) {
  if (!*any) {
    box->lo_x = box->hi_x = x;
    box->lo_y = box->hi_y = y;
    box->lo_z = box->hi_z = z;
    *any = true;
    return;
  }
  if (x < box->lo_x) box->lo_x = x;
  if (y < box->lo_y) box->lo_y = y;
  if (z < box->lo_z) box->lo_z = z;
  if (x > box->hi_x) box->hi_x = x;
  if (y > box->hi_y) box->hi_y = y;
  if (z > box->hi_z) box->hi_z = z;
}

TILEMD_HD inline double AxisGap(double a_lo, double a_hi, double b_lo, double b_hi) {
  if (a_hi < b_lo) return b_lo - a_hi;
  if (b_hi < a_lo) return a_lo - b_hi;
  return 0.0;
}

TILEMD_HD inline int ColumnRadius(int ncolumn, double reach, double edge) {
#if defined(__CUDA_ARCH__)
  int radius = static_cast<int>(::ceil(reach / edge));
#else
  int radius = static_cast<int>(std::ceil(reach / edge));
#endif
  if (radius > ncolumn) radius = ncolumn;
  if (radius < 0) radius = 0;
  return radius;
}

// Bounding boxes and column ranges for one cluster-pair walk. `emit(cj, shift)`
// is called in the same order the host list appends pairs.
struct ClusterPairView {
  const double* lo_x = nullptr;
  const double* lo_y = nullptr;
  const double* lo_z = nullptr;
  const double* hi_x = nullptr;
  const double* hi_y = nullptr;
  const double* hi_z = nullptr;
  const int* column_cluster = nullptr;
  const int* column_ncluster = nullptr;
  const int* cluster_column = nullptr;
  int nx = 1;
  int ny = 1;
  double reach = 0.0;
  double edge_x = 0.0;
  double edge_y = 0.0;
  double box_x = 0.0;
  double box_y = 0.0;
  double box_z = 0.0;
};

template <typename Emit>
TILEMD_HD void VisitClusterPairs(const ClusterPairView& view, int ci, Emit emit, int* examined) {
  if (view.lo_x[ci] > view.hi_x[ci]) return;
  const int column = view.cluster_column[ci];
  const int ix = column % view.nx;
  const int iy = column / view.nx;
  const int radius_x = ColumnRadius(view.nx, view.reach, view.edge_x);
  const int radius_y = ColumnRadius(view.ny, view.reach, view.edge_y);
  const double reach2 = view.reach * view.reach;
  for (int dy = -radius_y; dy <= radius_y; ++dy) {
    for (int dx = -radius_x; dx <= radius_x; ++dx) {
      int jx = ix + dx;
      int sx = 0;
      if (jx >= view.nx) {
        jx -= view.nx;
        sx = 1;
      } else if (jx < 0) {
        jx += view.nx;
        sx = -1;
      }
      int jy = iy + dy;
      int sy = 0;
      if (jy >= view.ny) {
        jy -= view.ny;
        sy = 1;
      } else if (jy < 0) {
        jy += view.ny;
        sy = -1;
      }
      const int jcolumn = jx + view.nx * jy;
      const int j_begin = view.column_cluster[jcolumn];
      const int j_count = view.column_ncluster[jcolumn];
      for (int local_j = 0; local_j < j_count; ++local_j) {
        const int cj = j_begin + local_j;
        if (examined != nullptr) ++*examined;
        if (view.lo_x[cj] > view.hi_x[cj]) continue;
        for (int sz = -1; sz <= 1; ++sz) {
          const double j_lo_z = view.lo_z[cj] + sz * view.box_z;
          const double j_hi_z = view.hi_z[cj] + sz * view.box_z;
          if (view.hi_z[ci] + view.reach < j_lo_z) continue;
          if (j_hi_z + view.reach < view.lo_z[ci]) continue;
          const double gap_x =
              AxisGap(view.lo_x[ci], view.hi_x[ci], view.lo_x[cj] + sx * view.box_x,
                      view.hi_x[cj] + sx * view.box_x);
          const double gap_y =
              AxisGap(view.lo_y[ci], view.hi_y[ci], view.lo_y[cj] + sy * view.box_y,
                      view.hi_y[cj] + sy * view.box_y);
          const double gap_z = AxisGap(view.lo_z[ci], view.hi_z[ci], j_lo_z, j_hi_z);
          if (gap_x * gap_x + gap_y * gap_y + gap_z * gap_z < reach2) {
            emit(cj, ShiftIndex(sx, sy, sz));
          }
        }
      }
    }
  }
}

}  // namespace tilemd
