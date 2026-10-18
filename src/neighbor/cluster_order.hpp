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

TILEMD_HD inline double AxisGap(double a_lo, double a_hi, double b_lo, double b_hi) {
  if (a_hi < b_lo) return b_lo - a_hi;
  if (b_hi < a_lo) return a_lo - b_hi;
  return 0.0;
}

}  // namespace tilemd
