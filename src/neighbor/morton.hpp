#pragma once

#ifndef TILEMD_HD
#if defined(__CUDACC__)
#define TILEMD_HD __host__ __device__
#else
#define TILEMD_HD
#endif
#endif

namespace tilemd {

// 10-bit Morton code. Stage-1 boxes stay under 1024 cells per axis.
TILEMD_HD inline unsigned Spread10(unsigned v) {
  v &= 0x3ffu;
  v = (v | (v << 16)) & 0x030000FFu;
  v = (v | (v << 8)) & 0x0300F00Fu;
  v = (v | (v << 4)) & 0x030C30C3u;
  v = (v | (v << 2)) & 0x09249249u;
  return v;
}

TILEMD_HD inline int MortonCell(int ix, int iy, int iz) {
  unsigned code = Spread10(static_cast<unsigned>(ix)) | (Spread10(static_cast<unsigned>(iy)) << 1) |
                  (Spread10(static_cast<unsigned>(iz)) << 2);
  return static_cast<int>(code);
}

}  // namespace tilemd
