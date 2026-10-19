#pragma once

#include <cmath>

#include "neighbor/cluster_order.hpp"

namespace tilemd {

// Minimum-image separation along one axis from D18 shadow coordinates.
// The box is `nfull` full cells plus `remainder` (both derived from the
// double box length). The three images are built with integer cell shifts, so
// a neighbor's cell delta stays small and the float error is about one ulp
// of the cell edge. A wrong carry cancels and does not change the result.
TILEMD_HD inline float ShadowMinImage(int cell_i, float local_i, int cell_j, float local_j,
                                      int nfull, float remainder) {
  constexpr float kEdge = 4.f;
  float best = 0.f;
  float best_abs = 1.0e30f;
  for (int image = -1; image <= 1; ++image) {
    int cell = cell_j;
    float local = local_j;
    if (image > 0) {
      float shifted = local_j + remainder;
      int carry = 0;
      if (shifted >= kEdge) {
        shifted -= kEdge;
        carry = 1;
      }
      local = shifted;
      cell = cell_j + nfull + carry;
    } else if (image < 0) {
      float shifted = local_j - remainder;
      int borrow = 0;
      if (shifted < 0.f) {
        shifted += kEdge;
        borrow = 1;
      }
      local = shifted;
      cell = cell_j - nfull - borrow;
    }
    const float delta = static_cast<float>(cell_i - cell) * kEdge + (local_i - local);
    const float mag = fabsf(delta);
    if (mag < best_abs) {
      best_abs = mag;
      best = delta;
    }
  }
  return best;
}

}  // namespace tilemd
