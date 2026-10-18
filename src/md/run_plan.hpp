#pragma once

#include <algorithm>
#include <stdexcept>

namespace tilemd {

// How many captured graphs and leftover single steps cover one segment.
struct SegmentLaunch {
  int graphs = 0;
  int tail = 0;
};

// K in the captured graph. A run shorter than the request captures only that
// many steps. The plan's default request is 50.
inline int GraphUnroll(int requested, int steps) {
  if (requested < 1) throw std::invalid_argument("graph_unroll must be positive");
  if (steps < 1) return 1;
  return std::min(requested, steps);
}

inline SegmentLaunch SegmentSteps(int segment, int unroll) {
  if (unroll < 1) throw std::invalid_argument("graph_unroll must be positive");
  if (segment < 0) throw std::invalid_argument("segment length must be non-negative");
  SegmentLaunch out;
  out.graphs = segment / unroll;
  out.tail = segment % unroll;
  return out;
}

}  // namespace tilemd
