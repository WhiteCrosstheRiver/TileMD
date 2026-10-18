#pragma once

#include <vector>

namespace tilemd {

// A fused step that begins at integer time `step` can see that time's force.
// The positions after `steps` integrations are one step ahead of the last
// kernel, so the final multiple is recorded by the extra force evaluation.
inline bool TileRecordsThermo(int step, int every) { return every > 0 && step % every == 0; }

inline std::vector<int> TileThermoSteps(int steps, int every) {
  std::vector<int> out;
  if (every <= 0 || steps < 0) return out;
  for (int step = 0; step < steps; ++step) {
    if (TileRecordsThermo(step, every)) out.push_back(step);
  }
  if (TileRecordsThermo(steps, every)) out.push_back(steps);
  return out;
}

}  // namespace tilemd
