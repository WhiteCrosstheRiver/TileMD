#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lj_reference.hpp"

namespace tilemd {

// One stage-1 run described by the Appendix D key-value file. Positions come
// from `box fcc` or `structure`. Velocities are a zero-momentum Gaussian at
// the requested temperature when `velocity` is present, and zero otherwise.
struct SimulationInput {
  ref::System system;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  double mass = 1.0;
  double skin = 0.3;
  double timestep = 0.005;
  bool langevin = false;
  ref::Langevin thermo;
  int thermo_every = 0;
  int dump_every = 0;
  int graph_unroll = 50;
  int steps = 0;
};

// `base_dir` resolves a relative `structure` path. `text` is the file body.
SimulationInput ParseInput(const std::string& text, const std::string& base_dir);

SimulationInput ParseInputFile(const std::string& path);

}  // namespace tilemd
