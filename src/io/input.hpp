#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lj_reference.hpp"

namespace tilemd {

// One stage-1 run described by the Appendix D key-value file. Positions come
// from `box fcc` or `structure`. `velocity` draws a zero-momentum Gaussian.
// Lennard-Jones uses variance `T/m`. Metal units use `k_B T / (m mvv2e)` and,
// for a setfl file, each element's mass.
// Without that line, velocities come from the structure file when it has a
// vel column, and are zero otherwise. `eam` names a funcfl or setfl file and
// selects metal units. `mass` is the single mass, or the mass of element 0
// when `mass_by_type` lists one mass per setfl element.
struct SimulationInput {
  ref::System system;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  bool metal = false;
  bool setfl = false;
  std::string eam_path;
  double mass = 1.0;
  std::vector<double> mass_by_type;
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
