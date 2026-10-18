#pragma once

#include <iosfwd>
#include <vector>

#include "io/input.hpp"

namespace tilemd {

// One thermo sample of the CPU reference. Temperature uses 2 KE / (3N − 3)
// when N > 1. Pressure is (N T + trace(W) / 3) / V.
struct ThermoRow {
  int step = 0;
  double time = 0.0;
  double temperature = 0.0;
  double potential_per_atom = 0.0;
  double kinetic_per_atom = 0.0;
  double total_per_atom = 0.0;
  double pressure = 0.0;
};

// Integrates `input.steps` with velocity Verlet, or BAOAB when `langevin` is set.
// An `eam` file uses the cluster-pair walk and the metal-unit factor
// `kMetalFtm2v`. The skin widens that list. A setfl file kicks each atom with
// that element's mass. Langevin is BAOAB; zero friction is the metal Verlet step.
// Samples step 0 and every `thermo_every` steps. `thermo_every == 0`
// samples nothing. When `dump_every > 0` and `dump_path` is not empty, those steps
// are appended as extended-XYZ frames with positions and velocities.
std::vector<ThermoRow> RunReference(const SimulationInput& input,
                                    const std::string& dump_path = "");

void WriteThermo(std::ostream& out, const std::vector<ThermoRow>& rows);

}  // namespace tilemd
