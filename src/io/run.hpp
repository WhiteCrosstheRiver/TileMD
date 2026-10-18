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
// Samples step 0 and every `thermo_every` steps. `thermo_every == 0` samples nothing.
// Extended-XYZ dumps are not written.
std::vector<ThermoRow> RunReference(const SimulationInput& input);

void WriteThermo(std::ostream& out, const std::vector<ThermoRow>& rows);

}  // namespace tilemd
