#include "io/run.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace tilemd {
namespace {

ThermoRow Sample(const SimulationInput& input, const ref::Dynamics& state, int step) {
  ref::System system = input.system;
  system.x = state.x;
  system.y = state.y;
  system.z = state.z;
  const ref::Observables obs = ref::Compute(system);
  const int n = ref::NumAtoms(system);
  double kinetic = 0.0;
  for (int i = 0; i < n; ++i) {
    const double vx = state.vx[static_cast<size_t>(i)];
    const double vy = state.vy[static_cast<size_t>(i)];
    const double vz = state.vz[static_cast<size_t>(i)];
    kinetic += 0.5 * input.mass * (vx * vx + vy * vy + vz * vz);
  }
  const double dof = (n > 1) ? static_cast<double>(3 * n - 3) : static_cast<double>(3 * n);
  const double volume = system.box[0] * system.box[1] * system.box[2];
  ThermoRow row;
  row.step = step;
  row.time = input.timestep * static_cast<double>(step);
  row.temperature = (2.0 * kinetic) / dof;
  row.potential_per_atom = obs.potential_energy / static_cast<double>(n);
  row.kinetic_per_atom = kinetic / static_cast<double>(n);
  row.total_per_atom = row.potential_per_atom + row.kinetic_per_atom;
  const double trace = obs.virial[0] + obs.virial[1] + obs.virial[2];
  row.pressure = (static_cast<double>(n) * row.temperature + trace / 3.0) / volume;
  return row;
}

void WriteFrame(std::ofstream& out, const SimulationInput& input, const ref::Dynamics& state,
                int step) {
  const int n = ref::NumAtoms(input.system);
  const char* shift = "none";
  if (input.system.shift == ref::Shift::Energy) shift = "energy";
  if (input.system.shift == ref::Shift::Force) shift = "force";
  out << n << "\n";
  out << "Lattice=\"" << input.system.box[0] << " 0 0 0 " << input.system.box[1] << " 0 0 0 "
      << input.system.box[2] << "\" Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 step=" << step
      << " shift=" << shift << " pbc=\"T T T\"\n";
  for (int i = 0; i < n; ++i) {
    const int type = input.system.type[static_cast<size_t>(i)];
    if (type < 0 || type > 25) throw std::runtime_error("type does not map to a species letter");
    const char species = static_cast<char>('A' + type);
    out << species << " " << state.x[static_cast<size_t>(i)] << " "
        << state.y[static_cast<size_t>(i)] << " " << state.z[static_cast<size_t>(i)] << " "
        << state.vx[static_cast<size_t>(i)] << " " << state.vy[static_cast<size_t>(i)] << " "
        << state.vz[static_cast<size_t>(i)] << " " << type << "\n";
  }
}

}  // namespace

std::vector<ThermoRow> RunReference(const SimulationInput& input, const std::string& dump_path) {
  if (input.steps < 0) throw std::runtime_error("run length must be non-negative");
  if (!(input.timestep > 0.0)) throw std::runtime_error("timestep must be positive");
  ref::System system = input.system;
  ref::Dynamics state = ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  std::ofstream dump;
  if (input.dump_every > 0 && !dump_path.empty()) {
    dump.open(dump_path);
    if (!dump) throw std::runtime_error("cannot write " + dump_path);
    dump.setf(std::ios::scientific);
    dump.precision(17);
  }
  std::vector<ThermoRow> rows;
  if (input.thermo_every > 0) rows.push_back(Sample(input, state, 0));
  if (dump) WriteFrame(dump, input, state, 0);
  for (int step = 1; step <= input.steps; ++step) {
    if (input.langevin) {
      ref::LangevinBaoabStep(system, state, input.timestep, input.mass, input.thermo,
                             static_cast<std::uint64_t>(step - 1));
    } else {
      ref::VelocityVerletStep(system, state, input.timestep, input.mass);
    }
    if (input.thermo_every > 0 && step % input.thermo_every == 0) {
      rows.push_back(Sample(input, state, step));
    }
    if (dump && step % input.dump_every == 0) WriteFrame(dump, input, state, step);
  }
  return rows;
}

void WriteThermo(std::ostream& out, const std::vector<ThermoRow>& rows) {
  out << std::setprecision(16);
  out << "step time temperature pe_per_atom ke_per_atom e_per_atom pressure\n";
  for (const ThermoRow& row : rows) {
    out << row.step << " " << row.time << " " << row.temperature << " " << row.potential_per_atom
        << " " << row.kinetic_per_atom << " " << row.total_per_atom << " " << row.pressure << "\n";
  }
}

}  // namespace tilemd
