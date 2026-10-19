#include "io/run.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

#include "force/eam.hpp"
#include "force/tile_attention.hpp"
#include "neighbor/cluster_order.hpp"

namespace tilemd {
namespace {

void WriteFrame(std::ofstream& out, const SimulationInput& input, const ref::Dynamics& state,
                int step, const ref::Observables* atom_virial);
void DumpFrame(std::ofstream& dump, const SimulationInput& input, const ref::Dynamics& state,
               int step, const AttentionResult* force);

double AtomMass(const SimulationInput& input, int atom) {
  if (input.mass_by_type.empty()) return input.mass;
  const int type = input.system.type.at(static_cast<size_t>(atom));
  return input.mass_by_type.at(static_cast<size_t>(type));
}

ThermoRow MakeRow(const SimulationInput& input, const ref::Dynamics& state, int step,
                  double potential, const double* virial, bool metal) {
  const int n = ref::NumAtoms(input.system);
  double kinetic = 0.0;
  for (int i = 0; i < n; ++i) {
    const double vx = state.vx[static_cast<size_t>(i)];
    const double vy = state.vy[static_cast<size_t>(i)];
    const double vz = state.vz[static_cast<size_t>(i)];
    kinetic += 0.5 * AtomMass(input, i) * (vx * vx + vy * vy + vz * vz);
  }
  if (metal) kinetic *= kMetalMvv2e;
  const double dof = (n > 1) ? static_cast<double>(3 * n - 3) : static_cast<double>(3 * n);
  const double boltz = metal ? kMetalBoltz : 1.0;
  const double nktv2p = metal ? kMetalNktv2p : 1.0;
  const double volume = input.system.box[0] * input.system.box[1] * input.system.box[2];
  ThermoRow row;
  row.step = step;
  row.time = input.timestep * static_cast<double>(step);
  row.temperature = (2.0 * kinetic) / (dof * boltz);
  row.potential_per_atom = potential / static_cast<double>(n);
  row.kinetic_per_atom = kinetic / static_cast<double>(n);
  row.total_per_atom = row.potential_per_atom + row.kinetic_per_atom;
  const double trace = virial[0] + virial[1] + virial[2];
  row.pressure = nktv2p * (static_cast<double>(n) * boltz * row.temperature + trace / 3.0) / volume;
  return row;
}

ThermoRow Sample(const SimulationInput& input, const ref::Dynamics& state, int step) {
  ref::System system = input.system;
  system.x = state.x;
  system.y = state.y;
  system.z = state.z;
  const ref::Observables obs = ref::Compute(system);
  return MakeRow(input, state, step, obs.potential_energy, obs.virial, false);
}

double WrapMetal(double value, double length, int* image) {
  return WrapImage(value, length, image);
}

template <typename Potential>
AttentionResult MetalForces(Potential& potential, const SimulationInput& input,
                            const ref::Dynamics& state) {
  const int n = ref::NumAtoms(input.system);
  return WarpLaneAttention(potential, input.system.box, state.x.data(), state.y.data(),
                           state.z.data(), input.system.type.data(), n, input.skin);
}

void DriftMetal(ref::Dynamics& state, const SimulationInput& input, double dt) {
  const int n = static_cast<int>(state.x.size());
  for (int i = 0; i < n; ++i) {
    state.x[static_cast<size_t>(i)] =
        WrapMetal(state.x[static_cast<size_t>(i)] + dt * state.vx[static_cast<size_t>(i)],
                  input.system.box[0], &state.imgx[static_cast<size_t>(i)]);
    state.y[static_cast<size_t>(i)] =
        WrapMetal(state.y[static_cast<size_t>(i)] + dt * state.vy[static_cast<size_t>(i)],
                  input.system.box[1], &state.imgy[static_cast<size_t>(i)]);
    state.z[static_cast<size_t>(i)] =
        WrapMetal(state.z[static_cast<size_t>(i)] + dt * state.vz[static_cast<size_t>(i)],
                  input.system.box[2], &state.imgz[static_cast<size_t>(i)]);
  }
}

void OrnsteinUhlenbeckMetal(ref::Dynamics& state, const SimulationInput& input,
                            std::uint64_t step) {
  const int n = static_cast<int>(state.x.size());
  const double decay = std::exp(-input.thermo.gamma * input.timestep);
  const double span = 1.0 - decay * decay;
  for (int i = 0; i < n; ++i) {
    const double mass = AtomMass(input, i);
    const double noise =
        std::sqrt(input.thermo.temperature * kMetalBoltz * kMetalFtm2v * span / mass);
    double draw[3] = {};
    if (noise > 0.0) ref::StandardNormals(input.thermo.seed, step, i, draw);
    state.vx[static_cast<size_t>(i)] = decay * state.vx[static_cast<size_t>(i)] + noise * draw[0];
    state.vy[static_cast<size_t>(i)] = decay * state.vy[static_cast<size_t>(i)] + noise * draw[1];
    state.vz[static_cast<size_t>(i)] = decay * state.vz[static_cast<size_t>(i)] + noise * draw[2];
  }
}

void KickMetal(ref::Dynamics& state, const AttentionResult& force, const SimulationInput& input) {
  const int n = static_cast<int>(state.x.size());
  const double scale = 0.5 * input.timestep * kMetalFtm2v;
  for (int i = 0; i < n; ++i) {
    const double half = scale / AtomMass(input, i);
    state.vx[static_cast<size_t>(i)] += half * force.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] += half * force.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] += half * force.fz[static_cast<size_t>(i)];
  }
}

template <typename Potential>
std::vector<ThermoRow> IntegrateMetal(Potential& potential, const SimulationInput& input,
                                      std::ofstream& dump) {
  ref::System system = input.system;
  ref::Dynamics state = ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  AttentionResult force = MetalForces(potential, input, state);
  std::vector<ThermoRow> rows;
  if (input.thermo_every > 0)
    rows.push_back(MakeRow(input, state, 0, force.potential_energy, force.virial, true));
  if (dump) DumpFrame(dump, input, state, 0, &force);
  for (int step = 1; step <= input.steps; ++step) {
    KickMetal(state, force, input);
    if (input.langevin && input.thermo.gamma > 0.0) {
      DriftMetal(state, input, 0.5 * input.timestep);
      OrnsteinUhlenbeckMetal(state, input, static_cast<std::uint64_t>(step - 1));
      DriftMetal(state, input, 0.5 * input.timestep);
    } else {
      DriftMetal(state, input, input.timestep);
    }
    force = MetalForces(potential, input, state);
    KickMetal(state, force, input);
    if (input.thermo_every > 0 && step % input.thermo_every == 0) {
      rows.push_back(MakeRow(input, state, step, force.potential_energy, force.virial, true));
    }
    if (dump && step % input.dump_every == 0) DumpFrame(dump, input, state, step, &force);
  }
  return rows;
}

std::vector<ThermoRow> RunMetal(const SimulationInput& input, std::ofstream& dump) {
  if (input.setfl) {
    EamSetfl potential(input.eam_path);
    const int n = ref::NumAtoms(input.system);
    potential.SetAtoms(input.system.type.data(), n);
    return IntegrateMetal(potential, input, dump);
  }
  EamFuncfl potential(input.eam_path);
  return IntegrateMetal(potential, input, dump);
}

ref::Observables VirialView(const AttentionResult& force) {
  ref::Observables obs;
  obs.wxx = force.wxx;
  obs.wyy = force.wyy;
  obs.wzz = force.wzz;
  obs.wxy = force.wxy;
  obs.wxz = force.wxz;
  obs.wyz = force.wyz;
  return obs;
}

void DumpFrame(std::ofstream& dump, const SimulationInput& input, const ref::Dynamics& state,
               int step, const AttentionResult* force) {
  ref::Observables held;
  const ref::Observables* view = nullptr;
  if (input.dump_virial) {
    if (force != nullptr) {
      held = VirialView(*force);
    } else {
      ref::System system = input.system;
      system.x = state.x;
      system.y = state.y;
      system.z = state.z;
      held = ref::Compute(system);
    }
    view = &held;
  }
  WriteFrame(dump, input, state, step, view);
}

void WriteFrame(std::ofstream& out, const SimulationInput& input, const ref::Dynamics& state,
                int step, const ref::Observables* atom_virial) {
  const int n = ref::NumAtoms(input.system);
  const char* shift = "none";
  if (input.system.shift == ref::Shift::Energy) shift = "energy";
  if (input.system.shift == ref::Shift::Force) shift = "force";
  const char* properties = atom_virial == nullptr
                               ? "species:S:1:pos:R:3:vel:R:3:type:I:1"
                               : "species:S:1:pos:R:3:vel:R:3:type:I:1:virial:R:6";
  out << n << "\n";
  out << "Lattice=\"" << input.system.box[0] << " 0 0 0 " << input.system.box[1] << " 0 0 0 "
      << input.system.box[2] << "\" Properties=" << properties << " step=" << step
      << " shift=" << shift << " pbc=\"T T T\"\n";
  for (int i = 0; i < n; ++i) {
    const int type = input.system.type[static_cast<size_t>(i)];
    if (type < 0 || type > 25) throw std::runtime_error("type does not map to a species letter");
    const char species = static_cast<char>('A' + type);
    out << species << " " << state.x[static_cast<size_t>(i)] << " "
        << state.y[static_cast<size_t>(i)] << " " << state.z[static_cast<size_t>(i)] << " "
        << state.vx[static_cast<size_t>(i)] << " " << state.vy[static_cast<size_t>(i)] << " "
        << state.vz[static_cast<size_t>(i)] << " " << type;
    if (atom_virial != nullptr) {
      const std::size_t atom = static_cast<std::size_t>(i);
      out << " " << atom_virial->wxx[atom] << " " << atom_virial->wyy[atom] << " "
          << atom_virial->wzz[atom] << " " << atom_virial->wxy[atom] << " "
          << atom_virial->wxz[atom] << " " << atom_virial->wyz[atom];
    }
    out << "\n";
  }
}

}  // namespace

std::vector<ThermoRow> RunReference(const SimulationInput& input, const std::string& dump_path) {
  if (input.steps < 0) throw std::runtime_error("run length must be non-negative");
  if (!(input.timestep > 0.0)) throw std::runtime_error("timestep must be positive");
  std::ofstream dump;
  if (input.dump_every > 0 && !dump_path.empty()) {
    dump.open(dump_path);
    if (!dump) throw std::runtime_error("cannot write " + dump_path);
    dump.setf(std::ios::scientific);
    dump.precision(17);
  }
  if (input.metal) return RunMetal(input, dump);
  ref::System system = input.system;
  ref::Dynamics state = ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  std::vector<ThermoRow> rows;
  if (input.thermo_every > 0) rows.push_back(Sample(input, state, 0));
  if (dump) DumpFrame(dump, input, state, 0, nullptr);
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
    if (dump && step % input.dump_every == 0) DumpFrame(dump, input, state, step, nullptr);
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
