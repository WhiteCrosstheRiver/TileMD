#include "io/input.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "io/run.hpp"

namespace {

const char* kMelt = R"(
units          lj
box            fcc 2 2 2 0.8442
types          1
lj             0 0  epsilon 1.0  sigma 1.0  cutoff 2.5
lj_shift       force
mass           0 1.0
velocity       1.44 seed 99
skin           0.3
timestep       0.005
ensemble       nve
thermo_every   100
dump_every     0
graph_unroll   50
run            10
)";

double Momentum(const std::vector<double>& v) {
  double sum = 0.0;
  for (double value : v) sum += value;
  return sum;
}

}  // namespace

TEST(Input, FccMeltMatchesTheSketch) {
  const tilemd::SimulationInput input = tilemd::ParseInput(kMelt, ".");
  const int n = tilemd::ref::NumAtoms(input.system);
  EXPECT_EQ(n, 32);
  const double volume = input.system.box[0] * input.system.box[1] * input.system.box[2];
  EXPECT_NEAR(static_cast<double>(n) / volume, 0.8442, 1.0e-12);
  EXPECT_NEAR(Momentum(input.vx), 0.0, 1.0e-10);
  EXPECT_NEAR(Momentum(input.vy), 0.0, 1.0e-10);
  EXPECT_NEAR(Momentum(input.vz), 0.0, 1.0e-10);
  EXPECT_EQ(input.system.shift, tilemd::ref::Shift::Force);
  EXPECT_FALSE(input.langevin);
  EXPECT_EQ(input.steps, 10);
  EXPECT_DOUBLE_EQ(input.skin, 0.3);
  EXPECT_DOUBLE_EQ(input.timestep, 0.005);
  EXPECT_EQ(input.thermo_every, 100);
  EXPECT_EQ(input.graph_unroll, 50);
  EXPECT_DOUBLE_EQ(input.mass, 1.0);

  tilemd::ref::System system = input.system;
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  for (int step = 0; step < 5; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, input.timestep, input.mass);
  }
  EXPECT_TRUE(std::isfinite(state.x[0]));
  EXPECT_GT(std::abs(state.vx[0]) + std::abs(state.vy[0]) + std::abs(state.vz[0]), 0.0);
}

TEST(Input, VelocitiesRepeatForTheSameSeed) {
  std::string text = kMelt;
  const auto langevin = text.find("ensemble       nve");
  text.replace(langevin, std::string("ensemble       nve").size(),
               "ensemble       langevin 1.0 0.5 7");
  const tilemd::SimulationInput first = tilemd::ParseInput(text, ".");
  const tilemd::SimulationInput second = tilemd::ParseInput(text, ".");
  ASSERT_TRUE(first.langevin);
  EXPECT_DOUBLE_EQ(first.thermo.temperature, 1.0);
  EXPECT_DOUBLE_EQ(first.thermo.gamma, 0.5);
  EXPECT_EQ(first.thermo.seed, 7u);
  EXPECT_EQ(first.vx, second.vx);
  std::string other = text;
  const auto seed = other.find("seed 99");
  other.replace(seed, 7, "seed 100");
  const tilemd::SimulationInput changed = tilemd::ParseInput(other, ".");
  EXPECT_NE(first.vx, changed.vx);
}

TEST(Input, ReferenceRunMatchesVerlet) {
  std::string text = kMelt;
  const auto steps = text.find("run            10");
  text.replace(steps, std::string("run            10").size(), "run            20");
  const auto thermo = text.find("thermo_every   100");
  text.replace(thermo, std::string("thermo_every   100").size(), "thermo_every   10");
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input);
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0].step, 0);
  EXPECT_EQ(rows[1].step, 10);
  EXPECT_EQ(rows[2].step, 20);
  const double e0 = rows[0].total_per_atom;
  const double e1 = rows[2].total_per_atom;
  std::cout << "cpu input nve e0 " << e0 << " e20 " << e1 << "\n";
  // A fresh T=1.44 lattice oscillates. This bound records that swing.
  const double swing = std::abs(e1 - e0) / std::abs(e0);
  EXPECT_GT(swing, 8.0e-4);
  EXPECT_LT(swing, 2.0e-3);
  const int n = tilemd::ref::NumAtoms(input.system);
  const double kinetic0 = rows[0].kinetic_per_atom * static_cast<double>(n);
  const double dof = static_cast<double>(3 * n - 3);
  EXPECT_NEAR(rows[0].temperature, 2.0 * kinetic0 / dof, 1.0e-12);
  tilemd::ref::System system = input.system;
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  for (int step = 0; step < 20; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, input.timestep, input.mass);
  }
  system.x = state.x;
  system.y = state.y;
  system.z = state.z;
  const tilemd::ref::Observables obs = tilemd::ref::Compute(system);
  double kinetic = 0.0;
  for (int i = 0; i < n; ++i) {
    kinetic += 0.5 * input.mass *
               (state.vx[static_cast<size_t>(i)] * state.vx[static_cast<size_t>(i)] +
                state.vy[static_cast<size_t>(i)] * state.vy[static_cast<size_t>(i)] +
                state.vz[static_cast<size_t>(i)] * state.vz[static_cast<size_t>(i)]);
  }
  const double total = (obs.potential_energy + kinetic) / static_cast<double>(n);
  EXPECT_NEAR(rows[2].total_per_atom, total, 1.0e-12);
}

TEST(Input, ZeroTemperatureLangevinCools) {
  std::string text = kMelt;
  const auto ensemble = text.find("ensemble       nve");
  text.replace(ensemble, std::string("ensemble       nve").size(),
               "ensemble       langevin 0 2.0 3");
  const auto steps = text.find("run            10");
  text.replace(steps, std::string("run            10").size(), "run            40");
  const auto thermo = text.find("thermo_every   100");
  text.replace(thermo, std::string("thermo_every   100").size(), "thermo_every   40");
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input);
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_LT(rows[1].kinetic_per_atom, 0.5 * rows[0].kinetic_per_atom);
}

TEST(Input, UnknownKeyIsRejected) {
  EXPECT_THROW(tilemd::ParseInput("nope 1\n", "."), std::runtime_error);
}

TEST(Input, DumpWritesAFrameOnTheRequestedSteps) {
  std::string text = kMelt;
  const auto dump = text.find("dump_every     0");
  text.replace(dump, std::string("dump_every     0").size(), "dump_every     10");
  const auto steps = text.find("run            10");
  text.replace(steps, std::string("run            10").size(), "run            20");
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "tilemd_input_dump.extxyz";
  std::filesystem::remove(path);
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input, path.string());
  EXPECT_EQ(rows.size(), 1u);
  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  int frames = 0;
  bool saw0 = false;
  bool saw10 = false;
  bool saw20 = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line == "32") ++frames;
    if (line.find("step=0 ") != std::string::npos) saw0 = true;
    if (line.find("step=10 ") != std::string::npos) saw10 = true;
    if (line.find("step=20 ") != std::string::npos) saw20 = true;
  }
  EXPECT_EQ(frames, 3);
  EXPECT_TRUE(saw0);
  EXPECT_TRUE(saw10);
  EXPECT_TRUE(saw20);
  in.close();
  const tilemd::ref::ExtxyzFrame first = tilemd::ref::ReadExtxyzFrame(path.string(), 0);
  const tilemd::ref::ExtxyzFrame later = tilemd::ref::ReadExtxyzFrame(path.string(), 2);
  ASSERT_EQ(tilemd::ref::NumAtoms(first.system), 32);
  ASSERT_EQ(first.vx.size(), 32u);
  tilemd::ref::System system = input.system;
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  double max_dx = 0.0;
  double max_dv = 0.0;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(first.system.type[static_cast<size_t>(i)], 0);
    max_dx = std::max(
        max_dx, std::abs(first.system.x[static_cast<size_t>(i)] - state.x[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(first.system.y[static_cast<size_t>(i)] - state.y[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(first.system.z[static_cast<size_t>(i)] - state.z[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(first.vx[static_cast<size_t>(i)] - state.vx[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(first.vy[static_cast<size_t>(i)] - state.vy[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(first.vz[static_cast<size_t>(i)] - state.vz[static_cast<size_t>(i)]));
  }
  EXPECT_LT(max_dx, 1.0e-12);
  EXPECT_LT(max_dv, 1.0e-12);
  for (int step = 0; step < 20; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, input.timestep, input.mass);
  }
  max_dx = 0.0;
  max_dv = 0.0;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(later.system.type[static_cast<size_t>(i)], 0);
    max_dx = std::max(
        max_dx, std::abs(later.system.x[static_cast<size_t>(i)] - state.x[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(later.system.y[static_cast<size_t>(i)] - state.y[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(later.system.z[static_cast<size_t>(i)] - state.z[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(later.vx[static_cast<size_t>(i)] - state.vx[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(later.vy[static_cast<size_t>(i)] - state.vy[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(later.vz[static_cast<size_t>(i)] - state.vz[static_cast<size_t>(i)]));
  }
  EXPECT_LT(max_dx, 1.0e-12);
  EXPECT_LT(max_dv, 1.0e-12);
  std::filesystem::remove(path);
}

TEST(Input, StructureLoadsPositions) {
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  const std::string text =
      "structure cutoff_500.extxyz\n"
      "lj 0 0 epsilon 1.0 sigma 1.0 cutoff 2.5\n";
  const tilemd::SimulationInput input = tilemd::ParseInput(text, dir.string());
  EXPECT_EQ(tilemd::ref::NumAtoms(input.system), 500);
  EXPECT_EQ(Momentum(input.vx), 0.0);
}
