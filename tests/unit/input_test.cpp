#include "io/input.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>

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

TEST(Input, UnknownKeyIsRejected) {
  EXPECT_THROW(tilemd::ParseInput("nope 1\n", "."), std::runtime_error);
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
