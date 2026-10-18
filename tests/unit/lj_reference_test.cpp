#include "lj_reference.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using tilemd::ref::Compute;
using tilemd::ref::Observables;
using tilemd::ref::PairCoeff;
using tilemd::ref::Shift;
using tilemd::ref::System;

PairCoeff Mono() {
  PairCoeff coeff;
  coeff.type_i = 0;
  coeff.type_j = 0;
  coeff.epsilon = 1.0;
  coeff.sigma = 1.0;
  coeff.cutoff = 2.5;
  return coeff;
}

System PairAt(double separation, Shift shift) {
  System system;
  system.box[0] = system.box[1] = system.box[2] = 20.0;
  system.shift = shift;
  system.pairs = {Mono()};
  system.type = {0, 0};
  system.x = {1.0, 1.0 + separation};
  system.y = {2.0, 2.0};
  system.z = {3.0, 3.0};
  return system;
}

double PairEnergy(double separation, Shift shift) {
  return Compute(PairAt(separation, shift)).potential_energy;
}

double RelativeForceError(const Observables& got, const Observables& expected) {
  const int n = static_cast<int>(expected.fx.size());
  double max_delta = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dx = got.fx[i] - expected.fx[i];
    const double dy = got.fy[i] - expected.fy[i];
    const double dz = got.fz[i] - expected.fz[i];
    max_delta = std::max(max_delta, std::sqrt(dx * dx + dy * dy + dz * dz));
    sum_sq += expected.fx[i] * expected.fx[i] + expected.fy[i] * expected.fy[i] +
              expected.fz[i] * expected.fz[i];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(n));
  return max_delta / std::max(rms, 1.0e-30);
}

std::filesystem::path SourcePath(const std::string& relative) {
  return std::filesystem::path(TILEMD_SOURCE_DIR) / relative;
}

}  // namespace

TEST(LjReference, PairForceMatchesAnalyticDerivative) {
  constexpr double kR = 1.0;
  constexpr double kH = 1.0e-6;
  const double slope =
      (PairEnergy(kR + kH, Shift::None) - PairEnergy(kR - kH, Shift::None)) / (2.0 * kH);
  const Observables obs = Compute(PairAt(kR, Shift::None));
  EXPECT_NEAR(obs.fx[0], slope, 1.0e-6);
  EXPECT_NEAR(obs.fx[1], -slope, 1.0e-6);
  EXPECT_NEAR(obs.fy[0], 0.0, 1.0e-12);
  EXPECT_NEAR(obs.potential_energy, 0.0, 1.0e-12);
  EXPECT_NEAR(obs.fx[0], -24.0, 1.0e-12);
}

TEST(LjReference, EnergyShiftDoesNotChangeForce) {
  const Observables plain = Compute(PairAt(1.2, Shift::None));
  const Observables shifted = Compute(PairAt(1.2, Shift::Energy));
  EXPECT_NEAR(shifted.fx[0], plain.fx[0], 1.0e-12);
  // U(rc) is negative for rc = 2.5σ, so subtracting it raises the pair energy.
  const double sr6 = std::pow(1.0 / 2.5, 6.0);
  const double u_shift = 4.0 * (sr6 * sr6 - sr6);
  EXPECT_NEAR(shifted.potential_energy - plain.potential_energy, -u_shift, 1.0e-12);
}

TEST(LjReference, ForceShiftMatchesAnalyticDerivative) {
  constexpr double kR = 2.0;
  constexpr double kH = 1.0e-6;
  const double slope =
      (PairEnergy(kR + kH, Shift::Force) - PairEnergy(kR - kH, Shift::Force)) / (2.0 * kH);
  const Observables obs = Compute(PairAt(kR, Shift::Force));
  EXPECT_NEAR(obs.fx[0], slope, 1.0e-6);
  EXPECT_NEAR(obs.fx[1], -slope, 1.0e-6);
}

TEST(LjReference, CutoffIsExclusive) {
  const Observables inside = Compute(PairAt(2.5 - 1.0e-4, Shift::None));
  const Observables outside = Compute(PairAt(2.5 + 1.0e-4, Shift::None));
  EXPECT_GT(std::abs(inside.fx[0]), 1.0e-4);
  EXPECT_NEAR(outside.fx[0], 0.0, 0.0);
  EXPECT_NEAR(outside.potential_energy, 0.0, 0.0);
}

TEST(LjReference, MinimumImageCrossesTheBoundary) {
  System system = PairAt(1.0, Shift::None);
  system.x[1] = 1.0 + 1.0 - system.box[0];
  const Observables direct = Compute(PairAt(1.0, Shift::None));
  const Observables wrapped = Compute(system);
  EXPECT_NEAR(wrapped.fx[0], direct.fx[0], 1.0e-12);
  EXPECT_NEAR(wrapped.fx[1], direct.fx[1], 1.0e-12);
}

TEST(LjReference, TotalForceVanishes) {
  System system = PairAt(1.3, Shift::None);
  system.type.push_back(0);
  system.x.push_back(5.0);
  system.y.push_back(6.0);
  system.z.push_back(7.0);
  const Observables obs = Compute(system);
  double fx = 0.0;
  double fy = 0.0;
  double fz = 0.0;
  for (int i = 0; i < 3; ++i) {
    fx += obs.fx[i];
    fy += obs.fy[i];
    fz += obs.fz[i];
  }
  EXPECT_NEAR(fx, 0.0, 1.0e-12);
  EXPECT_NEAR(fy, 0.0, 1.0e-12);
  EXPECT_NEAR(fz, 0.0, 1.0e-12);
}

TEST(LjReference, FixturesMatchRecomputedForces) {
  const std::filesystem::path dir = SourcePath("tests/fixtures");
  const char* names[] = {"fcc_500",      "boundary_500", "cutoff_500",
                         "twotype_2000", "dilute_1000",  "dense_2000"};
  int present = 0;
  for (const char* name : names) {
    const auto xyz = dir / (std::string(name) + ".extxyz");
    const auto json = dir / (std::string(name) + ".json");
    if (!std::filesystem::exists(xyz) || !std::filesystem::exists(json)) continue;
    ++present;
    System system = tilemd::ref::ReadExtxyz(xyz.string());
    tilemd::ref::ReadSystemHeader(json.string(), system);
    const Observables got = Compute(system);
    const Observables stored = tilemd::ref::ReadObservablesJson(json.string());
    if (std::string(name) == "fcc_500") {
      double max_force = 0.0;
      for (double component : got.fx) max_force = std::max(max_force, std::abs(component));
      for (double component : got.fy) max_force = std::max(max_force, std::abs(component));
      for (double component : got.fz) max_force = std::max(max_force, std::abs(component));
      EXPECT_LT(max_force, 1.0e-10);
    }
    EXPECT_LE(RelativeForceError(got, stored), 1.0e-12) << name;
    EXPECT_NEAR(got.potential_energy, stored.potential_energy,
                1.0e-10 * (1.0 + std::abs(stored.potential_energy)))
        << name;
  }
  if (present == 0) GTEST_SKIP() << "fixtures have not been generated";
  EXPECT_EQ(present, 6);
}

TEST(LjReference, MeltMatchesLammps) {
  const auto xyz = SourcePath("tests/fixtures/melt_4000.extxyz");
  const auto json = SourcePath("tests/fixtures/melt_4000.json");
  const auto lammps = SourcePath("tests/validation/lammps_ref/melt_4000_lammps.json");
  if (!std::filesystem::exists(xyz) || !std::filesystem::exists(lammps)) {
    GTEST_SKIP() << "LAMMPS melt fixture has not been generated";
  }
  System system = tilemd::ref::ReadExtxyz(xyz.string());
  tilemd::ref::ReadSystemHeader(json.string(), system);
  const Observables got = Compute(system);
  const Observables reference = tilemd::ref::ReadObservablesJson(lammps.string());
  const double error = RelativeForceError(got, reference);
  EXPECT_LE(error, 1.0e-10) << "relative force error " << error;
  EXPECT_NEAR(got.potential_energy, reference.potential_energy,
              1.0e-10 * (1.0 + std::abs(reference.potential_energy)));
  for (int k = 0; k < 6; ++k) {
    EXPECT_NEAR(got.virial[k], reference.virial[k], 1.0e-9 * (1.0 + std::abs(reference.virial[k])))
        << "virial component " << k;
  }
}

TEST(LjReference, ExtxyzVelocityColumnIsNotTheType) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "tilemd_extxyz_vel.extxyz";
  {
    std::ofstream out(path);
    out << "2\n";
    out << "Lattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 "
           "shift=energy pbc=\"T T T\"\n";
    out << "B 1.5 2.5 3.5 4.5 5.5 6.5 1\n";
    out << "A 0.25 0.5 0.75 9 8 7 0\n";
    out << "1\n";
    out << "Lattice=\"4 0 0 0 5 0 0 0 6\" Properties=species:S:1:pos:R:3:type:I:1 shift=force "
           "pbc=\"T T T\"\n";
    out << "C 1 2 3 2\n";
  }
  const tilemd::ref::ExtxyzFrame first = tilemd::ref::ReadExtxyzFrame(path.string(), 0);
  EXPECT_EQ(tilemd::ref::NumAtoms(first.system), 2);
  EXPECT_EQ(first.system.type[0], 1);
  EXPECT_EQ(first.system.type[1], 0);
  EXPECT_DOUBLE_EQ(first.system.x[0], 1.5);
  EXPECT_DOUBLE_EQ(first.system.y[0], 2.5);
  EXPECT_DOUBLE_EQ(first.system.z[0], 3.5);
  EXPECT_DOUBLE_EQ(first.vx[0], 4.5);
  EXPECT_DOUBLE_EQ(first.vy[0], 5.5);
  EXPECT_DOUBLE_EQ(first.vz[0], 6.5);
  EXPECT_EQ(first.system.shift, Shift::Energy);
  EXPECT_DOUBLE_EQ(first.system.box[0], 10.0);
  const tilemd::ref::ExtxyzFrame second = tilemd::ref::ReadExtxyzFrame(path.string(), 1);
  EXPECT_EQ(tilemd::ref::NumAtoms(second.system), 1);
  EXPECT_EQ(second.system.type[0], 2);
  EXPECT_TRUE(second.vx.empty());
  EXPECT_DOUBLE_EQ(second.system.x[0], 1.0);
  EXPECT_DOUBLE_EQ(second.system.box[2], 6.0);
  EXPECT_EQ(second.system.shift, Shift::Force);
  EXPECT_EQ(tilemd::ref::ReadExtxyz(path.string()).type[0], 1);
  EXPECT_THROW(tilemd::ref::ReadExtxyzFrame(path.string(), 2), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Reference, PerAtomVirialSumsToTheGlobalTensor) {
  const System system = PairAt(1.2, Shift::None);
  const Observables obs = Compute(system);
  ASSERT_EQ(obs.wxx.size(), 2u);
  const std::vector<double>* column[6] = {&obs.wxx, &obs.wyy, &obs.wzz,
                                          &obs.wxy, &obs.wxz, &obs.wyz};
  double sum[6] = {};
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 6; ++k) sum[k] += (*column[k])[static_cast<size_t>(i)];
  }
  for (int k = 0; k < 6; ++k) EXPECT_NEAR(sum[k], obs.virial[k], 1.0e-12);
  EXPECT_NEAR(obs.wxx[0], obs.wxx[1], 1.0e-12);
  EXPECT_GT(std::abs(obs.virial[0]), 1.0e-6);
  EXPECT_NEAR(obs.wxy[0], 0.0, 1.0e-12);
  EXPECT_NEAR(obs.wyy[0], 0.0, 1.0e-12);
}
