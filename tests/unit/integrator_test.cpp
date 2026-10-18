#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include "lj_reference.hpp"
#include "md/tile_thermo.hpp"
#include "neighbor/cluster_order.hpp"

namespace {

double MaxAbs(const std::vector<double>& a, const std::vector<double>& b) {
  double max_delta = 0.0;
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    max_delta =
        std::max(max_delta, std::abs(a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]));
  }
  return max_delta;
}

tilemd::ref::System SmallMelt() {
  tilemd::ref::System system;
  system.box[0] = 8.0;
  system.box[1] = 8.0;
  system.box[2] = 8.0;
  system.shift = tilemd::ref::Shift::None;
  tilemd::ref::PairCoeff coeff;
  system.pairs = {coeff};
  for (int i = 0; i < 40; ++i) {
    system.type.push_back(0);
    system.x.push_back(0.5 + (i % 5) * 1.5);
    system.y.push_back(0.5 + ((i / 5) % 4) * 1.8);
    system.z.push_back(0.6 + (i / 20) * 3.0);
  }
  return system;
}

}  // namespace

TEST(Integrator, LeapfrogMatchesVerletFor100Steps) {
  tilemd::ref::System origin = SmallMelt();
  std::mt19937 rng(11);
  std::normal_distribution<double> speed(0.0, 0.2);
  std::vector<double> vx(40);
  std::vector<double> vy(40);
  std::vector<double> vz(40);
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  for (int i = 0; i < 40; ++i) {
    vx[static_cast<size_t>(i)] = speed(rng);
    vy[static_cast<size_t>(i)] = speed(rng);
    vz[static_cast<size_t>(i)] = speed(rng);
    px += vx[static_cast<size_t>(i)];
    py += vy[static_cast<size_t>(i)];
    pz += vz[static_cast<size_t>(i)];
  }
  for (int i = 0; i < 40; ++i) {
    vx[static_cast<size_t>(i)] -= px / 40.0;
    vy[static_cast<size_t>(i)] -= py / 40.0;
    vz[static_cast<size_t>(i)] -= pz / 40.0;
  }

  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  tilemd::ref::System verlet_system = origin;
  tilemd::ref::System leap_system = origin;
  tilemd::ref::Dynamics verlet = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Dynamics leap = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::PrepareLeapfrog(leap_system, leap, dt, mass);

  double max_x = 0.0;
  double max_v = 0.0;
  for (int step = 0; step < 100; ++step) {
    tilemd::ref::VelocityVerletStep(verlet_system, verlet, dt, mass);
    tilemd::ref::LeapfrogStep(leap_system, leap, dt, mass);
    std::vector<double> ivx;
    std::vector<double> ivy;
    std::vector<double> ivz;
    tilemd::ref::IntegerTimeVelocity(leap_system, leap, dt, mass, ivx, ivy, ivz);
    max_x = std::max(max_x, MaxAbs(verlet.x, leap.x));
    max_x = std::max(max_x, MaxAbs(verlet.y, leap.y));
    max_x = std::max(max_x, MaxAbs(verlet.z, leap.z));
    max_v = std::max(max_v, MaxAbs(verlet.vx, ivx));
    max_v = std::max(max_v, MaxAbs(verlet.vy, ivy));
    max_v = std::max(max_v, MaxAbs(verlet.vz, ivz));
    for (int i = 0; i < 40; ++i) {
      EXPECT_EQ(verlet.imgx[static_cast<size_t>(i)], leap.imgx[static_cast<size_t>(i)]);
      EXPECT_EQ(verlet.imgy[static_cast<size_t>(i)], leap.imgy[static_cast<size_t>(i)]);
      EXPECT_EQ(verlet.imgz[static_cast<size_t>(i)], leap.imgz[static_cast<size_t>(i)]);
    }
  }
  std::cout << "leapfrog vs verlet max |dx| " << max_x << " max |dv| " << max_v << "\n";
  EXPECT_LT(max_x, 1.0e-8) << "position delta after 100 steps";
  EXPECT_LT(max_v, 1.0e-8) << "integer-time velocity delta after 100 steps";
}

TEST(Integrator, CrossingBelowZeroDecrementsTheImage) {
  tilemd::ref::System system;
  system.box[0] = 10.0;
  system.box[1] = 10.0;
  system.box[2] = 10.0;
  system.shift = tilemd::ref::Shift::None;
  tilemd::ref::PairCoeff coeff;
  system.pairs = {coeff};
  system.type = {0, 0};
  system.x = {0.01, 6.0};
  system.y = {1.0, 1.0};
  system.z = {1.0, 1.0};
  tilemd::ref::Dynamics state =
      tilemd::ref::MakeDynamics(system, {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0});
  tilemd::ref::VelocityVerletStep(system, state, 0.02, 1.0);
  EXPECT_EQ(state.imgx[0], -1);
  EXPECT_EQ(state.imgx[1], 0);
  EXPECT_GT(state.x[0], 9.0);
  EXPECT_LT(state.x[0], 10.0);
  int image = 3;
  EXPECT_DOUBLE_EQ(tilemd::WrapImage(10.0, 10.0, &image), 0.0);
  EXPECT_EQ(image, 4);
  image = 0;
  EXPECT_NEAR(tilemd::WrapImage(-25.0, 10.0, &image), 5.0, 1.0e-12);
  EXPECT_EQ(image, -3);
  image = 0;
  EXPECT_DOUBLE_EQ(tilemd::WrapImage(0.0, 10.0, &image), 0.0);
  EXPECT_EQ(image, 0);
}

tilemd::ref::Observables Observe(const tilemd::ref::System& origin,
                                 const tilemd::ref::Dynamics& state) {
  tilemd::ref::System system = origin;
  system.x = state.x;
  system.y = state.y;
  system.z = state.z;
  return tilemd::ref::Compute(system);
}

double Kinetic(const std::vector<double>& vx, const std::vector<double>& vy,
               const std::vector<double>& vz) {
  double ke = 0.0;
  for (int i = 0; i < static_cast<int>(vx.size()); ++i) {
    ke += 0.5 * (vx[static_cast<size_t>(i)] * vx[static_cast<size_t>(i)] +
                 vy[static_cast<size_t>(i)] * vy[static_cast<size_t>(i)] +
                 vz[static_cast<size_t>(i)] * vz[static_cast<size_t>(i)]);
  }
  return ke;
}

TEST(Integrator, FusedThermoStepsMatchVerlet) {
  const std::vector<int> steps = tilemd::TileThermoSteps(8, 2);
  ASSERT_EQ(steps.size(), 5u);
  EXPECT_EQ(steps[0], 0);
  EXPECT_EQ(steps[1], 2);
  EXPECT_EQ(steps[2], 4);
  EXPECT_EQ(steps[3], 6);
  EXPECT_EQ(steps[4], 8);
  EXPECT_TRUE(tilemd::TileThermoSteps(4, 0).empty());

  tilemd::ref::System origin = SmallMelt();
  std::vector<double> vx(40, 0.2);
  std::vector<double> vy(40, -0.1);
  std::vector<double> vz(40, 0.05);
  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  constexpr int every = 2;
  constexpr int nstep = 8;
  tilemd::ref::System verlet_system = origin;
  tilemd::ref::System leap_system = origin;
  tilemd::ref::Dynamics verlet = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Dynamics leap = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::PrepareLeapfrog(leap_system, leap, dt, mass);

  for (int step = 0; step < nstep; ++step) {
    if (tilemd::TileRecordsThermo(step, every)) {
      std::vector<double> ivx;
      std::vector<double> ivy;
      std::vector<double> ivz;
      tilemd::ref::IntegerTimeVelocity(leap_system, leap, dt, mass, ivx, ivy, ivz);
      const tilemd::ref::Observables got = Observe(origin, leap);
      const tilemd::ref::Observables oracle = Observe(origin, verlet);
      EXPECT_NEAR(got.potential_energy, oracle.potential_energy, 1e-12);
      EXPECT_NEAR(Kinetic(ivx, ivy, ivz), Kinetic(verlet.vx, verlet.vy, verlet.vz), 1e-12);
      for (int axis = 0; axis < 6; ++axis) {
        EXPECT_NEAR(got.virial[axis], oracle.virial[axis], 1e-12);
      }
    }
    tilemd::ref::VelocityVerletStep(verlet_system, verlet, dt, mass);
    tilemd::ref::LeapfrogStep(leap_system, leap, dt, mass);
  }
  ASSERT_TRUE(tilemd::TileRecordsThermo(nstep, every));
  std::vector<double> ivx;
  std::vector<double> ivy;
  std::vector<double> ivz;
  tilemd::ref::IntegerTimeVelocity(leap_system, leap, dt, mass, ivx, ivy, ivz);
  const tilemd::ref::Observables got = Observe(origin, leap);
  const tilemd::ref::Observables oracle = Observe(origin, verlet);
  EXPECT_NEAR(got.potential_energy, oracle.potential_energy, 1e-12);
  EXPECT_NEAR(Kinetic(ivx, ivy, ivz), Kinetic(verlet.vx, verlet.vy, verlet.vz), 1e-12);
}

TEST(Integrator, HalfStepAverageRecoversThePreMoveVelocity) {
  tilemd::ref::System origin = SmallMelt();
  std::vector<double> vx(40, 0.2);
  std::vector<double> vy(40, -0.1);
  std::vector<double> vz(40, 0.05);
  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  tilemd::ref::Dynamics start = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::System system = origin;
  tilemd::ref::Dynamics half = start;
  tilemd::ref::PrepareLeapfrog(system, half, dt, mass);
  std::vector<double> recovered_x;
  std::vector<double> recovered_y;
  std::vector<double> recovered_z;
  tilemd::ref::IntegerTimeVelocity(system, half, dt, mass, recovered_x, recovered_y, recovered_z);
  EXPECT_EQ(MaxAbs(start.vx, recovered_x), 0.0);
  EXPECT_EQ(MaxAbs(start.vy, recovered_y), 0.0);
  EXPECT_EQ(MaxAbs(start.vz, recovered_z), 0.0);
}

tilemd::ref::Observables ForceFromFloatCoordinates(tilemd::ref::System system,
                                                   const tilemd::ref::Dynamics& state) {
  const int n = tilemd::ref::NumAtoms(system);
  system.x.resize(static_cast<size_t>(n));
  system.y.resize(static_cast<size_t>(n));
  system.z.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    system.x[static_cast<size_t>(i)] =
        static_cast<double>(static_cast<float>(state.x[static_cast<size_t>(i)]));
    system.y[static_cast<size_t>(i)] =
        static_cast<double>(static_cast<float>(state.y[static_cast<size_t>(i)]));
    system.z[static_cast<size_t>(i)] =
        static_cast<double>(static_cast<float>(state.z[static_cast<size_t>(i)]));
  }
  return tilemd::ref::Compute(system);
}

void WrapLikeReference(double& value, double length) {
  double periods = std::floor(value / length);
  value -= periods * length;
  if (value >= length) value -= length;
  if (value < 0.0) value += length;
}

double ForceRelative(const tilemd::ref::Observables& got, const tilemd::ref::Observables& oracle) {
  const int n = static_cast<int>(oracle.fx.size());
  double max_delta = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ex = got.fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
    const double ey = got.fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
    const double ez = got.fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
    max_delta = std::max(max_delta, std::sqrt(ex * ex + ey * ey + ez * ez));
    sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
              oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
              oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(std::max(n, 1)));
  return max_delta / std::max(rms, 1.0e-30);
}

TEST(Integrator, FloatCoordinatesChangeTheMeltForce) {
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  tilemd::ref::System system = tilemd::ref::ReadExtxyz((dir / "melt_4000.extxyz").string());
  tilemd::ref::ReadSystemHeader((dir / "melt_4000.json").string(), system);
  tilemd::ref::Dynamics state;
  state.x = system.x;
  state.y = system.y;
  state.z = system.z;
  const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
  const tilemd::ref::Observables rounded = ForceFromFloatCoordinates(system, state);
  const double rel = ForceRelative(rounded, oracle);
  std::cout << "pos4 melt force rel " << rel << "\n";
  EXPECT_GT(rel, 4.0e-5);
  EXPECT_LT(rel, 8.0e-5);
}

TEST(Integrator, FloatCoordinatesDriftFromDoubleLeapfrog) {
  tilemd::ref::System origin = SmallMelt();
  const int n = 40;
  std::vector<double> vx(static_cast<size_t>(n));
  std::vector<double> vy(static_cast<size_t>(n));
  std::vector<double> vz(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] = 0.05 * ((i % 5) - 2);
    vy[static_cast<size_t>(i)] = 0.04 * ((i % 4) - 1);
    vz[static_cast<size_t>(i)] = 0.03 * ((i % 3) - 1);
  }
  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  tilemd::ref::System full_system = origin;
  tilemd::ref::System rounded_system = origin;
  tilemd::ref::Dynamics full = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Dynamics rounded = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::PrepareLeapfrog(full_system, full, dt, mass);
  {
    const tilemd::ref::Observables force = ForceFromFloatCoordinates(rounded_system, rounded);
    const double half = 0.5 * dt / mass;
    for (int i = 0; i < n; ++i) {
      rounded.vx[static_cast<size_t>(i)] -= half * force.fx[static_cast<size_t>(i)];
      rounded.vy[static_cast<size_t>(i)] -= half * force.fy[static_cast<size_t>(i)];
      rounded.vz[static_cast<size_t>(i)] -= half * force.fz[static_cast<size_t>(i)];
    }
  }
  double max_x = 0.0;
  for (int step = 0; step < 100; ++step) {
    tilemd::ref::LeapfrogStep(full_system, full, dt, mass);
    const tilemd::ref::Observables force = ForceFromFloatCoordinates(rounded_system, rounded);
    const double scale = dt / mass;
    for (int i = 0; i < n; ++i) {
      rounded.vx[static_cast<size_t>(i)] += scale * force.fx[static_cast<size_t>(i)];
      rounded.vy[static_cast<size_t>(i)] += scale * force.fy[static_cast<size_t>(i)];
      rounded.vz[static_cast<size_t>(i)] += scale * force.fz[static_cast<size_t>(i)];
      rounded.x[static_cast<size_t>(i)] += dt * rounded.vx[static_cast<size_t>(i)];
      rounded.y[static_cast<size_t>(i)] += dt * rounded.vy[static_cast<size_t>(i)];
      rounded.z[static_cast<size_t>(i)] += dt * rounded.vz[static_cast<size_t>(i)];
      WrapLikeReference(rounded.x[static_cast<size_t>(i)], origin.box[0]);
      WrapLikeReference(rounded.y[static_cast<size_t>(i)], origin.box[1]);
      WrapLikeReference(rounded.z[static_cast<size_t>(i)], origin.box[2]);
    }
    max_x = std::max(max_x, MaxAbs(full.x, rounded.x));
    max_x = std::max(max_x, MaxAbs(full.y, rounded.y));
    max_x = std::max(max_x, MaxAbs(full.z, rounded.z));
  }
  std::cout << "pos4 leapfrog 100 max |dx| " << max_x << "\n";
  EXPECT_GT(max_x, 1.0e-8);
  EXPECT_LT(max_x, 2.0e-7);
}

tilemd::ref::System FreeParticle() {
  tilemd::ref::System system;
  system.box[0] = 20.0;
  system.box[1] = 20.0;
  system.box[2] = 20.0;
  system.shift = tilemd::ref::Shift::None;
  system.pairs.push_back({0, 0, 1.0, 1.0, 2.5});
  system.type.push_back(0);
  system.x.push_back(1.0);
  system.y.push_back(2.0);
  system.z.push_back(3.0);
  return system;
}

TEST(Integrator, PhiloxWordsStayFixed) {
  const tilemd::ref::PhiloxWord zero = tilemd::ref::Philox4x32_10(0u, 0u, 0u, 0u, 0u, 0u);
  EXPECT_EQ(zero.w[0], 0x6627e8d5u);
  EXPECT_EQ(zero.w[1], 0xe169c58du);
  EXPECT_EQ(zero.w[2], 0xbc57ac4cu);
  EXPECT_EQ(zero.w[3], 0x9b00dbd8u);
  const tilemd::ref::PhiloxWord other = tilemd::ref::Philox4x32_10(1u, 2u, 3u, 4u, 5u, 6u);
  EXPECT_EQ(other.w[0], 0xc0c839bcu);
  EXPECT_EQ(other.w[1], 0x889c87c5u);
  EXPECT_EQ(other.w[2], 0x61986739u);
  EXPECT_EQ(other.w[3], 0x2d4623d0u);
}

TEST(Integrator, ZeroFrictionMatchesVerlet) {
  tilemd::ref::System origin = SmallMelt();
  const int n = 40;
  std::vector<double> vx(static_cast<size_t>(n), 0.1);
  std::vector<double> vy(static_cast<size_t>(n), -0.2);
  std::vector<double> vz(static_cast<size_t>(n), 0.05);
  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  tilemd::ref::System verlet_system = origin;
  tilemd::ref::System langevin_system = origin;
  tilemd::ref::Dynamics verlet = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Dynamics langevin = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Langevin thermo;
  thermo.gamma = 0.0;
  thermo.temperature = 1.5;
  thermo.seed = 7;
  for (int step = 0; step < 20; ++step) {
    tilemd::ref::VelocityVerletStep(verlet_system, verlet, dt, mass);
    tilemd::ref::LangevinBaoabStep(langevin_system, langevin, dt, mass, thermo,
                                   static_cast<std::uint64_t>(step));
  }
  EXPECT_EQ(MaxAbs(verlet.x, langevin.x), 0.0);
  EXPECT_EQ(MaxAbs(verlet.vx, langevin.vx), 0.0);
}

TEST(Integrator, ZeroTemperatureDampsTheFreeParticle) {
  tilemd::ref::System system = FreeParticle();
  std::vector<double> vx(1, 1.0);
  std::vector<double> vy(1, -0.5);
  std::vector<double> vz(1, 0.25);
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, vx, vy, vz);
  tilemd::ref::Langevin thermo;
  thermo.gamma = 2.0;
  thermo.temperature = 0.0;
  thermo.seed = 3;
  constexpr double dt = 0.005;
  constexpr int steps = 40;
  for (int step = 0; step < steps; ++step) {
    tilemd::ref::LangevinBaoabStep(system, state, dt, 1.0, thermo,
                                   static_cast<std::uint64_t>(step));
  }
  const double decay = std::exp(-thermo.gamma * dt * steps);
  EXPECT_NEAR(state.vx[0], decay, 1.0e-12);
  EXPECT_NEAR(state.vy[0], -0.5 * decay, 1.0e-12);
  EXPECT_NEAR(state.vz[0], 0.25 * decay, 1.0e-12);
}

TEST(Integrator, LangevinRepeatsForTheSameSeed) {
  tilemd::ref::System origin = FreeParticle();
  origin.type.push_back(0);
  origin.x.push_back(10.0);
  origin.y.push_back(10.0);
  origin.z.push_back(10.0);
  std::vector<double> vx(2, 0.2);
  std::vector<double> vy(2, 0.0);
  std::vector<double> vz(2, -0.1);
  tilemd::ref::Langevin thermo;
  thermo.gamma = 1.0;
  thermo.temperature = 1.2;
  thermo.seed = 11;
  tilemd::ref::System first_system = origin;
  tilemd::ref::System second_system = origin;
  tilemd::ref::System other_system = origin;
  tilemd::ref::Dynamics first = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Dynamics second = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::Dynamics other = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  for (int step = 0; step < 8; ++step) {
    const auto key = static_cast<std::uint64_t>(step);
    tilemd::ref::LangevinBaoabStep(first_system, first, 0.005, 1.0, thermo, key);
    tilemd::ref::LangevinBaoabStep(second_system, second, 0.005, 1.0, thermo, key);
    tilemd::ref::Langevin other_thermo = thermo;
    other_thermo.seed = 12;
    tilemd::ref::LangevinBaoabStep(other_system, other, 0.005, 1.0, other_thermo, key);
  }
  EXPECT_EQ(MaxAbs(first.vx, second.vx), 0.0);
  EXPECT_EQ(MaxAbs(first.x, second.x), 0.0);
  EXPECT_GT(MaxAbs(first.vx, other.vx), 0.0);
  EXPECT_NE(first.vx[0], first.vx[1]);
}
