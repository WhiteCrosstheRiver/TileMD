#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include "lj_reference.hpp"

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
