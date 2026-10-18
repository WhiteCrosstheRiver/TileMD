#include <gtest/gtest.h>

#include <cmath>
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
