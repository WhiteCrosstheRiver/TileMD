#include "force/fp32_walk.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "lj_reference.hpp"
#include "md/fused_atom.hpp"

namespace {

double MinImage(double delta, double length) { return delta - length * std::rint(delta / length); }

void FillNeighbors(const std::vector<double>& x, const std::vector<double>& y,
                   const std::vector<double>& z, double length, double cutoff, int max_neigh,
                   std::vector<int>* nn, std::vector<int>* nl) {
  const int n = static_cast<int>(x.size());
  nn->assign(n, 0);
  nl->assign(static_cast<std::size_t>(max_neigh) * n, -1);
  const double rc2 = cutoff * cutoff;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const double dx = MinImage(x[i] - x[j], length);
      const double dy = MinImage(y[i] - y[j], length);
      const double dz = MinImage(z[i] - z[j], length);
      if (dx * dx + dy * dy + dz * dz >= rc2) continue;
      if ((*nn)[i] >= max_neigh) throw std::runtime_error("neighbor list overflow");
      (*nl)[static_cast<std::size_t>((*nn)[i]) * n + i] = j;
      ++(*nn)[i];
    }
  }
}

tilemd::Fp32Lj UnitLj() {
  tilemd::Fp32Lj pair;
  pair.sigma2 = 1.f;
  pair.eps4 = 4.f;
  pair.eps24 = 24.f;
  pair.rc = 2.5f;
  pair.rc2 = 6.25f;
  return pair;
}

void MakeFcc(int cells, std::vector<double>* x, std::vector<double>* y, std::vector<double>* z,
             double* length) {
  const double a = std::cbrt(4.0 / 0.8442);
  *length = cells * a;
  const double basis[4][3] = {
      {0, 0, 0}, {0, 0.5 * a, 0.5 * a}, {0.5 * a, 0, 0.5 * a}, {0.5 * a, 0.5 * a, 0}};
  for (int iz = 0; iz < cells; ++iz) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int ix = 0; ix < cells; ++ix) {
        for (const auto& site : basis) {
          x->push_back(ix * a + site[0]);
          y->push_back(iy * a + site[1]);
          z->push_back(iz * a + site[2]);
        }
      }
    }
  }
}

}  // namespace

TEST(Fp32Walk, OnePairMatchesTheOracle) {
  const double box[3] = {20, 20, 20};
  const double x[2] = {1.0, 2.2};
  const double y[2] = {3.0, 3.0};
  const double z[2] = {4.0, 4.0};
  const int nn[2] = {1, 1};
  const int nl[2] = {1, 0};
  const tilemd::Fp32Walk got = tilemd::HostFp32Force(box, x, y, z, nn, nl, 2, UnitLj());

  tilemd::ref::System system;
  system.box[0] = system.box[1] = system.box[2] = 20;
  system.type = {0, 0};
  system.x = {x[0], x[1]};
  system.y = {y[0], y[1]};
  system.z = {z[0], z[1]};
  tilemd::ref::PairCoeff coeff;
  system.pairs = {coeff};
  const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
  tilemd::ref::Observables fp32;
  fp32.fx = {got.fx[0], got.fx[1]};
  fp32.fy = {got.fy[0], got.fy[1]};
  fp32.fz = {got.fz[0], got.fz[1]};
  fp32.potential_energy = got.potential_energy;
  for (int c = 0; c < 6; ++c) fp32.virial[c] = got.virial[c];
  const tilemd::ref::T1v2 gate = tilemd::ref::CompareT1v2(fp32, oracle);
  EXPECT_TRUE(gate.pass) << gate.force_rms_rel << " " << gate.force_max_rel << " "
                         << gate.energy_rel << " " << gate.virial_rel;
}

TEST(Fp32Walk, FccMatchesT1v2) {
  std::vector<double> x, y, z;
  double length = 0;
  MakeFcc(6, &x, &y, &z, &length);
  const int n = static_cast<int>(x.size());
  // A perfect fcc has a vanishing net force, so rms(|F|) is roundoff and T1-v2
  // is meaningless. A fixed displacement leaves a real force.
  for (int i = 0; i < n; ++i) {
    x[i] += 0.07 * std::sin(0.7 * i);
    y[i] += 0.05 * std::cos(0.3 * i);
    z[i] += 0.04 * std::sin(0.5 * i + 1.0);
    x[i] -= length * std::floor(x[i] / length);
    y[i] -= length * std::floor(y[i] / length);
    z[i] -= length * std::floor(z[i] / length);
  }
  std::vector<int> nn, nl;
  FillNeighbors(x, y, z, length, 2.5, 128, &nn, &nl);
  const double box[3] = {length, length, length};
  const tilemd::Fp32Walk got =
      tilemd::HostFp32Force(box, x.data(), y.data(), z.data(), nn.data(), nl.data(), n, UnitLj());

  tilemd::ref::System system;
  system.box[0] = system.box[1] = system.box[2] = length;
  system.type.assign(n, 0);
  system.x = x;
  system.y = y;
  system.z = z;
  tilemd::ref::PairCoeff coeff;
  system.pairs = {coeff};
  const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
  tilemd::ref::Observables fp32;
  fp32.fx.assign(got.fx.begin(), got.fx.end());
  fp32.fy.assign(got.fy.begin(), got.fy.end());
  fp32.fz.assign(got.fz.begin(), got.fz.end());
  fp32.potential_energy = got.potential_energy;
  for (int c = 0; c < 6; ++c) fp32.virial[c] = got.virial[c];
  const tilemd::ref::T1v2 gate = tilemd::ref::CompareT1v2(fp32, oracle);
  EXPECT_TRUE(gate.pass) << gate.force_rms_rel << " " << gate.force_max_rel << " "
                         << gate.energy_rel << " " << gate.virial_rel;
}

TEST(Fp32Walk, KickDriftWrapsOneAxis) {
  double coordinate = 9.99;
  double velocity = 10.0;
  int image = 0;
  tilemd::KickDriftAxis(&coordinate, &velocity, 0.f, 0.005, 1.0, 10.0, &image);
  EXPECT_NEAR(velocity, 10.0, 0.0);
  EXPECT_NEAR(coordinate, 0.04, 1e-15);
  EXPECT_EQ(image, 1);
}

TEST(Fp32Walk, ShortNveStaysInTheBox) {
  std::vector<double> x, y, z;
  double length = 0;
  MakeFcc(4, &x, &y, &z, &length);
  const int n = static_cast<int>(x.size());
  std::vector<double> vx(n, 0.05), vy(n, -0.02), vz(n, 0.01);
  std::vector<int> ix(n, 0), iy(n, 0), iz(n, 0);
  const double box[3] = {length, length, length};
  const double dt = 0.005;
  for (int step = 0; step < 10; ++step) {
    std::vector<int> nn, nl;
    FillNeighbors(x, y, z, length, 2.5, 128, &nn, &nl);
    const tilemd::Fp32Walk force =
        tilemd::HostFp32Force(box, x.data(), y.data(), z.data(), nn.data(), nl.data(), n, UnitLj());
    for (int i = 0; i < n; ++i) {
      tilemd::KickDriftAxis(&x[i], &vx[i], force.fx[i], dt, 1.0, length, &ix[i]);
      tilemd::KickDriftAxis(&y[i], &vy[i], force.fy[i], dt, 1.0, length, &iy[i]);
      tilemd::KickDriftAxis(&z[i], &vz[i], force.fz[i], dt, 1.0, length, &iz[i]);
      EXPECT_GE(x[i], 0.0);
      EXPECT_LT(x[i], length);
      EXPECT_TRUE(std::isfinite(force.fx[i]));
    }
  }
}
