#include "force/fp32_walk.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "io/lammps_dump.hpp"
#include "lj_reference.hpp"
#include "md/fused_atom.hpp"
#include "neighbor/cell_list.hpp"

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

std::vector<int> Row(const tilemd::CellList& list, int i, int n) {
  std::vector<int> row;
  for (int k = 0; k < list.nn[i]; ++k) row.push_back(list.nl[static_cast<std::size_t>(k) * n + i]);
  return row;
}

TEST(Fp32Walk, CellListMatchesAllPairs) {
  std::vector<double> x, y, z;
  double length = 0;
  MakeFcc(6, &x, &y, &z, &length);
  const int n = static_cast<int>(x.size());
  for (int i = 0; i < n; ++i) {
    x[i] += 0.07 * std::sin(0.7 * i);
    y[i] += 0.05 * std::cos(0.3 * i);
    z[i] += 0.04 * std::sin(0.5 * i + 1.0);
    x[i] -= length * std::floor(x[i] / length);
    y[i] -= length * std::floor(y[i] / length);
    z[i] -= length * std::floor(z[i] / length);
  }
  const double box[3] = {length, length, length};
  const double reach = 2.8;
  const tilemd::CellList cells = tilemd::BuildCellList(box, x.data(), y.data(), z.data(), n, reach);
  std::vector<int> nn, nl;
  FillNeighbors(x, y, z, length, reach, 256, &nn, &nl);
  for (int i = 0; i < n; ++i) {
    std::vector<int> brute;
    for (int k = 0; k < nn[i]; ++k) brute.push_back(nl[static_cast<std::size_t>(k) * n + i]);
    std::sort(brute.begin(), brute.end());
    EXPECT_EQ(Row(cells, i, n), brute) << i;
  }
}

TEST(Fp32Walk, MeltDumpHasTheBaselineAtoms) {
  const std::string path =
      std::string(TILEMD_SOURCE_DIR) + "/tests/validation/lammps_ref/init_32k.dump";
  const tilemd::DumpFrame frame = tilemd::ReadLammpsDump(path);
  EXPECT_EQ(frame.x.size(), 32000u);
  EXPECT_NEAR(frame.box[0], 33.591923827650149, 1e-12);
  EXPECT_NEAR(frame.x[0], 0.0, 0.0);
  EXPECT_NEAR(frame.vx[0], 0.55710519079511855, 1e-15);
}

double MeanNeighborGap(const tilemd::CellList& list, int n) {
  double sum = 0.0;
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < list.nn[i]; ++k) {
      sum += std::abs(list.nl[static_cast<std::size_t>(k) * n + i] - i);
      ++count;
    }
  }
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

TEST(Fp32Walk, MortonOrderPullsNeighborsTogether) {
  std::vector<double> x, y, z;
  double length = 0;
  MakeFcc(6, &x, &y, &z, &length);
  const int n = static_cast<int>(x.size());
  std::vector<int> shuffled(n);
  std::iota(shuffled.begin(), shuffled.end(), 0);
  std::mt19937 rng(20261019);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  tilemd::ApplyOrder(&x, shuffled);
  tilemd::ApplyOrder(&y, shuffled);
  tilemd::ApplyOrder(&z, shuffled);
  const double box[3] = {length, length, length};
  const tilemd::CellList before = tilemd::BuildCellList(box, x.data(), y.data(), z.data(), n, 2.8);
  const std::vector<int> order = tilemd::MortonOrder(box, x.data(), y.data(), z.data(), n, 2.8);
  tilemd::ApplyOrder(&x, order);
  tilemd::ApplyOrder(&y, order);
  tilemd::ApplyOrder(&z, order);
  const tilemd::CellList after = tilemd::BuildCellList(box, x.data(), y.data(), z.data(), n, 2.8);
  EXPECT_LT(MeanNeighborGap(after, n), MeanNeighborGap(before, n));
  std::vector<int> nn, nl;
  FillNeighbors(x, y, z, length, 2.8, 256, &nn, &nl);
  for (int i = 0; i < n; ++i) {
    std::vector<int> brute;
    for (int k = 0; k < nn[i]; ++k) brute.push_back(nl[static_cast<std::size_t>(k) * n + i]);
    std::sort(brute.begin(), brute.end());
    EXPECT_EQ(Row(after, i, n), brute) << i;
  }
}

TEST(Fp32Walk, SkinRebuildsOnlyAfterHalfTheSkin) {
  std::vector<double> x, y, z;
  double length = 0;
  MakeFcc(4, &x, &y, &z, &length);
  const int n = static_cast<int>(x.size());
  std::vector<double> vx(n, 0.0), vy(n, 0.0), vz(n, 0.0);
  std::vector<int> ix(n, 0), iy(n, 0), iz(n, 0);
  const double box[3] = {length, length, length};
  const tilemd::Fp32SkinStats still =
      tilemd::HostFp32SkinSteps(box, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                                ix.data(), iy.data(), iz.data(), n, UnitLj(), 0.005, 1.0, 0.3, 5);
  EXPECT_EQ(still.rebuilds, 0);
  EXPECT_LT(still.max_disp2, 0.0225f);

  vx[0] = 40.0;
  const tilemd::Fp32SkinStats crossed =
      tilemd::HostFp32SkinSteps(box, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                                ix.data(), iy.data(), iz.data(), n, UnitLj(), 0.005, 1.0, 0.3, 1);
  EXPECT_EQ(crossed.rebuilds, 1);
  EXPECT_GT(crossed.max_disp2, 0.0225f);
  EXPECT_GE(x[0], 0.0);
  EXPECT_LT(x[0], length);
}

TEST(Fp32Walk, CellListSeesThePeriodicFace) {
  const double box[3] = {10, 10, 10};
  const double x[2] = {0.001, 9.999};
  const double y[2] = {1, 1};
  const double z[2] = {2, 2};
  const tilemd::CellList cells = tilemd::BuildCellList(box, x, y, z, 2, 0.5);
  ASSERT_EQ(cells.nn[0], 1);
  EXPECT_EQ(cells.nl[0], 1);
  EXPECT_EQ(cells.nl[1], 0);
}
