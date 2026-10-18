#include "neighbor/cluster_list.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lj_reference.hpp"

namespace {

bool ContainsPair(const tilemd::ClusterPairs& pairs, int ci, int cj, int shift) {
  for (int k = pairs.ci_start[static_cast<size_t>(ci)];
       k < pairs.ci_start[static_cast<size_t>(ci) + 1]; ++k) {
    if (pairs.cj[static_cast<size_t>(k)] == cj && pairs.shift[static_cast<size_t>(k)] == shift) {
      return true;
    }
  }
  return false;
}

int ClusterOf(const tilemd::ClusterGrid& grid, int gid) {
  for (int slot = 0; slot < static_cast<int>(grid.gid.size()); ++slot) {
    if (grid.gid[static_cast<size_t>(slot)] == gid) return slot / tilemd::kClusterSize;
  }
  return -1;
}

}  // namespace

TEST(ClusterList, PadsAndSortsEachColumn) {
  const double box[3] = {10.0, 10.0, 10.0};
  const double x[3] = {1.0, 1.2, 1.1};
  const double y[3] = {1.0, 1.0, 1.0};
  const double z[3] = {3.0, 1.0, 1.0};
  const int type[3] = {0, 0, 0};
  const tilemd::ClusterGrid grid = tilemd::BuildClusterGrid(box, x, y, z, type, 3, 2.5, 1, 1);
  ASSERT_EQ(grid.ncluster, 1);
  ASSERT_EQ(grid.gid.size(), 8u);
  // Equal z keeps insertion order: gid 1 was inserted before gid 2.
  EXPECT_EQ(grid.gid[0], 1);
  EXPECT_EQ(grid.gid[1], 2);
  EXPECT_EQ(grid.gid[2], 0);
  for (int slot = 3; slot < 8; ++slot) {
    EXPECT_EQ(grid.gid[static_cast<size_t>(slot)], -1);
    EXPECT_EQ(grid.type[static_cast<size_t>(slot)], tilemd::kMaskType);
  }
  EXPECT_DOUBLE_EQ(grid.lo_z[0], 1.0);
  EXPECT_DOUBLE_EQ(grid.hi_z[0], 3.0);
  EXPECT_DOUBLE_EQ(grid.lo_x[0], 1.0);
  EXPECT_DOUBLE_EQ(grid.hi_x[0], 1.2);
}

TEST(ClusterList, KeepsAPeriodicImageAndDropsADistantColumn) {
  const double box[3] = {20.0, 10.0, 10.0};
  const double x[3] = {0.2, 19.7, 10.0};
  const double y[3] = {5.0, 5.0, 5.0};
  const double z[3] = {5.0, 5.0, 5.0};
  const int type[3] = {0, 0, 0};
  const tilemd::ClusterGrid grid = tilemd::BuildClusterGrid(box, x, y, z, type, 3, 2.5, 2, 1);
  ASSERT_EQ(grid.ncolumn[0], 2);
  const int left = ClusterOf(grid, 0);
  const int right = ClusterOf(grid, 1);
  const int middle = ClusterOf(grid, 2);
  ASSERT_GE(left, 0);
  ASSERT_GE(right, 0);
  ASSERT_GE(middle, 0);
  EXPECT_NE(left, middle);
  const tilemd::ClusterPairs pairs = tilemd::BuildClusterPairs(grid);
  EXPECT_TRUE(ContainsPair(pairs, left, right, tilemd::ShiftIndex(-1, 0, 0)));
  EXPECT_FALSE(ContainsPair(pairs, left, middle, tilemd::ShiftIndex(0, 0, 0)));
  EXPECT_EQ(pairs.ci_start.back(), static_cast<int>(pairs.cj.size()));
  EXPECT_EQ(pairs.kept, static_cast<int>(pairs.cj.size()));
}

TEST(ClusterList, EveryRealAtomAppearsOnce) {
  const double box[3] = {8.0, 8.0, 8.0};
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<int> type;
  for (int i = 0; i < 40; ++i) {
    x.push_back(0.3 + (i % 5) * 1.5);
    y.push_back(0.4 + ((i / 5) % 4) * 1.7);
    z.push_back(0.2 + (i % 7) * 0.9);
    type.push_back(0);
  }
  const tilemd::ClusterGrid grid =
      tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(), type.data(), 40, 2.8);
  std::vector<int> seen(40, 0);
  for (int gid : grid.gid) {
    if (gid < 0) continue;
    ASSERT_LT(gid, 40);
    ++seen[static_cast<size_t>(gid)];
  }
  for (int count : seen) EXPECT_EQ(count, 1);
  EXPECT_EQ(static_cast<int>(grid.gid.size()) % tilemd::kClusterSize, 0);

  tilemd::ref::System system;
  system.box[0] = box[0];
  system.box[1] = box[1];
  system.box[2] = box[2];
  system.shift = tilemd::ref::Shift::Force;
  system.type = type;
  system.x = x;
  system.y = y;
  system.z = z;
  tilemd::ref::PairCoeff coeff;
  system.pairs = {coeff};
  const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
  tilemd::LjPair pair;
  const tilemd::TileForces got = tilemd::ComputeTileForces(
      box, x.data(), y.data(), z.data(), type.data(), 40, &pair, 1, 0.3, tilemd::LjShift::Force);
  for (int i = 0; i < 40; ++i) {
    EXPECT_NEAR(got.fx[static_cast<size_t>(i)], oracle.fx[static_cast<size_t>(i)], 1e-9);
    EXPECT_NEAR(got.fy[static_cast<size_t>(i)], oracle.fy[static_cast<size_t>(i)], 1e-9);
    EXPECT_NEAR(got.fz[static_cast<size_t>(i)], oracle.fz[static_cast<size_t>(i)], 1e-9);
  }
  EXPECT_NEAR(got.potential_energy, oracle.potential_energy, 1e-8);
  EXPECT_GT(got.pairs_inside, 0);
  EXPECT_GE(got.pairs_visited, got.pairs_inside);
  for (int column = 0; column < grid.ncolumn[0] * grid.ncolumn[1]; ++column) {
    const int begin = grid.column_cluster[static_cast<size_t>(column)] * tilemd::kClusterSize;
    const int count = grid.column_ncluster[static_cast<size_t>(column)] * tilemd::kClusterSize;
    for (int slot = begin + 1; slot < begin + count; ++slot) {
      if (grid.gid[static_cast<size_t>(slot)] < 0) continue;
      if (grid.gid[static_cast<size_t>(slot - 1)] < 0) continue;
      EXPECT_LE(grid.z[static_cast<size_t>(slot - 1)], grid.z[static_cast<size_t>(slot)]);
    }
  }
}

TEST(ClusterList, ReportsTileEfficiency) {
  auto efficiency = [](const std::vector<double>& x, const std::vector<double>& y,
                       const std::vector<double>& z, const double* box) {
    const int n = static_cast<int>(x.size());
    std::vector<int> type(static_cast<size_t>(n), 0);
    tilemd::LjPair pair;
    const tilemd::TileForces got = tilemd::ComputeTileForces(
        box, x.data(), y.data(), z.data(), type.data(), n, &pair, 1, 0.3, tilemd::LjShift::None);
    EXPECT_GT(got.pairs_visited, 0);
    EXPECT_LE(got.pairs_inside, got.pairs_visited);
    const double ratio =
        static_cast<double>(got.pairs_inside) / static_cast<double>(got.pairs_visited);
    std::cout << "tile efficiency n=" << n << " inside=" << got.pairs_inside
              << " visited=" << got.pairs_visited << " ratio=" << ratio << "\n";
    return ratio;
  };

  const tilemd::ref::System melt =
      tilemd::ref::ReadExtxyz(std::string(TILEMD_SOURCE_DIR) + "/tests/fixtures/melt_4000.extxyz");
  const double melt_ratio = efficiency(melt.x, melt.y, melt.z, melt.box);
  EXPECT_GT(melt_ratio, 0.08);
  EXPECT_LT(melt_ratio, 0.13);

  std::ifstream dump(std::string(TILEMD_SOURCE_DIR) + "/tests/validation/lammps_ref/init_32k.dump");
  ASSERT_TRUE(dump) << "32k initial dump";
  std::string line;
  double box[3] = {};
  int n = -1;
  while (std::getline(dump, line)) {
    if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) {
      dump >> n;
      std::getline(dump, line);
    } else if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) {
      double lo = 0.0;
      double hi = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        dump >> lo >> hi;
        box[axis] = hi - lo;
        std::getline(dump, line);
      }
    } else if (line.rfind("ITEM: ATOMS", 0) == 0) {
      break;
    }
  }
  ASSERT_EQ(n, 32000);
  std::vector<double> x(32000);
  std::vector<double> y(32000);
  std::vector<double> z(32000);
  for (int row = 0; row < n; ++row) {
    int id = 0;
    int type = 0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    dump >> id >> type >> x[static_cast<size_t>(id - 1)] >> y[static_cast<size_t>(id - 1)] >>
        z[static_cast<size_t>(id - 1)] >> vx >> vy >> vz;
  }
  ASSERT_TRUE(dump);
  const double melt32_ratio = efficiency(x, y, z, box);
  EXPECT_GT(melt32_ratio, 0.07);
  EXPECT_LT(melt32_ratio, 0.12);
}

TEST(ClusterList, TileForcesMatchTheOracleOnFixtures) {
  const char* names[] = {"fcc_500", "boundary_500", "cutoff_500", "twotype_2000", "melt_4000"};
  const std::string dir = std::string(TILEMD_SOURCE_DIR) + "/tests/fixtures/";
  for (const char* name : names) {
    tilemd::ref::System system = tilemd::ref::ReadExtxyz(dir + name + ".extxyz");
    tilemd::ref::ReadSystemHeader(dir + name + ".json", system);
    const int n = tilemd::ref::NumAtoms(system);
    std::vector<tilemd::LjPair> pairs;
    for (const tilemd::ref::PairCoeff& coeff : system.pairs) {
      tilemd::LjPair pair;
      pair.type_i = coeff.type_i;
      pair.type_j = coeff.type_j;
      pair.epsilon = coeff.epsilon;
      pair.sigma = coeff.sigma;
      pair.cutoff = coeff.cutoff;
      pairs.push_back(pair);
    }
    tilemd::LjShift shift = tilemd::LjShift::None;
    if (system.shift == tilemd::ref::Shift::Energy) shift = tilemd::LjShift::Energy;
    if (system.shift == tilemd::ref::Shift::Force) shift = tilemd::LjShift::Force;
    const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
    const tilemd::TileForces got = tilemd::ComputeTileForces(
        system.box, system.x.data(), system.y.data(), system.z.data(), system.type.data(), n,
        pairs.data(), static_cast<int>(pairs.size()), 0.3, shift);
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
    const double rms = std::sqrt(sum_sq / static_cast<double>(n));
    double virial_delta = 0.0;
    for (int k = 0; k < 6; ++k) {
      virial_delta = std::max(virial_delta, std::abs(got.virial[k] - oracle.virial[k]));
    }
    std::cout << name << " pairs_inside " << got.pairs_inside << " force_max " << max_delta
              << " rms " << rms << " energy_delta "
              << std::abs(got.potential_energy - oracle.potential_energy) << " virial_delta "
              << virial_delta << "\n";
    if (rms < 1.0e-8) {
      EXPECT_LT(max_delta, 1.0e-8) << name;
    } else {
      EXPECT_LT(max_delta / rms, 1.0e-10) << name;
    }
    EXPECT_NEAR(got.potential_energy, oracle.potential_energy,
                1.0e-8 * std::max(1.0, std::abs(oracle.potential_energy)))
        << name;
    double virial_scale = 1.0;
    for (int k = 0; k < 6; ++k) {
      virial_scale = std::max(virial_scale, std::abs(oracle.virial[k]));
    }
    for (int k = 0; k < 6; ++k) {
      EXPECT_NEAR(got.virial[k], oracle.virial[k], 1.0e-8 * virial_scale)
          << name << " virial " << k;
    }
  }
}
