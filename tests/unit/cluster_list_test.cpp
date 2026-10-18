#include "neighbor/cluster_list.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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

TEST(ClusterList, ColumnChoiceMatchesTheGrid) {
  const double box[3] = {10.0, 20.0, 30.0};
  std::vector<double> x(64, 1.0);
  std::vector<double> y(64, 1.0);
  std::vector<double> z(64, 1.0);
  std::vector<int> type(64, 0);
  const tilemd::ClusterShape shape = tilemd::ChooseClusterColumns(64, box);
  const tilemd::ClusterGrid grid =
      tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(), type.data(), 64, 2.5);
  EXPECT_EQ(grid.ncolumn[0], shape.nx);
  EXPECT_EQ(grid.ncolumn[1], shape.ny);
  EXPECT_DOUBLE_EQ(grid.edge[0], shape.edge_x);
  EXPECT_DOUBLE_EQ(grid.edge[1], shape.edge_y);
}

namespace {

void ExpectDeviceOrder(const double* box, const double* x, const double* y, const double* z,
                       const int* type, int n, int nx = 0, int ny = 0) {
  const tilemd::ClusterGrid grid = tilemd::BuildClusterGrid(box, x, y, z, type, n, 1.0, nx, ny);
  const tilemd::ClusterSlots slots = tilemd::SlotsInDeviceOrder(box, x, y, z, type, n, nx, ny);
  EXPECT_EQ(slots.ncluster, grid.ncluster);
  EXPECT_EQ(slots.type, grid.type);
  EXPECT_EQ(slots.gid, grid.gid);
  for (int cluster = 0; cluster < grid.ncluster; ++cluster) {
    tilemd::ClusterBox bounds;
    bool any = false;
    for (int local = 0; local < tilemd::kClusterSize; ++local) {
      const int slot = cluster * tilemd::kClusterSize + local;
      if (grid.type[static_cast<size_t>(slot)] == tilemd::kMaskType) continue;
      const int atom = grid.gid[static_cast<size_t>(slot)];
      tilemd::IncludeInBox(&bounds, &any, tilemd::WrapCoord(x[atom], box[0]),
                           tilemd::WrapCoord(y[atom], box[1]), tilemd::WrapCoord(z[atom], box[2]));
    }
    EXPECT_DOUBLE_EQ(grid.lo_x[static_cast<size_t>(cluster)], bounds.lo_x);
    EXPECT_DOUBLE_EQ(grid.lo_y[static_cast<size_t>(cluster)], bounds.lo_y);
    EXPECT_DOUBLE_EQ(grid.lo_z[static_cast<size_t>(cluster)], bounds.lo_z);
    EXPECT_DOUBLE_EQ(grid.hi_x[static_cast<size_t>(cluster)], bounds.hi_x);
    EXPECT_DOUBLE_EQ(grid.hi_y[static_cast<size_t>(cluster)], bounds.hi_y);
    EXPECT_DOUBLE_EQ(grid.hi_z[static_cast<size_t>(cluster)], bounds.hi_z);
  }
}

}  // namespace

TEST(ClusterList, HalfSkinIsTheRebuildLine) {
  EXPECT_FALSE(tilemd::DisplacementNeedsRebuild(0.0225f, 0.0225f));
  EXPECT_FALSE(tilemd::DisplacementNeedsRebuild(0.0f, 0.0225f));
  EXPECT_TRUE(tilemd::DisplacementNeedsRebuild(0.0226f, 0.0225f));
  EXPECT_TRUE(tilemd::DisplacementNeedsRebuild(0.1f, 0.0f));
}

TEST(ClusterList, DeviceSortKeyMatchesTheHostGrid) {
  const double box[3] = {10.0, 10.0, 10.0};
  const int type3[3] = {0, 1, 2};
  const double same_x[3] = {1.0, 1.2, 1.4};
  const double same_y[3] = {1.0, 1.0, 1.0};
  const double same_z[3] = {2.0, 2.0, 4.0};
  ExpectDeviceOrder(box, same_x, same_y, same_z, type3, 3);

  const int type2[2] = {0, 1};
  const double wrap_x[2] = {1.0, 1.0};
  const double wrap_y[2] = {1.0, 1.0};
  const double wrap_z[2] = {-1.0, 1.0};
  ExpectDeviceOrder(box, wrap_x, wrap_y, wrap_z, type2, 2);
  const tilemd::ClusterSlots wrapped =
      tilemd::SlotsInDeviceOrder(box, wrap_x, wrap_y, wrap_z, type2, 2);
  ASSERT_GE(wrapped.gid.size(), 2u);
  EXPECT_EQ(wrapped.gid[0], 1);
  EXPECT_EQ(wrapped.gid[1], 0);
  const tilemd::ClusterGrid wrapped_grid =
      tilemd::BuildClusterGrid(box, wrap_x, wrap_y, wrap_z, type2, 2, 1.0);
  ASSERT_EQ(wrapped_grid.ncluster, 1);
  EXPECT_DOUBLE_EQ(wrapped_grid.lo_z[0], 1.0);
  EXPECT_DOUBLE_EQ(wrapped_grid.hi_z[0], 9.0);

  const double neg0 = -0.0;
  const double pos0 = 0.0;
  const double zero_x[3] = {1.0, 1.1, 1.2};
  const double zero_y[3] = {1.0, 1.0, 1.0};
  const double zero_z[3] = {neg0, pos0, 3.0};
  ExpectDeviceOrder(box, zero_x, zero_y, zero_z, type3, 3);

  const double col_x[2] = {-0.2, 0.2};
  const double col_y[2] = {1.0, 1.0};
  const double col_z[2] = {1.0, 2.0};
  ExpectDeviceOrder(box, col_x, col_y, col_z, type2, 2, 2, 1);

  const int n = 64;
  const double wide[3] = {8.0, 8.0, 8.0};
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n));
  std::vector<int> type(static_cast<size_t>(n), 0);
  int atom = 0;
  for (int iz = 0; iz < 4; ++iz) {
    for (int iy = 0; iy < 4; ++iy) {
      for (int ix = 0; ix < 4; ++ix) {
        x[static_cast<size_t>(atom)] = 0.5 + 2.0 * ix;
        y[static_cast<size_t>(atom)] = 0.5 + 2.0 * iy;
        z[static_cast<size_t>(atom)] = 0.5 + 2.0 * iz;
        ++atom;
      }
    }
  }
  ExpectDeviceOrder(wide, x.data(), y.data(), z.data(), type.data(), n);
}

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

  std::vector<int> type32(static_cast<size_t>(n), 0);
  const tilemd::ClusterGrid grid =
      tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(), type32.data(), n, 2.8);
  const tilemd::ClusterPairs list = tilemd::BuildClusterPairs(grid);
  long long kernel_bytes = static_cast<long long>(grid.ncluster + 1) * 4;
  for (int slot = 0; slot < static_cast<int>(grid.gid.size()); ++slot) {
    kernel_bytes += 8;
    if (grid.type[static_cast<size_t>(slot)] != tilemd::kMaskType &&
        grid.gid[static_cast<size_t>(slot)] >= 0) {
      kernel_bytes += 16;
    }
  }
  for (int pair = 0; pair < static_cast<int>(list.cj.size()); ++pair) {
    kernel_bytes += 8;
    const int other = list.cj[static_cast<size_t>(pair)];
    for (int local = 0; local < tilemd::kClusterSize; ++local) {
      const int slot = other * tilemd::kClusterSize + local;
      kernel_bytes += 8;
      if (grid.type[static_cast<size_t>(slot)] != tilemd::kMaskType &&
          grid.gid[static_cast<size_t>(slot)] >= 0) {
        kernel_bytes += 16;
      }
    }
  }
  // Owning atom: read and write x, v (96), images (24), xref and pos_next (32).
  kernel_bytes += static_cast<long long>(grid.n_real) * 152;
  const double bytes_per_atom =
      static_cast<double>(kernel_bytes) / static_cast<double>(grid.n_real);
  const double upload =
      (static_cast<double>(grid.gid.size()) * 8.0 + static_cast<double>(grid.ncluster + 1) * 4.0 +
       static_cast<double>(list.cj.size()) * 8.0) /
      static_cast<double>(grid.n_real);
  std::cout << std::scientific << std::setprecision(9) << "fused kernel bytes/atom "
            << bytes_per_atom << " list upload bytes/atom " << upload << " clusters "
            << grid.ncluster << " pairs " << list.cj.size() << " kernel_bytes " << kernel_bytes
            << "\n";
  EXPECT_EQ(grid.ncluster, 4264);
  EXPECT_EQ(static_cast<int>(list.cj.size()), 329108);
  EXPECT_EQ(kernel_bytes, 69219396LL);
  EXPECT_DOUBLE_EQ(bytes_per_atom, 2163.106125);
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

void ColumnCounts(int n, const double* box, int target, int* nx, int* ny) {
  const double columns = std::max(1.0, static_cast<double>(n) / static_cast<double>(target));
  const double aspect = box[0] / box[1];
  *nx = std::max(1, static_cast<int>(std::lround(std::sqrt(columns * aspect))));
  *ny = std::max(1, static_cast<int>(std::lround(columns / static_cast<double>(*nx))));
}

TEST(ClusterList, ColumnWidthKeepsTheForce) {
  const int targets[] = {32, 64, 128, 256};
  const char* names[] = {"fcc_500", "melt_4000"};
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
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
                oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
                oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(n));
    for (int target : targets) {
      int nx = 0;
      int ny = 0;
      ColumnCounts(n, system.box, target, &nx, &ny);
      const tilemd::TileForces got = tilemd::ComputeTileForces(
          system.box, system.x.data(), system.y.data(), system.z.data(), system.type.data(), n,
          pairs.data(), static_cast<int>(pairs.size()), 0.3, shift, nx, ny);
      double max_delta = 0.0;
      for (int i = 0; i < n; ++i) {
        const double ex = got.fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
        const double ey = got.fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
        const double ez = got.fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
        max_delta = std::max(max_delta, std::sqrt(ex * ex + ey * ey + ez * ez));
      }
      const double ratio =
          static_cast<double>(got.pairs_inside) / static_cast<double>(got.pairs_visited);
      std::cout << "columns " << name << " target " << target << " nx " << nx << " ny " << ny
                << " inside " << got.pairs_inside << " visited " << got.pairs_visited << " ratio "
                << ratio << " force_max " << max_delta << "\n";
      EXPECT_GT(got.pairs_visited, 0) << name << " target " << target;
      if (std::string(name) == "fcc_500") EXPECT_EQ(got.pairs_inside, 27000) << target;
      if (std::string(name) == "melt_4000") EXPECT_EQ(got.pairs_inside, 219324) << target;
      EXPECT_LT(ratio, 0.20) << name << " target " << target;
      if (std::string(name) == "melt_4000" && target == 256) EXPECT_LT(ratio, 0.09);
      if (rms < 1.0e-8) {
        EXPECT_LT(max_delta, 1.0e-8) << name << " target " << target;
      } else {
        EXPECT_LT(max_delta / rms, 1.0e-9) << name << " target " << target;
      }
    }
  }
}

float MaxDisp2(const std::vector<double>& x, const std::vector<double>& y,
               const std::vector<double>& z, const std::vector<float>& xref,
               const std::vector<float>& yref, const std::vector<float>& zref, const double* box) {
  float max_d = 0.f;
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    const float dx = static_cast<float>(x[static_cast<size_t>(i)]) - xref[static_cast<size_t>(i)];
    const float dy = static_cast<float>(y[static_cast<size_t>(i)]) - yref[static_cast<size_t>(i)];
    const float dz = static_cast<float>(z[static_cast<size_t>(i)]) - zref[static_cast<size_t>(i)];
    const float lx = static_cast<float>(box[0]);
    const float ly = static_cast<float>(box[1]);
    const float lz = static_cast<float>(box[2]);
    const float mx = dx - lx * rintf(dx / lx);
    const float my = dy - ly * rintf(dy / ly);
    const float mz = dz - lz * rintf(dz / lz);
    max_d = std::max(max_d, mx * mx + my * my + mz * mz);
  }
  return max_d;
}

void StoreRef(const std::vector<double>& x, const std::vector<double>& y,
              const std::vector<double>& z, std::vector<float>* xref, std::vector<float>* yref,
              std::vector<float>* zref) {
  xref->resize(x.size());
  yref->resize(y.size());
  zref->resize(z.size());
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    (*xref)[static_cast<size_t>(i)] = static_cast<float>(x[static_cast<size_t>(i)]);
    (*yref)[static_cast<size_t>(i)] = static_cast<float>(y[static_cast<size_t>(i)]);
    (*zref)[static_cast<size_t>(i)] = static_cast<float>(z[static_cast<size_t>(i)]);
  }
}

void LeapTile(std::vector<double>* x, std::vector<double>* y, std::vector<double>* z,
              std::vector<double>* vx, std::vector<double>* vy, std::vector<double>* vz,
              const tilemd::TileForces& force, double dt, double mass, const double* box) {
  for (int i = 0; i < static_cast<int>(x->size()); ++i) {
    (*vx)[static_cast<size_t>(i)] += dt * force.fx[static_cast<size_t>(i)] / mass;
    (*vy)[static_cast<size_t>(i)] += dt * force.fy[static_cast<size_t>(i)] / mass;
    (*vz)[static_cast<size_t>(i)] += dt * force.fz[static_cast<size_t>(i)] / mass;
    (*x)[static_cast<size_t>(i)] = tilemd::WrapCoord(
        (*x)[static_cast<size_t>(i)] + dt * (*vx)[static_cast<size_t>(i)], box[0]);
    (*y)[static_cast<size_t>(i)] = tilemd::WrapCoord(
        (*y)[static_cast<size_t>(i)] + dt * (*vy)[static_cast<size_t>(i)], box[1]);
    (*z)[static_cast<size_t>(i)] = tilemd::WrapCoord(
        (*z)[static_cast<size_t>(i)] + dt * (*vz)[static_cast<size_t>(i)], box[2]);
  }
}

double MaxForceDelta(const tilemd::TileForces& a, const tilemd::TileForces& b) {
  double max_delta = 0.0;
  for (int i = 0; i < static_cast<int>(a.fx.size()); ++i) {
    max_delta =
        std::max(max_delta, std::abs(a.fx[static_cast<size_t>(i)] - b.fx[static_cast<size_t>(i)]));
    max_delta =
        std::max(max_delta, std::abs(a.fy[static_cast<size_t>(i)] - b.fy[static_cast<size_t>(i)]));
    max_delta =
        std::max(max_delta, std::abs(a.fz[static_cast<size_t>(i)] - b.fz[static_cast<size_t>(i)]));
  }
  return max_delta;
}

TEST(ClusterList, HeldListSurvivesAWrapBeforeTheSkin) {
  const double box[3] = {10.0, 10.0, 10.0};
  const double skin = 0.3;
  const double dt = 0.01;
  const double mass = 1.0;
  const tilemd::LjPair pair;
  std::vector<int> type = {0, 0};
  std::vector<double> x = {0.02, 2.2};
  std::vector<double> y = {5.0, 5.0};
  std::vector<double> z = {5.0, 5.0};
  std::vector<double> vx = {-0.5, 0.5};
  std::vector<double> vy = {0.0, 0.0};
  std::vector<double> vz = {0.0, 0.0};
  const tilemd::TileForces initial = tilemd::ComputeTileForces(
      box, x.data(), y.data(), z.data(), type.data(), 2, &pair, 1, skin, tilemd::LjShift::None);
  for (int i = 0; i < 2; ++i) {
    vx[static_cast<size_t>(i)] -= 0.5 * dt * initial.fx[static_cast<size_t>(i)] / mass;
    vy[static_cast<size_t>(i)] -= 0.5 * dt * initial.fy[static_cast<size_t>(i)] / mass;
    vz[static_cast<size_t>(i)] -= 0.5 * dt * initial.fz[static_cast<size_t>(i)] / mass;
  }
  tilemd::ClusterGrid held = tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(),
                                                      type.data(), 2, pair.cutoff + skin);
  tilemd::ClusterPairs held_pairs = tilemd::BuildClusterPairs(held);
  std::vector<double> x_hold = x;
  std::vector<double> y_hold = y;
  std::vector<double> z_hold = z;
  std::vector<float> xref;
  std::vector<float> yref;
  std::vector<float> zref;
  StoreRef(x, y, z, &xref, &yref, &zref);
  const float trigger2 = static_cast<float>((skin * 0.5) * (skin * 0.5));
  bool wrapped = false;
  double max_force = 0.0;
  for (int step = 0; step < 20; ++step) {
    const float disp2 = MaxDisp2(x, y, z, xref, yref, zref, box);
    EXPECT_FALSE(tilemd::DisplacementNeedsRebuild(disp2, trigger2)) << step;
    const tilemd::TileForces rebuilt = tilemd::ComputeTileForces(
        box, x.data(), y.data(), z.data(), type.data(), 2, &pair, 1, skin, tilemd::LjShift::None);
    const tilemd::TileForces kept = tilemd::ComputeTileForcesOnList(
        held, held_pairs, x.data(), y.data(), z.data(), x_hold.data(), y_hold.data(), z_hold.data(),
        &pair, 1, tilemd::LjShift::None);
    max_force = std::max(max_force, MaxForceDelta(rebuilt, kept));
    const double x_before = x[0];
    LeapTile(&x, &y, &z, &vx, &vy, &vz, kept, dt, mass, box);
    if (x_before < 1.0 && x[0] > 5.0) {
      wrapped = true;
      const float after = MaxDisp2(x, y, z, xref, yref, zref, box);
      EXPECT_FALSE(tilemd::DisplacementNeedsRebuild(after, trigger2));
      break;
    }
  }
  std::cout << "wrap held-list max |dF| " << max_force << "\n";
  EXPECT_TRUE(wrapped);
  EXPECT_LT(max_force, 1.0e-12);
}

TEST(ClusterList, SkinRebuildMatchesEveryStepFor1000) {
  const double box[3] = {8.0, 8.0, 8.0};
  const double skin = 0.3;
  const double dt = 0.005;
  const double mass = 1.0;
  const int n = 40;
  const tilemd::LjPair pair;
  std::vector<int> type(static_cast<size_t>(n), 0);
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = 0.5 + (i % 5) * 1.5;
    y[static_cast<size_t>(i)] = 0.5 + ((i / 5) % 4) * 1.8;
    z[static_cast<size_t>(i)] = 0.6 + (i / 20) * 3.0;
  }
  std::mt19937 rng(11);
  std::normal_distribution<double> speed(0.0, 0.2);
  std::vector<double> vx(static_cast<size_t>(n));
  std::vector<double> vy(static_cast<size_t>(n));
  std::vector<double> vz(static_cast<size_t>(n));
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] = speed(rng);
    vy[static_cast<size_t>(i)] = speed(rng);
    vz[static_cast<size_t>(i)] = speed(rng);
    px += vx[static_cast<size_t>(i)];
    py += vy[static_cast<size_t>(i)];
    pz += vz[static_cast<size_t>(i)];
  }
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] -= px / static_cast<double>(n);
    vy[static_cast<size_t>(i)] -= py / static_cast<double>(n);
    vz[static_cast<size_t>(i)] -= pz / static_cast<double>(n);
  }
  const tilemd::TileForces initial = tilemd::ComputeTileForces(
      box, x.data(), y.data(), z.data(), type.data(), n, &pair, 1, skin, tilemd::LjShift::None);
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] -= 0.5 * dt * initial.fx[static_cast<size_t>(i)] / mass;
    vy[static_cast<size_t>(i)] -= 0.5 * dt * initial.fy[static_cast<size_t>(i)] / mass;
    vz[static_cast<size_t>(i)] -= 0.5 * dt * initial.fz[static_cast<size_t>(i)] / mass;
  }
  std::vector<double> x_every = x;
  std::vector<double> y_every = y;
  std::vector<double> z_every = z;
  std::vector<double> vx_every = vx;
  std::vector<double> vy_every = vy;
  std::vector<double> vz_every = vz;
  tilemd::ClusterGrid held = tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(),
                                                      type.data(), n, pair.cutoff + skin);
  tilemd::ClusterPairs held_pairs = tilemd::BuildClusterPairs(held);
  std::vector<double> x_hold = x;
  std::vector<double> y_hold = y;
  std::vector<double> z_hold = z;
  std::vector<float> xref;
  std::vector<float> yref;
  std::vector<float> zref;
  StoreRef(x, y, z, &xref, &yref, &zref);
  const float trigger2 = static_cast<float>((skin * 0.5) * (skin * 0.5));
  int rebuilds = 1;
  double max_x = 0.0;
  for (int step = 0; step < 1000; ++step) {
    const tilemd::TileForces every =
        tilemd::ComputeTileForces(box, x_every.data(), y_every.data(), z_every.data(), type.data(),
                                  n, &pair, 1, skin, tilemd::LjShift::None);
    const float disp2 = MaxDisp2(x, y, z, xref, yref, zref, box);
    if (tilemd::DisplacementNeedsRebuild(disp2, trigger2)) {
      held = tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(), type.data(), n,
                                      pair.cutoff + skin);
      held_pairs = tilemd::BuildClusterPairs(held);
      x_hold = x;
      y_hold = y;
      z_hold = z;
      StoreRef(x, y, z, &xref, &yref, &zref);
      ++rebuilds;
    }
    const tilemd::TileForces kept = tilemd::ComputeTileForcesOnList(
        held, held_pairs, x.data(), y.data(), z.data(), x_hold.data(), y_hold.data(), z_hold.data(),
        &pair, 1, tilemd::LjShift::None);
    LeapTile(&x_every, &y_every, &z_every, &vx_every, &vy_every, &vz_every, every, dt, mass, box);
    LeapTile(&x, &y, &z, &vx, &vy, &vz, kept, dt, mass, box);
    for (int i = 0; i < n; ++i) {
      max_x =
          std::max(max_x, std::abs(x[static_cast<size_t>(i)] - x_every[static_cast<size_t>(i)]));
      max_x =
          std::max(max_x, std::abs(y[static_cast<size_t>(i)] - y_every[static_cast<size_t>(i)]));
      max_x =
          std::max(max_x, std::abs(z[static_cast<size_t>(i)] - z_every[static_cast<size_t>(i)]));
    }
  }
  std::cout << "skin vs every-step rebuilds " << rebuilds << " max |dx| " << max_x << "\n";
  EXPECT_GT(rebuilds, 1);
  EXPECT_LT(rebuilds, 1000);
  EXPECT_LT(max_x, 1.0e-6);
}
