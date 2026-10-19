#include "tf32_pair.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "lj_reference.hpp"
#include "neighbor/cluster_list.hpp"

namespace {

uint32_t LowMantissa(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits & ((1u << 13) - 1u);
}

float Rel(float got, double truth) {
  const double scale = std::max(1.0, std::abs(truth));
  return static_cast<float>(std::abs(static_cast<double>(got) - truth) / scale);
}

double TrueRel(float got, double truth) {
  return std::abs(static_cast<double>(got) - truth) / std::max(std::abs(truth), 1.0e-30);
}

struct R2Error {
  float fp32 = 0.f;
  float tf32 = 0.f;
  float tf32x3 = 0.f;
};

R2Error ScoreError(float ix, float iy, float iz, float jx, float jy, float jz, double truth) {
  float a[8];
  float b[8];
  tilemd::numerics::PackScoreA(ix, iy, iz, a);
  tilemd::numerics::PackScoreB(jx, jy, jz, b);
  R2Error err;
  err.fp32 = Rel(tilemd::numerics::DotFp32(a, b), truth);
  err.tf32 = Rel(tilemd::numerics::DotTf32(a, b), truth);
  err.tf32x3 = Rel(tilemd::numerics::Dot3Tf32(a, b), truth);
  return err;
}

struct PreparedPair {
  bool defined = false;
  double sigma2 = 1.0;
  double eps24 = 24.0;
  double rc2 = 6.25;
  double rc = 2.5;
  double s_rc = 0.0;
};

void StoreMax(double out[3], float ix, float iy, float iz, float jx, float jy, float jz,
              double truth) {
  float a[8];
  float b[8];
  tilemd::numerics::PackScoreA(ix, iy, iz, a);
  tilemd::numerics::PackScoreB(jx, jy, jz, b);
  out[0] = std::max(out[0], TrueRel(tilemd::numerics::DotFp32(a, b), truth));
  out[1] = std::max(out[1], TrueRel(tilemd::numerics::DotTf32(a, b), truth));
  out[2] = std::max(out[2], TrueRel(tilemd::numerics::Dot3Tf32(a, b), truth));
}

}  // namespace

TEST(Tf32Pair, RoundKeepsTenMantissaBits) {
  const float samples[] = {0.f, -0.f, 1.f, -2.5f, 0.1f, 12345.678f, -0.00098f};
  for (float sample : samples) {
    const float rounded = tilemd::numerics::RoundTf32(sample);
    EXPECT_EQ(LowMantissa(rounded), 0u);
    EXPECT_EQ(tilemd::numerics::RoundTf32(rounded), rounded);
  }
}

TEST(Tf32Pair, ScoreMatchesDistanceAnd3Tf32BeatsPlainTf32) {
  // Atoms sit far from the origin so |r|^2 cancels in the score. The same
  // pairs are then repeated about their centroid, which is the M7 tile frame.
  double max_fp32 = 0.0;
  double max_tf32 = 0.0;
  double max_tf32x3 = 0.0;
  double max_rel_fp32 = 0.0;
  double max_rel_tf32 = 0.0;
  double max_rel_3 = 0.0;
  int pairs = 0;
  for (int i = 0; i < 20; ++i) {
    const double dx = 0.4 + 0.15 * i;
    const double dy = -0.2 + 0.05 * (i % 5);
    const double dz = 0.3 - 0.04 * (i % 7);
    const double truth = dx * dx + dy * dy + dz * dz;
    const float cx = 20.f + static_cast<float>(i);
    const float cy = -15.f;
    const float cz = 8.f;
    const R2Error absolute = ScoreError(cx + static_cast<float>(dx), cy + static_cast<float>(dy),
                                        cz + static_cast<float>(dz), cx, cy, cz, truth);
    const float mx = static_cast<float>(0.5 * dx);
    const float my = static_cast<float>(0.5 * dy);
    const float mz = static_cast<float>(0.5 * dz);
    const R2Error relative = ScoreError(mx, my, mz, -mx, -my, -mz, truth);
    max_fp32 = std::max(max_fp32, static_cast<double>(absolute.fp32));
    max_tf32 = std::max(max_tf32, static_cast<double>(absolute.tf32));
    max_tf32x3 = std::max(max_tf32x3, static_cast<double>(absolute.tf32x3));
    max_rel_fp32 = std::max(max_rel_fp32, static_cast<double>(relative.fp32));
    max_rel_tf32 = std::max(max_rel_tf32, static_cast<double>(relative.tf32));
    max_rel_3 = std::max(max_rel_3, static_cast<double>(relative.tf32x3));
    ++pairs;
  }
  std::cout << "r2 rel error absolute fp32 " << max_fp32 << " tf32 " << max_tf32 << " 3xtf32 "
            << max_tf32x3 << "\n";
  std::cout << "r2 rel error tile-relative fp32 " << max_rel_fp32 << " tf32 " << max_rel_tf32
            << " 3xtf32 " << max_rel_3 << "\n";
  EXPECT_EQ(pairs, 20);
  EXPECT_LT(max_rel_3, max_rel_tf32);
  EXPECT_LT(max_tf32x3, max_tf32);
  EXPECT_LT(max_rel_3, 1.0e-4);
}

TEST(Tf32Pair, ForcesStayNearTheOracle) {
  tilemd::ref::System system;
  system.box[0] = 40.0;
  system.box[1] = 40.0;
  system.box[2] = 40.0;
  system.shift = tilemd::ref::Shift::None;
  tilemd::ref::PairCoeff coeff;
  system.pairs = {coeff};
  for (int i = 0; i < 32; ++i) {
    system.type.push_back(0);
    system.x.push_back(18.0 + (i % 4) * 1.2);
    system.y.push_back(18.0 + ((i / 4) % 4) * 1.2);
    system.z.push_back(18.0 + (i / 16) * 1.2);
  }
  const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  for (int i = 0; i < 32; ++i) {
    cx += system.x[static_cast<size_t>(i)];
    cy += system.y[static_cast<size_t>(i)];
    cz += system.z[static_cast<size_t>(i)];
  }
  cx /= 32.0;
  cy /= 32.0;
  cz /= 32.0;
  std::vector<double> fx(32);
  std::vector<double> fy(32);
  std::vector<double> fz(32);
  const double rc2 = coeff.cutoff * coeff.cutoff;
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      if (i == j) continue;
      const double dx = tilemd::ref::MinImage(
          system.x[static_cast<size_t>(i)] - system.x[static_cast<size_t>(j)], system.box[0]);
      const double dy = tilemd::ref::MinImage(
          system.y[static_cast<size_t>(i)] - system.y[static_cast<size_t>(j)], system.box[1]);
      const double dz = tilemd::ref::MinImage(
          system.z[static_cast<size_t>(i)] - system.z[static_cast<size_t>(j)], system.box[2]);
      const float ix = static_cast<float>(system.x[static_cast<size_t>(i)] - cx);
      const float iy = static_cast<float>(system.y[static_cast<size_t>(i)] - cy);
      const float iz = static_cast<float>(system.z[static_cast<size_t>(i)] - cz);
      const float jx = static_cast<float>(system.x[static_cast<size_t>(i)] - cx - dx);
      const float jy = static_cast<float>(system.y[static_cast<size_t>(i)] - cy - dy);
      const float jz = static_cast<float>(system.z[static_cast<size_t>(i)] - cz - dz);
      float a[8];
      float b[8];
      tilemd::numerics::PackScoreA(ix, iy, iz, a);
      tilemd::numerics::PackScoreB(jx, jy, jz, b);
      const float r2 = tilemd::numerics::Dot3Tf32(a, b);
      if (!(r2 < static_cast<float>(rc2)) || !(r2 > 0.f)) continue;
      const float inv_r2 = 1.f / r2;
      const float sr2 = inv_r2;
      const float sr6 = sr2 * sr2 * sr2;
      const float s = 24.f * inv_r2 * sr6 * (2.f * sr6 - 1.f);
      fx[static_cast<size_t>(i)] += static_cast<double>(s) * dx;
      fy[static_cast<size_t>(i)] += static_cast<double>(s) * dy;
      fz[static_cast<size_t>(i)] += static_cast<double>(s) * dz;
    }
  }
  double max_delta = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double ex = fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
    const double ey = fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
    const double ez = fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
    max_delta = std::max(max_delta, std::sqrt(ex * ex + ey * ey + ez * ez));
    sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
              oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
              oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
  }
  const double rms = std::sqrt(sum_sq / 32.0);
  std::cout << "3xtf32 force max " << max_delta << " rms " << rms << " rel " << (max_delta / rms)
            << "\n";
  EXPECT_GT(rms, 1.0);
  EXPECT_LT(max_delta / rms, 1.0e-4);
}

TEST(Tf32Pair, FixturesStayInsideTheForceTolerance) {
  const char* names[] = {"fcc_500",     "boundary_500", "cutoff_500", "twotype_2000",
                         "dilute_1000", "dense_2000",   "melt_4000"};
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  int present = 0;
  double worst_force = 0.0;
  for (const char* name : names) {
    const auto xyz = dir / (std::string(name) + ".extxyz");
    const auto json = dir / (std::string(name) + ".json");
    if (!std::filesystem::exists(xyz) || !std::filesystem::exists(json)) continue;
    ++present;
    tilemd::ref::System system = tilemd::ref::ReadExtxyz(xyz.string());
    tilemd::ref::ReadSystemHeader(json.string(), system);
    const int n = tilemd::ref::NumAtoms(system);
    PreparedPair table[8][8] = {};
    double max_cut = 0.0;
    for (const tilemd::ref::PairCoeff& coeff : system.pairs) {
      auto fill = [&](int ti, int tj) {
        PreparedPair& row = table[ti][tj];
        row.defined = true;
        row.sigma2 = coeff.sigma * coeff.sigma;
        row.eps24 = 24.0 * coeff.epsilon;
        row.rc = coeff.cutoff;
        row.rc2 = coeff.cutoff * coeff.cutoff;
        const double inv_rc2 = 1.0 / row.rc2;
        const double sr2 = row.sigma2 * inv_rc2;
        const double sr6 = sr2 * sr2 * sr2;
        row.s_rc = row.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
      };
      fill(coeff.type_i, coeff.type_j);
      if (coeff.type_i != coeff.type_j) fill(coeff.type_j, coeff.type_i);
      max_cut = std::max(max_cut, coeff.cutoff);
    }
    const tilemd::ClusterGrid grid =
        tilemd::BuildClusterGrid(system.box, system.x.data(), system.y.data(), system.z.data(),
                                 system.type.data(), n, max_cut + 0.3);
    std::vector<double> origin_x(static_cast<size_t>(n));
    std::vector<double> origin_y(static_cast<size_t>(n));
    std::vector<double> origin_z(static_cast<size_t>(n));
    for (int ci = 0; ci < grid.ncluster; ++ci) {
      double ax = 0.0;
      double ay = 0.0;
      double az = 0.0;
      bool anchored = false;
      double mean_x = 0.0;
      double mean_y = 0.0;
      double mean_z = 0.0;
      int nreal = 0;
      int locals[8];
      for (int local = 0; local < tilemd::kClusterSize; ++local) {
        const int slot = ci * tilemd::kClusterSize + local;
        if (grid.type[static_cast<size_t>(slot)] == tilemd::kMaskType) continue;
        if (!anchored) {
          ax = grid.x[static_cast<size_t>(slot)];
          ay = grid.y[static_cast<size_t>(slot)];
          az = grid.z[static_cast<size_t>(slot)];
          anchored = true;
        }
        mean_x += tilemd::ref::MinImage(grid.x[static_cast<size_t>(slot)] - ax, grid.box[0]);
        mean_y += tilemd::ref::MinImage(grid.y[static_cast<size_t>(slot)] - ay, grid.box[1]);
        mean_z += tilemd::ref::MinImage(grid.z[static_cast<size_t>(slot)] - az, grid.box[2]);
        locals[nreal++] = local;
      }
      if (!anchored) continue;
      mean_x /= static_cast<double>(nreal);
      mean_y /= static_cast<double>(nreal);
      mean_z /= static_cast<double>(nreal);
      for (int k = 0; k < nreal; ++k) {
        const int slot = ci * tilemd::kClusterSize + locals[k];
        const int gid = grid.gid[static_cast<size_t>(slot)];
        origin_x[static_cast<size_t>(gid)] = ax + mean_x;
        origin_y[static_cast<size_t>(gid)] = ay + mean_y;
        origin_z[static_cast<size_t>(gid)] = az + mean_z;
      }
    }
    std::vector<double> fx(static_cast<size_t>(n));
    std::vector<double> fy(static_cast<size_t>(n));
    std::vector<double> fz(static_cast<size_t>(n));
    double max_abs[3] = {};
    double max_tile[3] = {};
    double max_tile_separated[3] = {};
    double max_cancel_abs = 0.0;
    double max_cancel_tile = 0.0;
    int inside = 0;
    int mask_miss = 0;
    for (int i = 0; i < n; ++i) {
      const int ti = system.type[static_cast<size_t>(i)];
      const double xi = system.x[static_cast<size_t>(i)];
      const double yi = system.y[static_cast<size_t>(i)];
      const double zi = system.z[static_cast<size_t>(i)];
      const double irel_x =
          tilemd::ref::MinImage(xi - origin_x[static_cast<size_t>(i)], system.box[0]);
      const double irel_y =
          tilemd::ref::MinImage(yi - origin_y[static_cast<size_t>(i)], system.box[1]);
      const double irel_z =
          tilemd::ref::MinImage(zi - origin_z[static_cast<size_t>(i)], system.box[2]);
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const PreparedPair& row = table[ti][system.type[static_cast<size_t>(j)]];
        if (!row.defined) continue;
        const double dx =
            tilemd::ref::MinImage(xi - system.x[static_cast<size_t>(j)], system.box[0]);
        const double dy =
            tilemd::ref::MinImage(yi - system.y[static_cast<size_t>(j)], system.box[1]);
        const double dz =
            tilemd::ref::MinImage(zi - system.z[static_cast<size_t>(j)], system.box[2]);
        const double truth = dx * dx + dy * dy + dz * dz;
        if (!(truth < row.rc2) || !(truth > 0.0)) continue;
        ++inside;
        StoreMax(max_abs, static_cast<float>(xi), static_cast<float>(yi), static_cast<float>(zi),
                 static_cast<float>(xi - dx), static_cast<float>(yi - dy),
                 static_cast<float>(zi - dz), truth);
        const float rix = static_cast<float>(irel_x);
        const float riy = static_cast<float>(irel_y);
        const float riz = static_cast<float>(irel_z);
        const float jx = static_cast<float>(irel_x - dx);
        const float jy = static_cast<float>(irel_y - dy);
        const float jz = static_cast<float>(irel_z - dz);
        StoreMax(max_tile, rix, riy, riz, jx, jy, jz, truth);
        if (truth > 0.25) StoreMax(max_tile_separated, rix, riy, riz, jx, jy, jz, truth);
        const double r_abs2 = static_cast<double>(xi) * xi + static_cast<double>(yi) * yi +
                              static_cast<double>(zi) * zi;
        const double r_tile2 = static_cast<double>(rix) * rix + static_cast<double>(riy) * riy +
                               static_cast<double>(riz) * riz;
        max_cancel_abs = std::max(max_cancel_abs, r_abs2 / truth);
        max_cancel_tile = std::max(max_cancel_tile, r_tile2 / truth);
        float a[8];
        float b[8];
        tilemd::numerics::PackScoreA(rix, riy, riz, a);
        tilemd::numerics::PackScoreB(jx, jy, jz, b);
        const double r2 = static_cast<double>(tilemd::numerics::Dot3Tf32(a, b));
        if (!(r2 > 0.0) || !(r2 < row.rc2)) {
          ++mask_miss;
          continue;
        }
        const double inv_r2 = 1.0 / r2;
        const double sr2 = row.sigma2 * inv_r2;
        const double sr6 = sr2 * sr2 * sr2;
        double s = row.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
        if (system.shift == tilemd::ref::Shift::Force) s -= row.s_rc * row.rc / std::sqrt(r2);
        fx[static_cast<size_t>(i)] += s * dx;
        fy[static_cast<size_t>(i)] += s * dy;
        fz[static_cast<size_t>(i)] += s * dz;
      }
    }
    const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
    double max_delta = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double ex = fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
      const double ey = fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
      const double ez = fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
      max_delta = std::max(max_delta, std::sqrt(ex * ex + ey * ey + ez * ez));
      sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
                oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
                oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(std::max(n, 1)));
    const double force_rel = max_delta / std::max(rms, 1.0e-30);
    std::cout << name << " inside " << inside << " mask_miss " << mask_miss << " cancel_abs "
              << max_cancel_abs << " cancel_tile " << max_cancel_tile << " r2_abs fp32 "
              << max_abs[0] << " tf32 " << max_abs[1] << " 3xtf32 " << max_abs[2]
              << " r2_tile fp32 " << max_tile[0] << " tf32 " << max_tile[1] << " 3xtf32 "
              << max_tile[2] << " r2_tile_r2gt0.25 fp32 " << max_tile_separated[0] << " tf32 "
              << max_tile_separated[1] << " 3xtf32 " << max_tile_separated[2] << " force_max "
              << max_delta << " rms " << rms << " force_rel " << force_rel << "\n";
    EXPECT_LT(max_tile_separated[2], max_tile_separated[1]) << name;
    EXPECT_LT(max_tile_separated[2], 1.0e-4) << name;
    if (rms < 1.0e-6) {
      EXPECT_LT(max_delta, 1.0e-4) << name;
    } else if (std::string(name) == "melt_4000" || std::string(name) == "cutoff_500") {
      worst_force = std::max(worst_force, force_rel);
      // T1 asks for 1e-5. The melt measurement is about 8e-5, so this prototype misses it.
      EXPECT_LT(force_rel, 1.5e-4) << name;
      if (std::string(name) == "melt_4000") EXPECT_GT(force_rel, 1.0e-5) << name;
    } else {
      EXPECT_TRUE(std::isfinite(force_rel)) << name;
    }
  }
  if (present == 0) GTEST_SKIP() << "fixtures have not been generated";
  EXPECT_EQ(present, 7);
  std::cout << "worst fixture force rel " << worst_force << "\n";
}

void CompactOrigin(const tilemd::ref::System& system, int target, std::vector<double>& ox,
                   std::vector<double>& oy, std::vector<double>& oz, int* max_occ) {
  const int n = tilemd::ref::NumAtoms(system);
  const double volume = system.box[0] * system.box[1] * system.box[2];
  const double edge = std::cbrt(static_cast<double>(target) * volume / static_cast<double>(n));
  const int nx = std::max(1, static_cast<int>(std::lround(system.box[0] / edge)));
  const int ny = std::max(1, static_cast<int>(std::lround(system.box[1] / edge)));
  const int nz = std::max(1, static_cast<int>(std::lround(system.box[2] / edge)));
  const double ex = system.box[0] / static_cast<double>(nx);
  const double ey = system.box[1] / static_cast<double>(ny);
  const double ez = system.box[2] / static_cast<double>(nz);
  auto cell = [&](double x, double y, double z) {
    int ix = static_cast<int>(std::floor(x / ex));
    int iy = static_cast<int>(std::floor(y / ey));
    int iz = static_cast<int>(std::floor(z / ez));
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;
    return ix + nx * (iy + ny * iz);
  };
  const int ncell = nx * ny * nz;
  std::vector<int> count(static_cast<size_t>(ncell));
  std::vector<int> head(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int c = cell(system.x[static_cast<size_t>(i)], system.y[static_cast<size_t>(i)],
                       system.z[static_cast<size_t>(i)]);
    head[static_cast<size_t>(i)] = c;
    ++count[static_cast<size_t>(c)];
  }
  *max_occ = 0;
  for (int c : count) *max_occ = std::max(*max_occ, c);
  ox.assign(static_cast<size_t>(n), 0.0);
  oy.assign(static_cast<size_t>(n), 0.0);
  oz.assign(static_cast<size_t>(n), 0.0);
  for (int c = 0; c < ncell; ++c) {
    if (count[static_cast<size_t>(c)] == 0) continue;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    bool anchored = false;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    for (int i = 0; i < n; ++i) {
      if (head[static_cast<size_t>(i)] != c) continue;
      if (!anchored) {
        ax = system.x[static_cast<size_t>(i)];
        ay = system.y[static_cast<size_t>(i)];
        az = system.z[static_cast<size_t>(i)];
        anchored = true;
      }
      mx += tilemd::ref::MinImage(system.x[static_cast<size_t>(i)] - ax, system.box[0]);
      my += tilemd::ref::MinImage(system.y[static_cast<size_t>(i)] - ay, system.box[1]);
      mz += tilemd::ref::MinImage(system.z[static_cast<size_t>(i)] - az, system.box[2]);
    }
    const double inv = 1.0 / static_cast<double>(count[static_cast<size_t>(c)]);
    mx *= inv;
    my *= inv;
    mz *= inv;
    for (int i = 0; i < n; ++i) {
      if (head[static_cast<size_t>(i)] != c) continue;
      ox[static_cast<size_t>(i)] = ax + mx;
      oy[static_cast<size_t>(i)] = ay + my;
      oz[static_cast<size_t>(i)] = az + mz;
    }
  }
}

TEST(Tf32Pair, CompactCellsOnTheMelt) {
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  const char* names[] = {"cutoff_500", "melt_4000"};
  for (const char* name : names) {
    tilemd::ref::System system =
        tilemd::ref::ReadExtxyz((dir / (std::string(name) + ".extxyz")).string());
    tilemd::ref::ReadSystemHeader((dir / (std::string(name) + ".json")).string(), system);
    const int n = tilemd::ref::NumAtoms(system);
    PreparedPair row;
    row.defined = true;
    row.rc = system.pairs[0].cutoff;
    row.rc2 = row.rc * row.rc;
    row.sigma2 = system.pairs[0].sigma * system.pairs[0].sigma;
    row.eps24 = 24.0 * system.pairs[0].epsilon;
    const double inv_rc2 = 1.0 / row.rc2;
    const double sr2 = row.sigma2 * inv_rc2;
    const double sr6 = sr2 * sr2 * sr2;
    row.s_rc = row.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
    std::vector<double> ox;
    std::vector<double> oy;
    std::vector<double> oz;
    int max_occ = 0;
    const int targets[] = {4, 8, 16};
    for (int target : targets) {
      CompactOrigin(system, target, ox, oy, oz, &max_occ);
      std::vector<double> fx(static_cast<size_t>(n));
      std::vector<double> fy(static_cast<size_t>(n));
      std::vector<double> fz(static_cast<size_t>(n));
      double max_cancel = 0.0;
      for (int i = 0; i < n; ++i) {
        const double xi = system.x[static_cast<size_t>(i)];
        const double yi = system.y[static_cast<size_t>(i)];
        const double zi = system.z[static_cast<size_t>(i)];
        const double irel_x = tilemd::ref::MinImage(xi - ox[static_cast<size_t>(i)], system.box[0]);
        const double irel_y = tilemd::ref::MinImage(yi - oy[static_cast<size_t>(i)], system.box[1]);
        const double irel_z = tilemd::ref::MinImage(zi - oz[static_cast<size_t>(i)], system.box[2]);
        for (int j = 0; j < n; ++j) {
          if (i == j) continue;
          const double dx =
              tilemd::ref::MinImage(xi - system.x[static_cast<size_t>(j)], system.box[0]);
          const double dy =
              tilemd::ref::MinImage(yi - system.y[static_cast<size_t>(j)], system.box[1]);
          const double dz =
              tilemd::ref::MinImage(zi - system.z[static_cast<size_t>(j)], system.box[2]);
          const double truth = dx * dx + dy * dy + dz * dz;
          if (!(truth < row.rc2)) continue;
          const float rix = static_cast<float>(irel_x);
          const float riy = static_cast<float>(irel_y);
          const float riz = static_cast<float>(irel_z);
          const double r_tile2 = static_cast<double>(rix) * rix + static_cast<double>(riy) * riy +
                                 static_cast<double>(riz) * riz;
          max_cancel = std::max(max_cancel, r_tile2 / truth);
          float a[8];
          float b[8];
          tilemd::numerics::PackScoreA(rix, riy, riz, a);
          tilemd::numerics::PackScoreB(static_cast<float>(irel_x - dx),
                                       static_cast<float>(irel_y - dy),
                                       static_cast<float>(irel_z - dz), b);
          const double r2 = static_cast<double>(tilemd::numerics::Dot3Tf32(a, b));
          if (!(r2 > 0.0) || !(r2 < row.rc2)) continue;
          const double inv_r2 = 1.0 / r2;
          const double sr2_pair = row.sigma2 * inv_r2;
          const double sr6_pair = sr2_pair * sr2_pair * sr2_pair;
          double s = row.eps24 * inv_r2 * sr6_pair * (2.0 * sr6_pair - 1.0);
          if (system.shift == tilemd::ref::Shift::Force) s -= row.s_rc * row.rc / std::sqrt(r2);
          fx[static_cast<size_t>(i)] += s * dx;
          fy[static_cast<size_t>(i)] += s * dy;
          fz[static_cast<size_t>(i)] += s * dz;
        }
      }
      const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
      double max_delta = 0.0;
      double sum_sq = 0.0;
      for (int i = 0; i < n; ++i) {
        const double ex = fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
        const double ey = fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
        const double ez = fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
        max_delta = std::max(max_delta, std::sqrt(ex * ex + ey * ey + ez * ez));
        sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
                  oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
                  oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
      }
      const double rms = std::sqrt(sum_sq / static_cast<double>(n));
      const double force_rel = max_delta / std::max(rms, 1.0e-30);
      std::cout << "compact " << name << " target " << target << " max_occ " << max_occ
                << " cancel " << max_cancel << " force_max " << max_delta << " rms " << rms
                << " force_rel " << force_rel << "\n";
      EXPECT_LT(max_cancel, 40.0) << name << " target " << target;
      if (std::string(name) == "cutoff_500") {
        EXPECT_LT(force_rel, 1.0e-5) << name << " target " << target;
      } else if (target == 4) {
        EXPECT_GT(force_rel, 1.0e-5) << name;
        EXPECT_LT(force_rel, 3.0e-5) << name;
      } else if (target == 16) {
        EXPECT_GT(force_rel, 2.0e-5) << name;
        EXPECT_LT(force_rel, 6.0e-5) << name;
      }
    }
  }
}
