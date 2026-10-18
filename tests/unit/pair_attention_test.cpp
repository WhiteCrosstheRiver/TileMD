#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "force/eam.hpp"
#include "force/lj_attention.hpp"
#include "force/tile_attention.hpp"
#include "lj_reference.hpp"

namespace {

double RelativeForce(const std::vector<double>& fx, const std::vector<double>& fy,
                     const std::vector<double>& fz, const std::vector<double>& gx,
                     const std::vector<double>& gy, const std::vector<double>& gz) {
  const int n = static_cast<int>(gx.size());
  double max_delta = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dx = fx[static_cast<size_t>(i)] - gx[static_cast<size_t>(i)];
    const double dy = fy[static_cast<size_t>(i)] - gy[static_cast<size_t>(i)];
    const double dz = fz[static_cast<size_t>(i)] - gz[static_cast<size_t>(i)];
    max_delta = std::max(max_delta, std::sqrt(dx * dx + dy * dy + dz * dz));
    sum_sq += gx[static_cast<size_t>(i)] * gx[static_cast<size_t>(i)] +
              gy[static_cast<size_t>(i)] * gy[static_cast<size_t>(i)] +
              gz[static_cast<size_t>(i)] * gz[static_cast<size_t>(i)];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(n));
  return max_delta / std::max(rms, 1.0e-30);
}

tilemd::AttentionShift ToShift(tilemd::ref::Shift shift) {
  if (shift == tilemd::ref::Shift::Energy) return tilemd::AttentionShift::Energy;
  if (shift == tilemd::ref::Shift::Force) return tilemd::AttentionShift::Force;
  return tilemd::AttentionShift::None;
}

void ExpectLj(const std::string& name, bool tiles) {
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  tilemd::ref::System system = tilemd::ref::ReadExtxyz((dir / (name + ".extxyz")).string());
  tilemd::ref::ReadSystemHeader((dir / (name + ".json")).string(), system);
  std::vector<tilemd::AttentionPair> pairs;
  for (const tilemd::ref::PairCoeff& coeff : system.pairs) {
    tilemd::AttentionPair pair;
    pair.type_i = coeff.type_i;
    pair.type_j = coeff.type_j;
    pair.epsilon = coeff.epsilon;
    pair.sigma = coeff.sigma;
    pair.cutoff = coeff.cutoff;
    pairs.push_back(pair);
  }
  tilemd::LjAttention potential(pairs.data(), static_cast<int>(pairs.size()),
                                ToShift(system.shift));
  const int n_atoms = tilemd::ref::NumAtoms(system);
  const tilemd::AttentionResult got =
      tiles
          ? tilemd::EvaluateAttentionTiles(potential, system.box, system.x.data(), system.y.data(),
                                           system.z.data(), system.type.data(), n_atoms, 0.3)
          : tilemd::EvaluateAttention(potential, system.box, system.x.data(), system.y.data(),
                                      system.z.data(), system.type.data(), n_atoms);
  const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
  double max_delta = 0.0;
  double sum_sq = 0.0;
  const int n = tilemd::ref::NumAtoms(system);
  for (int i = 0; i < n; ++i) {
    const double dx = got.fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
    const double dy = got.fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
    const double dz = got.fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
    max_delta = std::max(max_delta, std::sqrt(dx * dx + dy * dy + dz * dz));
    sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
              oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
              oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(n));
  if (rms < 1.0e-8) {
    EXPECT_LT(max_delta, 1.0e-8) << name;
  } else {
    EXPECT_LT(max_delta / rms, 1.0e-10) << name;
  }
  EXPECT_NEAR(got.potential_energy, oracle.potential_energy,
              1.0e-10 * (1.0 + std::abs(oracle.potential_energy)))
      << name;
  double virial_scale = 1.0;
  for (int k = 0; k < 6; ++k) {
    virial_scale = std::max(virial_scale, std::abs(oracle.virial[k]));
  }
  for (int k = 0; k < 6; ++k) {
    EXPECT_NEAR(got.virial[k], oracle.virial[k], 1.0e-9 * virial_scale) << name << " virial " << k;
  }
}

}  // namespace

TEST(PairAttention, LennardJonesMatchesTheOracle) {
  const char* names[] = {"fcc_500", "boundary_500", "cutoff_500", "twotype_2000", "melt_4000"};
  for (const char* name : names) ExpectLj(name, false);
}

TEST(PairAttention, TileListMatchesTheOracle) {
  const char* names[] = {"fcc_500", "boundary_500", "cutoff_500", "twotype_2000", "melt_4000"};
  for (const char* name : names) ExpectLj(name, true);
}

TEST(PairAttention, EamForceMatchesEnergyDerivative) {
  const std::filesystem::path table =
      std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "potentials" / "Cu_u3.eam";
  if (!std::filesystem::exists(table)) GTEST_SKIP() << "Cu_u3.eam is not present";
  const double box[3] = {30.0, 30.0, 30.0};
  const double x0[] = {10.0, 12.6, 10.4, 11.2};
  const double y0[] = {10.0, 10.2, 12.7, 11.1};
  const double z0[] = {10.0, 10.3, 10.2, 12.5};
  int type[] = {0, 0, 0, 0};
  double x[4];
  double y[4];
  double z[4];
  for (int i = 0; i < 4; ++i) {
    x[i] = x0[i];
    y[i] = y0[i];
    z[i] = z0[i];
  }
  tilemd::EamFuncfl potential(table.string());
  const tilemd::AttentionResult base = tilemd::EvaluateAttention(potential, box, x, y, z, type, 4);
  const double h = 1.0e-5;
  x[0] = x0[0] + h;
  tilemd::EamFuncfl plus_table(table.string());
  const double e_plus =
      tilemd::EvaluateAttention(plus_table, box, x, y, z, type, 4).potential_energy;
  x[0] = x0[0] - h;
  tilemd::EamFuncfl minus_table(table.string());
  const double e_minus =
      tilemd::EvaluateAttention(minus_table, box, x, y, z, type, 4).potential_energy;
  const double numerical = -(e_plus - e_minus) / (2.0 * h);
  EXPECT_NEAR(base.fx[0], numerical, 1.0e-6 * (1.0 + std::abs(numerical)));
  double total[3] = {};
  for (int i = 0; i < 4; ++i) {
    total[0] += base.fx[static_cast<size_t>(i)];
    total[1] += base.fy[static_cast<size_t>(i)];
    total[2] += base.fz[static_cast<size_t>(i)];
  }
  EXPECT_NEAR(total[0], 0.0, 1.0e-8);
  EXPECT_NEAR(total[1], 0.0, 1.0e-8);
  EXPECT_NEAR(total[2], 0.0, 1.0e-8);
}

TEST(PairAttention, EamMatchesLammpsCopper) {
  const std::filesystem::path table =
      std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "potentials" / "Cu_u3.eam";
  const std::filesystem::path ref = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                    "validation" / "lammps_ref" / "cu_u3_108.txt";
  if (!std::filesystem::exists(table) || !std::filesystem::exists(ref)) {
    GTEST_SKIP() << "Cu EAM reference has not been generated";
  }
  std::ifstream in(ref);
  int n = 0;
  double box[3];
  double energy = 0.0;
  in >> n >> box[0] >> box[1] >> box[2] >> energy;
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n));
  std::vector<double> fx(static_cast<size_t>(n));
  std::vector<double> fy(static_cast<size_t>(n));
  std::vector<double> fz(static_cast<size_t>(n));
  std::vector<int> type(static_cast<size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    in >> x[static_cast<size_t>(i)] >> y[static_cast<size_t>(i)] >> z[static_cast<size_t>(i)] >>
        fx[static_cast<size_t>(i)] >> fy[static_cast<size_t>(i)] >> fz[static_cast<size_t>(i)];
  }
  ASSERT_TRUE(static_cast<bool>(in));
  tilemd::EamFuncfl potential(table.string());
  const tilemd::AttentionResult got =
      tilemd::EvaluateAttention(potential, box, x.data(), y.data(), z.data(), type.data(), n);
  const double error = RelativeForce(got.fx, got.fy, got.fz, fx, fy, fz);
  std::cout << std::scientific << std::setprecision(8) << "eam vs lammps n=" << n << " force_rel "
            << error << " energy " << got.potential_energy << " lammps " << energy << "\n";
  EXPECT_LT(error, 1.0e-5);
  EXPECT_NEAR(got.potential_energy, energy, 1.0e-8 * (1.0 + std::abs(energy)));
  tilemd::EamFuncfl tiled(table.string());
  const tilemd::AttentionResult from_tiles =
      tilemd::EvaluateAttentionTiles(tiled, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
  const double tile_error = RelativeForce(from_tiles.fx, from_tiles.fy, from_tiles.fz, fx, fy, fz);
  std::cout << std::scientific << std::setprecision(8) << "eam tiles vs lammps force_rel "
            << tile_error << " energy " << from_tiles.potential_energy << "\n";
  EXPECT_LT(tile_error, 1.0e-5);
  EXPECT_NEAR(from_tiles.potential_energy, energy, 1.0e-8 * (1.0 + std::abs(energy)));
}

void CopperVelocity(int n, std::vector<double>& vx, std::vector<double>& vy,
                    std::vector<double>& vz) {
  vx.resize(static_cast<size_t>(n));
  vy.resize(static_cast<size_t>(n));
  vz.resize(static_cast<size_t>(n));
  double mx = 0.0;
  double my = 0.0;
  double mz = 0.0;
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] = 0.8 * ((i % 7) - 3);
    vy[static_cast<size_t>(i)] = 0.5 * ((i % 5) - 2);
    vz[static_cast<size_t>(i)] = 0.4 * ((i % 3) - 1);
    mx += vx[static_cast<size_t>(i)];
    my += vy[static_cast<size_t>(i)];
    mz += vz[static_cast<size_t>(i)];
  }
  mx /= static_cast<double>(n);
  my /= static_cast<double>(n);
  mz /= static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] -= mx;
    vy[static_cast<size_t>(i)] -= my;
    vz[static_cast<size_t>(i)] -= mz;
  }
}

void WrapBox(double* value, double length) {
  double periods = std::floor(*value / length);
  *value -= periods * length;
  if (*value >= length) *value -= length;
  if (*value < 0.0) *value += length;
}

TEST(PairAttention, EamNveMatchesLammpsFor100Steps) {
  const std::filesystem::path table =
      std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "potentials" / "Cu_u3.eam";
  const std::filesystem::path start = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                      "validation" / "lammps_ref" / "cu_u3_108.txt";
  const std::filesystem::path ref = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                    "validation" / "lammps_ref" / "cu_u3_nve_100.txt";
  if (!std::filesystem::exists(table) || !std::filesystem::exists(start) ||
      !std::filesystem::exists(ref)) {
    GTEST_SKIP() << "Cu NVE reference has not been generated";
  }
  std::ifstream initial(start);
  int n = 0;
  double box[3];
  double unused_energy = 0.0;
  initial >> n >> box[0] >> box[1] >> box[2] >> unused_energy;
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n));
  std::vector<int> type(static_cast<size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    initial >> x[static_cast<size_t>(i)] >> y[static_cast<size_t>(i)] >>
        z[static_cast<size_t>(i)] >> fx >> fy >> fz;
  }
  ASSERT_TRUE(static_cast<bool>(initial));
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  CopperVelocity(n, vx, vy, vz);
  tilemd::EamFuncfl potential(table.string());
  const double mass = potential.Mass();
  const double dt = 0.001;
  const int steps = 100;
  for (int step = 0; step < steps; ++step) {
    const tilemd::AttentionResult force =
        tilemd::EvaluateAttention(potential, box, x.data(), y.data(), z.data(), type.data(), n);
    const double half = 0.5 * dt * tilemd::kMetalFtm2v / mass;
    for (int i = 0; i < n; ++i) {
      vx[static_cast<size_t>(i)] += half * force.fx[static_cast<size_t>(i)];
      vy[static_cast<size_t>(i)] += half * force.fy[static_cast<size_t>(i)];
      vz[static_cast<size_t>(i)] += half * force.fz[static_cast<size_t>(i)];
      x[static_cast<size_t>(i)] += dt * vx[static_cast<size_t>(i)];
      y[static_cast<size_t>(i)] += dt * vy[static_cast<size_t>(i)];
      z[static_cast<size_t>(i)] += dt * vz[static_cast<size_t>(i)];
      WrapBox(&x[static_cast<size_t>(i)], box[0]);
      WrapBox(&y[static_cast<size_t>(i)], box[1]);
      WrapBox(&z[static_cast<size_t>(i)], box[2]);
    }
    const tilemd::AttentionResult second =
        tilemd::EvaluateAttention(potential, box, x.data(), y.data(), z.data(), type.data(), n);
    for (int i = 0; i < n; ++i) {
      vx[static_cast<size_t>(i)] += half * second.fx[static_cast<size_t>(i)];
      vy[static_cast<size_t>(i)] += half * second.fy[static_cast<size_t>(i)];
      vz[static_cast<size_t>(i)] += half * second.fz[static_cast<size_t>(i)];
    }
  }
  const tilemd::AttentionResult final_force =
      tilemd::EvaluateAttention(potential, box, x.data(), y.data(), z.data(), type.data(), n);
  std::ifstream in(ref);
  int n_ref = 0;
  double box_ref[3];
  double dt_ref = 0.0;
  int steps_ref = 0;
  double energy = 0.0;
  in >> n_ref >> box_ref[0] >> box_ref[1] >> box_ref[2] >> dt_ref >> steps_ref >> energy;
  ASSERT_EQ(n_ref, n);
  ASSERT_EQ(steps_ref, steps);
  double max_dx = 0.0;
  double max_dv = 0.0;
  for (int i = 0; i < n; ++i) {
    double rx = 0.0;
    double ry = 0.0;
    double rz = 0.0;
    double rvx = 0.0;
    double rvy = 0.0;
    double rvz = 0.0;
    in >> rx >> ry >> rz >> rvx >> rvy >> rvz;
    const double dx = x[static_cast<size_t>(i)] - rx -
                      box[0] * std::rint((x[static_cast<size_t>(i)] - rx) / box[0]);
    const double dy = y[static_cast<size_t>(i)] - ry -
                      box[1] * std::rint((y[static_cast<size_t>(i)] - ry) / box[1]);
    const double dz = z[static_cast<size_t>(i)] - rz -
                      box[2] * std::rint((z[static_cast<size_t>(i)] - rz) / box[2]);
    max_dx = std::max(max_dx, std::sqrt(dx * dx + dy * dy + dz * dz));
    const double dvx = vx[static_cast<size_t>(i)] - rvx;
    const double dvy = vy[static_cast<size_t>(i)] - rvy;
    const double dvz = vz[static_cast<size_t>(i)] - rvz;
    max_dv = std::max(max_dv, std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz));
  }
  ASSERT_TRUE(static_cast<bool>(in));
  std::cout << std::scientific << std::setprecision(8) << "eam nve 100 max |dx| " << max_dx
            << " max |dv| " << max_dv << " energy " << final_force.potential_energy << " lammps "
            << energy << "\n";
  EXPECT_LT(max_dx, 1.0e-8);
  EXPECT_LT(max_dv, 1.0e-8);
  EXPECT_NEAR(final_force.potential_energy, energy, 1.0e-6 * (1.0 + std::abs(energy)));
}
