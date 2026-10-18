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

void ExpectLj(const std::string& name) {
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
  const tilemd::AttentionResult got =
      tilemd::EvaluateAttention(potential, system.box, system.x.data(), system.y.data(),
                                system.z.data(), system.type.data(), tilemd::ref::NumAtoms(system));
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
  for (int k = 0; k < 6; ++k) {
    EXPECT_NEAR(got.virial[k], oracle.virial[k], 1.0e-9 * (1.0 + std::abs(oracle.virial[k])))
        << name << " virial " << k;
  }
}

}  // namespace

TEST(PairAttention, LennardJonesMatchesTheOracle) {
  const char* names[] = {"fcc_500", "boundary_500", "cutoff_500", "twotype_2000", "melt_4000"};
  for (const char* name : names) ExpectLj(name);
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
}
