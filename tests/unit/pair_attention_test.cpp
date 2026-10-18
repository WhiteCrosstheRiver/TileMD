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

// Host replay of LjAttentionTileKernel: lane = i_local + 8 * j_sub, four
// j-clusters per step, shuffle source = atom + 8 * j_sub, then a sum over the
// four lanes that own one i atom.
tilemd::AttentionResult WarpLaneWalk(const tilemd::ref::System& system) {
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
  const int n = tilemd::ref::NumAtoms(system);
  const tilemd::ClusterGrid grid =
      tilemd::BuildClusterGrid(system.box, system.x.data(), system.y.data(), system.z.data(),
                               system.type.data(), n, potential.Cutoff() + 0.3);
  const tilemd::ClusterPairs list = tilemd::BuildClusterPairs(grid);
  tilemd::LjRow table[tilemd::LjAttention::kMaxTypes * tilemd::LjAttention::kMaxTypes];
  potential.CopyTable(table);
  const int force_shift = potential.ForceShift();
  tilemd::AttentionResult out;
  out.fx.assign(static_cast<size_t>(n), 0.0);
  out.fy.assign(static_cast<size_t>(n), 0.0);
  out.fz.assign(static_cast<size_t>(n), 0.0);
  out.energy.assign(static_cast<size_t>(n), 0.0);

  struct Lane {
    int ti = tilemd::kMaskType;
    int gi = -1;
    double xi = 0.0;
    double yi = 0.0;
    double zi = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double a1 = 0.0;
    double ae = 0.0;
    double axx = 0.0;
    double ayy = 0.0;
    double azz = 0.0;
    double axy = 0.0;
    double axz = 0.0;
    double ayz = 0.0;
  };

  for (int ci = 0; ci < grid.ncluster; ++ci) {
    Lane lane[32];
    for (int lane_id = 0; lane_id < 32; ++lane_id) {
      const int i_local = lane_id % tilemd::kClusterSize;
      const int slot_i = ci * tilemd::kClusterSize + i_local;
      lane[lane_id].ti = grid.type[static_cast<size_t>(slot_i)];
      lane[lane_id].gi = grid.gid[static_cast<size_t>(slot_i)];
      lane[lane_id].xi = grid.x[static_cast<size_t>(slot_i)];
      lane[lane_id].yi = grid.y[static_cast<size_t>(slot_i)];
      lane[lane_id].zi = grid.z[static_cast<size_t>(slot_i)];
    }
    const int begin = list.ci_start[static_cast<size_t>(ci)];
    const int end = list.ci_start[static_cast<size_t>(ci) + 1];
    for (int pair0 = begin; pair0 < end; pair0 += 4) {
      double xj[32] = {};
      double yj[32] = {};
      double zj[32] = {};
      int tj[32];
      int active[32];
      int other[32];
      int sx[32];
      int sy[32];
      int sz[32];
      for (int lane_id = 0; lane_id < 32; ++lane_id) {
        const int i_local = lane_id % tilemd::kClusterSize;
        const int j_sub = lane_id / tilemd::kClusterSize;
        const int pair = pair0 + j_sub;
        active[lane_id] = pair < end ? 1 : 0;
        other[lane_id] = active[lane_id] ? list.cj[static_cast<size_t>(pair)] : 0;
        sx[lane_id] = 0;
        sy[lane_id] = 0;
        sz[lane_id] = 0;
        if (active[lane_id]) {
          const int shift = list.shift[static_cast<size_t>(pair)];
          sx[lane_id] = shift % 3 - 1;
          sy[lane_id] = (shift / 3) % 3 - 1;
          sz[lane_id] = shift / 9 - 1;
        }
        const int slot_j = other[lane_id] * tilemd::kClusterSize + i_local;
        tj[lane_id] = active[lane_id] ? grid.type[static_cast<size_t>(slot_j)] : tilemd::kMaskType;
        if (active[lane_id] && tj[lane_id] != tilemd::kMaskType) {
          xj[lane_id] = grid.x[static_cast<size_t>(slot_j)] + sx[lane_id] * grid.box[0];
          yj[lane_id] = grid.y[static_cast<size_t>(slot_j)] + sy[lane_id] * grid.box[1];
          zj[lane_id] = grid.z[static_cast<size_t>(slot_j)] + sz[lane_id] * grid.box[2];
        }
      }
      for (int atom = 0; atom < tilemd::kClusterSize; ++atom) {
        for (int lane_id = 0; lane_id < 32; ++lane_id) {
          const int i_local = lane_id % tilemd::kClusterSize;
          const int j_sub = lane_id / tilemd::kClusterSize;
          const int source = atom + tilemd::kClusterSize * j_sub;
          const int bt = tj[source];
          if (!active[lane_id] || lane[lane_id].ti == tilemd::kMaskType ||
              bt == tilemd::kMaskType) {
            continue;
          }
          if (ci == other[lane_id] && atom == i_local && sx[lane_id] == 0 && sy[lane_id] == 0 &&
              sz[lane_id] == 0) {
            continue;
          }
          const tilemd::LjRow& row = table[lane[lane_id].ti * tilemd::LjAttention::kMaxTypes + bt];
          if (!row.defined) continue;
          const double dx = lane[lane_id].xi - xj[source];
          const double dy = lane[lane_id].yi - yj[source];
          const double dz = lane[lane_id].zi - zj[source];
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (!(r2 < row.rc2) || !(r2 > 0.0)) continue;
          double s = 0.0;
          double u = 0.0;
          tilemd::LjEvaluate(row, r2, force_shift, &s, &u);
          lane[lane_id].ax += s * (lane[lane_id].xi - dx);
          lane[lane_id].ay += s * (lane[lane_id].yi - dy);
          lane[lane_id].az += s * (lane[lane_id].zi - dz);
          lane[lane_id].a1 += s;
          lane[lane_id].ae += 0.5 * u;
          const double fix = s * dx;
          const double fiy = s * dy;
          const double fiz = s * dz;
          lane[lane_id].axx += 0.5 * dx * fix;
          lane[lane_id].ayy += 0.5 * dy * fiy;
          lane[lane_id].azz += 0.5 * dz * fiz;
          lane[lane_id].axy += 0.5 * dx * fiy;
          lane[lane_id].axz += 0.5 * dx * fiz;
          lane[lane_id].ayz += 0.5 * dy * fiz;
        }
      }
    }
    for (int i_local = 0; i_local < tilemd::kClusterSize; ++i_local) {
      const Lane& owner = lane[i_local];
      if (owner.ti == tilemd::kMaskType || owner.gi < 0) continue;
      double rx = 0.0;
      double ry = 0.0;
      double rz = 0.0;
      double r1 = 0.0;
      double re = 0.0;
      double rxx = 0.0;
      double ryy = 0.0;
      double rzz = 0.0;
      double rxy = 0.0;
      double rxz = 0.0;
      double ryz = 0.0;
      for (int j_sub = 0; j_sub < 4; ++j_sub) {
        const Lane& src = lane[i_local + tilemd::kClusterSize * j_sub];
        rx += src.ax;
        ry += src.ay;
        rz += src.az;
        r1 += src.a1;
        re += src.ae;
        rxx += src.axx;
        ryy += src.ayy;
        rzz += src.azz;
        rxy += src.axy;
        rxz += src.axz;
        ryz += src.ayz;
      }
      const int gi = owner.gi;
      out.fx[static_cast<size_t>(gi)] = owner.xi * r1 - rx;
      out.fy[static_cast<size_t>(gi)] = owner.yi * r1 - ry;
      out.fz[static_cast<size_t>(gi)] = owner.zi * r1 - rz;
      out.energy[static_cast<size_t>(gi)] = re;
      out.virial[0] += rxx;
      out.virial[1] += ryy;
      out.virial[2] += rzz;
      out.virial[3] += rxy;
      out.virial[4] += rxz;
      out.virial[5] += ryz;
    }
  }
  for (int i = 0; i < n; ++i) out.potential_energy += out.energy[static_cast<size_t>(i)];
  return out;
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

TEST(PairAttention, WarpLaneWalkMatchesTheOracle) {
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  const char* names[] = {"fcc_500", "boundary_500", "cutoff_500", "twotype_2000", "melt_4000"};
  for (const char* name : names) {
    tilemd::ref::System system =
        tilemd::ref::ReadExtxyz((dir / (std::string(name) + ".extxyz")).string());
    tilemd::ref::ReadSystemHeader((dir / (std::string(name) + ".json")).string(), system);
    const tilemd::AttentionResult got = WarpLaneWalk(system);
    const tilemd::ref::Observables oracle = tilemd::ref::Compute(system);
    const int n = tilemd::ref::NumAtoms(system);
    double max_delta = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double dx = got.fx[static_cast<size_t>(i)] - oracle.fx[static_cast<size_t>(i)];
      const double dy = got.fy[static_cast<size_t>(i)] - oracle.fy[static_cast<size_t>(i)];
      const double dz = got.fz[static_cast<size_t>(i)] - oracle.fz[static_cast<size_t>(i)];
      max_delta = std::max(max_delta, std::sqrt(dx * dx + dy * dy + dz * dz));
      sum_sq += oracle.fx[static_cast<size_t>(i)] * oracle.fx[static_cast<size_t>(i)] +
                oracle.fy[static_cast<size_t>(i)] * oracle.fy[static_cast<size_t>(i)] +
                oracle.fz[static_cast<size_t>(i)] * oracle.fz[static_cast<size_t>(i)];
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(std::max(n, 1)));
    if (rms < 1.0e-8) {
      EXPECT_LT(max_delta, 1.0e-8) << name;
    } else {
      EXPECT_LT(max_delta / rms, 1.0e-10) << name;
    }
    EXPECT_NEAR(got.potential_energy, oracle.potential_energy,
                1.0e-10 * (1.0 + std::abs(oracle.potential_energy)))
        << name;
    double virial_scale = 1.0;
    for (int k = 0; k < 6; ++k) virial_scale = std::max(virial_scale, std::abs(oracle.virial[k]));
    for (int k = 0; k < 6; ++k) {
      EXPECT_NEAR(got.virial[k], oracle.virial[k], 1.0e-9 * virial_scale)
          << name << " virial " << k;
    }
  }
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

TEST(PairAttention, EamTileNveMatchesLammpsFor100Steps) {
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
    const tilemd::AttentionResult force = tilemd::EvaluateAttentionTiles(
        potential, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
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
    const tilemd::AttentionResult second = tilemd::EvaluateAttentionTiles(
        potential, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
    for (int i = 0; i < n; ++i) {
      vx[static_cast<size_t>(i)] += half * second.fx[static_cast<size_t>(i)];
      vy[static_cast<size_t>(i)] += half * second.fy[static_cast<size_t>(i)];
      vz[static_cast<size_t>(i)] += half * second.fz[static_cast<size_t>(i)];
    }
  }
  const tilemd::AttentionResult final_force = tilemd::EvaluateAttentionTiles(
      potential, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
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
  std::cout << std::scientific << std::setprecision(8) << "eam tile nve 100 max |dx| " << max_dx
            << " max |dv| " << max_dv << " energy " << final_force.potential_energy << " lammps "
            << energy << "\n";
  EXPECT_LT(max_dx, 1.0e-8);
  EXPECT_LT(max_dv, 1.0e-8);
  EXPECT_NEAR(final_force.potential_energy, energy, 1.0e-6 * (1.0 + std::abs(energy)));
}
