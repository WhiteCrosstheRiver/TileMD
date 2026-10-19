#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "force/eam_attention.hpp"
#include "force/lj_attention.hpp"
#include "force/tile_attention.hpp"
#include "lj_reference.hpp"
#include "md/engine.hpp"
#include "md/tile_rebuild.hpp"
#include "md/tile_thermo.hpp"
#include "neighbor/cluster_list.hpp"

namespace {

bool HasGpu() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

tilemd::Shift ToEngine(tilemd::ref::Shift shift) {
  switch (shift) {
    case tilemd::ref::Shift::None:
      return tilemd::Shift::None;
    case tilemd::ref::Shift::Energy:
      return tilemd::Shift::Energy;
    case tilemd::ref::Shift::Force:
      return tilemd::Shift::Force;
  }
  return tilemd::Shift::None;
}

tilemd::HostSystem FromReference(const tilemd::ref::System& system, double skin, int max_neigh,
                                 bool rebuild_every_step) {
  tilemd::HostSystem host;
  host.box[0] = system.box[0];
  host.box[1] = system.box[1];
  host.box[2] = system.box[2];
  host.shift = ToEngine(system.shift);
  host.skin = skin;
  host.max_neigh = max_neigh;
  host.rebuild_every_step = rebuild_every_step;
  host.type = system.type;
  host.x = system.x;
  host.y = system.y;
  host.z = system.z;
  host.vx.assign(system.x.size(), 0.0);
  host.vy.assign(system.x.size(), 0.0);
  host.vz.assign(system.x.size(), 0.0);
  for (const tilemd::ref::PairCoeff& coeff : system.pairs) {
    tilemd::PairSpec spec;
    spec.type_i = coeff.type_i;
    spec.type_j = coeff.type_j;
    spec.epsilon = coeff.epsilon;
    spec.sigma = coeff.sigma;
    spec.cutoff = coeff.cutoff;
    host.pairs.push_back(spec);
  }
  return host;
}

void ExpectClose(const tilemd::GpuForces& got, const tilemd::ref::Observables& expected,
                 const char* name) {
  ASSERT_EQ(got.overflow, 0) << name;
  const int n = static_cast<int>(expected.fx.size());
  double max_delta = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dx = static_cast<double>(got.fx[i]) - expected.fx[i];
    const double dy = static_cast<double>(got.fy[i]) - expected.fy[i];
    const double dz = static_cast<double>(got.fz[i]) - expected.fz[i];
    max_delta = std::max(max_delta, std::sqrt(dx * dx + dy * dy + dz * dz));
    sum_sq += expected.fx[i] * expected.fx[i] + expected.fy[i] * expected.fy[i] +
              expected.fz[i] * expected.fz[i];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(n));
  if (rms < 1.0e-8) {
    // A cancelled crystal has no scale for a relative error. PLAN.md uses an
    // absolute bound for fcc_500.
    EXPECT_LE(max_delta, 1.0e-4) << name;
  } else {
    EXPECT_LE(max_delta / rms, 1.0e-5) << name << " relative force error " << (max_delta / rms);
  }
  EXPECT_NEAR(got.potential_energy, expected.potential_energy,
              1.0e-6 * (1.0 + std::abs(expected.potential_energy)))
      << name;
  for (int k = 0; k < 6; ++k) {
    EXPECT_NEAR(got.virial[k], expected.virial[k], 1.0e-5 * (1.0 + std::abs(expected.virial[k])))
        << name << " virial " << k;
  }
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double force_sq = 0.0;
  for (int i = 0; i < static_cast<int>(got.fx.size()); ++i) {
    sum_x += got.fx[i];
    sum_y += got.fy[i];
    sum_z += got.fz[i];
    force_sq += static_cast<double>(got.fx[i]) * got.fx[i] +
                static_cast<double>(got.fy[i]) * got.fy[i] +
                static_cast<double>(got.fz[i]) * got.fz[i];
  }
  const double total = std::sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z);
  const double force_rms = std::sqrt(force_sq / static_cast<double>(got.fx.size()));
  const double bound =
      1.0e-4 * std::max(force_rms, 1.0e-8) * std::sqrt(static_cast<double>(got.fx.size()));
  EXPECT_LE(total, bound) << name << " total force " << total;
}

std::filesystem::path SourcePath(const std::string& relative) {
  return std::filesystem::path(TILEMD_SOURCE_DIR) / relative;
}

void AssignThermalVelocities(tilemd::HostSystem& host, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, std::sqrt(1.44));
  const int n = static_cast<int>(host.x.size());
  host.vx.resize(n);
  host.vy.resize(n);
  host.vz.resize(n);
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (int i = 0; i < n; ++i) {
    host.vx[i] = normal(rng);
    host.vy[i] = normal(rng);
    host.vz[i] = normal(rng);
    sx += host.vx[i];
    sy += host.vy[i];
    sz += host.vz[i];
  }
  for (int i = 0; i < n; ++i) {
    host.vx[i] -= sx / n;
    host.vy[i] -= sy / n;
    host.vz[i] -= sz / n;
  }
}

}  // namespace

TEST(GpuLj, FixturesMatchReference) {
  if (!HasGpu()) GTEST_SKIP() << "No CUDA device";
  const char* names[] = {"fcc_500",     "boundary_500", "cutoff_500", "twotype_2000",
                         "dilute_1000", "dense_2000",   "melt_4000"};
  int present = 0;
  for (const char* name : names) {
    const auto xyz = SourcePath(std::string("tests/fixtures/") + name + ".extxyz");
    const auto json = SourcePath(std::string("tests/fixtures/") + name + ".json");
    if (!std::filesystem::exists(xyz) || !std::filesystem::exists(json)) continue;
    ++present;
    tilemd::ref::System system = tilemd::ref::ReadExtxyz(xyz.string());
    tilemd::ref::ReadSystemHeader(json.string(), system);
    const tilemd::ref::Observables expected = tilemd::ref::Compute(system);
    tilemd::Engine engine(FromReference(system, 0.0, 0, true));
    const tilemd::GpuForces got = engine.ComputeForces();
    if (std::string(name) == "fcc_500") {
      double max_force = 0.0;
      for (float component : got.fx) {
        max_force = std::max(max_force, static_cast<double>(std::fabs(component)));
      }
      for (float component : got.fy) {
        max_force = std::max(max_force, static_cast<double>(std::fabs(component)));
      }
      for (float component : got.fz) {
        max_force = std::max(max_force, static_cast<double>(std::fabs(component)));
      }
      EXPECT_LE(max_force, 1.0e-4);
    }
    ExpectClose(got, expected, name);
  }
  if (present == 0) GTEST_SKIP() << "fixtures have not been generated";
  EXPECT_EQ(present, 7);
}

TEST(GpuLj, ShiftAndMinimumImage) {
  if (!HasGpu()) GTEST_SKIP() << "No CUDA device";
  tilemd::ref::PairCoeff coeff;
  coeff.type_i = 0;
  coeff.type_j = 0;
  coeff.epsilon = 1.0;
  coeff.sigma = 1.0;
  coeff.cutoff = 2.5;
  const tilemd::ref::Shift shifts[] = {tilemd::ref::Shift::None, tilemd::ref::Shift::Energy,
                                       tilemd::ref::Shift::Force};
  for (tilemd::ref::Shift shift : shifts) {
    tilemd::ref::System system;
    system.box[0] = system.box[1] = system.box[2] = 20.0;
    system.shift = shift;
    system.pairs = {coeff};
    system.type = {0, 0};
    system.x = {0.1, 19.9};
    system.y = {1.0, 1.0};
    system.z = {2.0, 2.0};
    const tilemd::ref::Observables expected = tilemd::ref::Compute(system);
    tilemd::Engine engine(FromReference(system, 0.0, 32, true));
    ExpectClose(engine.ComputeForces(), expected, "boundary pair");
  }
}

TEST(GpuLj, DeterministicNve) {
  if (!HasGpu()) GTEST_SKIP() << "No CUDA device";
  const auto xyz = SourcePath("tests/fixtures/melt_4000.extxyz");
  const auto json = SourcePath("tests/fixtures/melt_4000.json");
  if (!std::filesystem::exists(xyz)) GTEST_SKIP() << "melt fixture is missing";
  tilemd::ref::System system = tilemd::ref::ReadExtxyz(xyz.string());
  tilemd::ref::ReadSystemHeader(json.string(), system);
  tilemd::HostSystem host = FromReference(system, 0.3, 0, true);
  AssignThermalVelocities(host, 20261018);
  tilemd::Engine first(host);
  tilemd::Engine second(host);
  first.Run(10000);
  second.Run(10000);
  std::vector<double> x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2;
  first.DownloadPositions(x1, y1, z1);
  second.DownloadPositions(x2, y2, z2);
  first.DownloadVelocities(vx1, vy1, vz1);
  second.DownloadVelocities(vx2, vy2, vz2);
  EXPECT_EQ(0, std::memcmp(x1.data(), x2.data(), x1.size() * sizeof(double)));
  EXPECT_EQ(0, std::memcmp(y1.data(), y2.data(), y1.size() * sizeof(double)));
  EXPECT_EQ(0, std::memcmp(z1.data(), z2.data(), z1.size() * sizeof(double)));
  EXPECT_EQ(0, std::memcmp(vx1.data(), vx2.data(), vx1.size() * sizeof(double)));
  EXPECT_EQ(0, std::memcmp(vy1.data(), vy2.data(), vy1.size() * sizeof(double)));
  EXPECT_EQ(0, std::memcmp(vz1.data(), vz2.data(), vz1.size() * sizeof(double)));
}

TEST(GpuLj, SkinRebuildMatchesEveryStep) {
  if (!HasGpu()) GTEST_SKIP() << "No CUDA device";
  const auto xyz = SourcePath("tests/fixtures/melt_4000.extxyz");
  const auto json = SourcePath("tests/fixtures/melt_4000.json");
  if (!std::filesystem::exists(xyz)) GTEST_SKIP() << "melt fixture is missing";
  tilemd::ref::System system = tilemd::ref::ReadExtxyz(xyz.string());
  tilemd::ref::ReadSystemHeader(json.string(), system);
  tilemd::HostSystem every = FromReference(system, 0.3, 0, true);
  AssignThermalVelocities(every, 20261018);
  tilemd::HostSystem skin = every;
  skin.rebuild_every_step = false;
  tilemd::Engine every_engine(every);
  tilemd::Engine skin_engine(skin);
  every_engine.Run(1000);
  skin_engine.Run(1000);
  std::vector<double> x1, y1, z1, x2, y2, z2;
  every_engine.DownloadPositions(x1, y1, z1);
  skin_engine.DownloadPositions(x2, y2, z2);
  double max_delta = 0.0;
  for (int i = 0; i < static_cast<int>(x1.size()); ++i) {
    max_delta = std::max(max_delta, std::abs(x1[i] - x2[i]));
    max_delta = std::max(max_delta, std::abs(y1[i] - y2[i]));
    max_delta = std::max(max_delta, std::abs(z1[i] - z2[i]));
  }
  EXPECT_LE(max_delta, 1.0e-6) << "rebuilds " << skin_engine.RebuildCount();
  EXPECT_GT(skin_engine.RebuildCount(), 1);
  EXPECT_LT(skin_engine.RebuildCount(), every_engine.RebuildCount());
}

TEST(GpuLj, OverflowIsDetectedThenGrows) {
  if (!HasGpu()) GTEST_SKIP() << "No CUDA device";
  const auto xyz = SourcePath("tests/fixtures/fcc_500.extxyz");
  const auto json = SourcePath("tests/fixtures/fcc_500.json");
  if (!std::filesystem::exists(xyz)) GTEST_SKIP() << "fcc fixture is missing";
  tilemd::ref::System system = tilemd::ref::ReadExtxyz(xyz.string());
  tilemd::ref::ReadSystemHeader(json.string(), system);
  tilemd::Engine tight(FromReference(system, 0.0, 8, true));
  const tilemd::GpuForces overflowed = tight.ComputeForces(false);
  EXPECT_EQ(overflowed.overflow, 1);
  const tilemd::GpuForces grown = tight.ComputeForces(true);
  tilemd::Engine roomy(FromReference(system, 0.0, 256, true));
  const tilemd::GpuForces reference = roomy.ComputeForces();
  ASSERT_EQ(grown.overflow, 0);
  ASSERT_EQ(reference.overflow, 0);
  ASSERT_GT(grown.max_neigh, 8);
  for (int i = 0; i < static_cast<int>(grown.fx.size()); ++i) {
    EXPECT_EQ(grown.fx[i], reference.fx[i]);
    EXPECT_EQ(grown.fy[i], reference.fy[i]);
    EXPECT_EQ(grown.fz[i], reference.fz[i]);
  }
  EXPECT_NEAR(grown.potential_energy, reference.potential_energy, 0.0);
}

struct ThermoRow {
  int step = 0;
  double temp = 0.0;
  double pe = 0.0;
  double ke = 0.0;
  double press = 0.0;
};

std::vector<ThermoRow> ReadThermo(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open thermo file");
  std::vector<ThermoRow> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    ThermoRow row;
    std::istringstream text(line);
    text >> row.step >> row.temp >> row.pe >> row.ke >> row.press;
    if (!text) throw std::runtime_error("truncated thermo row");
    rows.push_back(row);
  }
  return rows;
}

tilemd::HostSystem LoadLammpsInitial(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open initial dump");
  tilemd::HostSystem host;
  host.shift = tilemd::Shift::None;
  host.skin = 0.3;
  host.dt = 0.005;
  host.mass = 1.0;
  host.rebuild_every_step = false;
  tilemd::PairSpec spec;
  spec.cutoff = 2.5;
  host.pairs = {spec};
  std::string line;
  int n = -1;
  while (std::getline(in, line)) {
    if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) {
      in >> n;
      std::getline(in, line);
    } else if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) {
      double lo = 0.0;
      double hi = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        in >> lo >> hi;
        host.box[axis] = hi - lo;
        std::getline(in, line);
      }
    } else if (line.rfind("ITEM: ATOMS", 0) == 0) {
      if (n < 1) throw std::runtime_error("dump has no atom count");
      host.type.assign(n, 0);
      host.x.assign(n, 0.0);
      host.y.assign(n, 0.0);
      host.z.assign(n, 0.0);
      host.vx.assign(n, 0.0);
      host.vy.assign(n, 0.0);
      host.vz.assign(n, 0.0);
      for (int row = 0; row < n; ++row) {
        int id = 0;
        int type = 0;
        double x, y, z, vx, vy, vz;
        in >> id >> type >> x >> y >> z >> vx >> vy >> vz;
        if (!in) throw std::runtime_error("truncated initial dump");
        if (id < 1 || id > n) throw std::runtime_error("LAMMPS id is outside 1..N");
        const int slot = id - 1;
        host.type[slot] = type - 1;
        host.x[slot] = x;
        host.y[slot] = y;
        host.z[slot] = z;
        host.vx[slot] = vx;
        host.vy[slot] = vy;
        host.vz[slot] = vz;
      }
      return host;
    }
  }
  throw std::runtime_error("dump has no atoms");
}

double Mean(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values) sum += value;
  return sum / static_cast<double>(values.size());
}

double Stderr(const std::vector<double>& values) {
  const double mean = Mean(values);
  double acc = 0.0;
  for (double value : values) acc += (value - mean) * (value - mean);
  return std::sqrt(acc / static_cast<double>(values.size() - 1) /
                   static_cast<double>(values.size()));
}

// T3: same 32k initial state as CPU LAMMPS, 10 000 NVE steps. Float forces
// make the trajectories diverge, so the gate is the mean of the thermo samples
// rather than a pointwise match.
TEST(GpuLj, ShortNveMatchesLammps) {
  if (!HasGpu()) GTEST_SKIP() << "No CUDA device";
  const auto dump = SourcePath("tests/validation/lammps_ref/init_32k.dump");
  const auto thermo = SourcePath("tests/validation/lammps_ref/melt_32k_thermo.txt");
  if (!std::filesystem::exists(dump) || !std::filesystem::exists(thermo)) {
    GTEST_SKIP() << "32k LAMMPS reference is missing";
  }
  const std::vector<ThermoRow> reference = ReadThermo(thermo);
  ASSERT_EQ(reference.size(), 11u);
  tilemd::HostSystem host = LoadLammpsInitial(dump);
  ASSERT_EQ(host.x.size(), 32000u);
  tilemd::Engine engine(host);
  std::vector<double> pe;
  std::vector<double> ke;
  std::vector<double> press;
  const double n = static_cast<double>(host.x.size());
  const double volume = host.box[0] * host.box[1] * host.box[2];
  for (int sample = 0; sample < static_cast<int>(reference.size()); ++sample) {
    const tilemd::GpuForces forces = engine.ComputeForces();
    ASSERT_EQ(forces.overflow, 0);
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
    engine.DownloadVelocities(vx, vy, vz);
    double kinetic = 0.0;
    for (int i = 0; i < static_cast<int>(vx.size()); ++i) {
      kinetic += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }
    pe.push_back(forces.potential_energy / n);
    ke.push_back(kinetic / n);
    const double trace = forces.virial[0] + forces.virial[1] + forces.virial[2];
    press.push_back((2.0 * kinetic + trace) / (3.0 * volume));
    EXPECT_NEAR(pe.back(), reference[sample].pe, 0.05) << "step " << reference[sample].step;
    EXPECT_NEAR(ke.back(), reference[sample].ke, 0.05) << "step " << reference[sample].step;
    EXPECT_NEAR(press.back(), reference[sample].press, 0.15) << "step " << reference[sample].step;
    if (sample + 1 < static_cast<int>(reference.size())) engine.Run(1000);
  }
  std::vector<double> ref_pe, ref_ke, ref_press;
  for (const ThermoRow& row : reference) {
    ref_pe.push_back(row.pe);
    ref_ke.push_back(row.ke);
    ref_press.push_back(row.press);
  }
  EXPECT_NEAR(Mean(pe), Mean(ref_pe), 0.02);
  EXPECT_NEAR(Mean(ke), Mean(ref_ke), 0.02);
  EXPECT_NEAR(Mean(press), Mean(ref_press), 0.05);
  std::cout << "T3 mean pe " << Mean(pe) << " lammps " << Mean(ref_pe) << " stderr " << Stderr(pe)
            << "\nT3 mean ke " << Mean(ke) << " lammps " << Mean(ref_ke) << " stderr " << Stderr(ke)
            << "\nT3 mean P " << Mean(press) << " lammps " << Mean(ref_press) << " stderr "
            << Stderr(press) << "\n";
}

TEST(GpuLj, ClusterTileMatchesOracle) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  tilemd::ref::System system;
  // reach is about (2.5 + 0.3) * (1 + 1e-4). floor(8 / reach) is 2, and the
  // engine requires 3 cells on each axis. L = 12 leaves 4 cells. Atom 1 is the
  // periodic image of a contact with atom 0.
  system.box[0] = 12.0;
  system.box[1] = 12.0;
  system.box[2] = 12.0;
  system.shift = tilemd::ref::Shift::Force;
  system.pairs.push_back({0, 0, 1.0, 1.0, 2.5});
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> dist(0.5, 11.5);
  for (int i = 0; i < 40; ++i) {
    system.type.push_back(0);
    system.x.push_back(dist(rng));
    system.y.push_back(dist(rng));
    system.z.push_back(dist(rng));
  }
  system.x[0] = 0.2;
  system.x[1] = 11.8;
  system.y[0] = system.y[1] = 6.0;
  system.z[0] = system.z[1] = 6.0;
  const tilemd::ref::Observables expected = tilemd::ref::Compute(system);
  tilemd::HostSystem host = FromReference(system, 0.3, 64, true);
  host.use_cluster_tiles = true;
  tilemd::Engine engine(std::move(host));
  ExpectClose(engine.ComputeForces(), expected, "cluster tiles");
}

TEST(GpuLj, AttentionKernelMatchesOracle) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  const char* names[] = {"boundary_500", "cutoff_500", "twotype_2000", "melt_4000"};
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  for (const char* name : names) {
    tilemd::ref::System system =
        tilemd::ref::ReadExtxyz((dir / (std::string(name) + ".extxyz")).string());
    tilemd::ref::ReadSystemHeader((dir / (std::string(name) + ".json")).string(), system);
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
    tilemd::AttentionShift shift = tilemd::AttentionShift::None;
    if (system.shift == tilemd::ref::Shift::Energy) shift = tilemd::AttentionShift::Energy;
    if (system.shift == tilemd::ref::Shift::Force) shift = tilemd::AttentionShift::Force;
    const tilemd::AttentionResult got = tilemd::RunLjAttentionGpu(
        system.box, system.x.data(), system.y.data(), system.z.data(), system.type.data(),
        tilemd::ref::NumAtoms(system), pairs.data(), static_cast<int>(pairs.size()), shift);
    const tilemd::ref::Observables expected = tilemd::ref::Compute(system);
    const int n = tilemd::ref::NumAtoms(system);
    double max_delta = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double dx = got.fx[static_cast<size_t>(i)] - expected.fx[static_cast<size_t>(i)];
      const double dy = got.fy[static_cast<size_t>(i)] - expected.fy[static_cast<size_t>(i)];
      const double dz = got.fz[static_cast<size_t>(i)] - expected.fz[static_cast<size_t>(i)];
      max_delta = std::max(max_delta, std::sqrt(dx * dx + dy * dy + dz * dz));
      sum_sq += expected.fx[static_cast<size_t>(i)] * expected.fx[static_cast<size_t>(i)] +
                expected.fy[static_cast<size_t>(i)] * expected.fy[static_cast<size_t>(i)] +
                expected.fz[static_cast<size_t>(i)] * expected.fz[static_cast<size_t>(i)];
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(n));
    EXPECT_LT(max_delta / std::max(rms, 1.0), 1.0e-10) << name;
    EXPECT_NEAR(got.potential_energy, expected.potential_energy,
                1.0e-10 * (1.0 + std::abs(expected.potential_energy)))
        << name;
  }
}

TEST(GpuLj, DeviceClusterRebuildMatchesTheHostList) {
  if (!HasGpu()) GTEST_SKIP();
  const double box[3] = {8.0, 8.0, 8.0};
  const int n = 64;
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
  const double reach = 3.0;
  const tilemd::ClusterGrid host_grid =
      tilemd::BuildClusterGrid(box, x.data(), y.data(), z.data(), type.data(), n, reach);
  const tilemd::ClusterPairs host_pairs = tilemd::BuildClusterPairs(host_grid);
  const tilemd::HostClusterList device =
      tilemd::RebuildClusterListOnDevice(box, x.data(), y.data(), z.data(), type.data(), n, reach);
  EXPECT_EQ(device.ncluster, host_grid.ncluster);
  EXPECT_EQ(device.type, host_grid.type);
  EXPECT_EQ(device.gid, host_grid.gid);
  EXPECT_EQ(device.ci_start, host_pairs.ci_start);
  EXPECT_EQ(device.cj, host_pairs.cj);
  EXPECT_EQ(device.shift, host_pairs.shift);
}

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
  const double rms = std::sqrt(sum_sq / static_cast<double>(std::max(n, 1)));
  return max_delta / std::max(rms, 1.0e-30);
}

TEST(GpuLj, EamFuncflKernelMatchesTheLaneWalk) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
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
  std::vector<int> type(static_cast<size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    in >> x[static_cast<size_t>(i)] >> y[static_cast<size_t>(i)] >> z[static_cast<size_t>(i)] >>
        fx >> fy >> fz;
  }
  ASSERT_TRUE(static_cast<bool>(in));
  tilemd::EamFuncfl lanes(table.string());
  tilemd::EamFuncfl device_potential(table.string());
  const tilemd::AttentionResult host =
      tilemd::WarpLaneAttention(lanes, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
  const tilemd::AttentionResult got = tilemd::RunEamFuncflGpu(
      device_potential, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
  const double error = RelativeForce(got.fx, got.fy, got.fz, host.fx, host.fy, host.fz);
  EXPECT_LT(error, 1.0e-9);
  EXPECT_NEAR(got.potential_energy, host.potential_energy,
              1.0e-9 * (1.0 + std::abs(host.potential_energy)));
}

TEST(GpuLj, EamSetflKernelMatchesTheLaneWalk) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  const std::filesystem::path bundled =
      std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "potentials" / "CuNi.eam.alloy";
  const std::filesystem::path installed(
      "C:/ProgramData/anaconda3/Lib/site-packages/lammps/share/lammps/potentials/CuNi.eam.alloy");
  const std::filesystem::path ref = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                    "validation" / "lammps_ref" / "cuni_setfl_4.txt";
  std::string table;
  if (std::filesystem::exists(bundled))
    table = bundled.string();
  else if (std::filesystem::exists(installed))
    table = installed.string();
  if (table.empty() || !std::filesystem::exists(ref)) {
    GTEST_SKIP() << "CuNi setfl table is not available";
  }
  std::ifstream in(ref);
  int n = 0;
  double box[3];
  double energy = 0.0;
  in >> n >> box[0] >> box[1] >> box[2] >> energy;
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n));
  std::vector<int> type(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    in >> type[static_cast<size_t>(i)] >> x[static_cast<size_t>(i)] >> y[static_cast<size_t>(i)] >>
        z[static_cast<size_t>(i)];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    in >> fx >> fy >> fz;
  }
  ASSERT_TRUE(static_cast<bool>(in));
  tilemd::EamSetfl lanes(table);
  tilemd::EamSetfl device_potential(table);
  lanes.SetAtoms(type.data(), n);
  const tilemd::AttentionResult host =
      tilemd::WarpLaneAttention(lanes, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
  const tilemd::AttentionResult got = tilemd::RunEamSetflGpu(
      device_potential, box, x.data(), y.data(), z.data(), type.data(), n, 0.3);
  const double error = RelativeForce(got.fx, got.fy, got.fz, host.fx, host.fy, host.fz);
  EXPECT_LT(error, 1.0e-9);
  EXPECT_NEAR(got.potential_energy, host.potential_energy,
              1.0e-9 * (1.0 + std::abs(host.potential_energy)));
}

tilemd::ref::System LatticeBox() {
  tilemd::ref::System system;
  system.box[0] = 12.0;
  system.box[1] = 12.0;
  system.box[2] = 12.0;
  system.shift = tilemd::ref::Shift::Force;
  system.pairs.push_back({0, 0, 1.0, 1.0, 2.5});
  for (int iz = 0; iz < 5; ++iz) {
    for (int iy = 0; iy < 5; ++iy) {
      for (int ix = 0; ix < 5; ++ix) {
        system.type.push_back(0);
        system.x.push_back(1.0 + 2.2 * ix);
        system.y.push_back(1.0 + 2.2 * iy);
        system.z.push_back(1.0 + 2.2 * iz);
      }
    }
  }
  return system;
}

double MaxAbsDelta(const std::vector<double>& a, const std::vector<double>& b) {
  double max_delta = 0.0;
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    max_delta =
        std::max(max_delta, std::abs(a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]));
  }
  return max_delta;
}

// Device fused leapfrog against the double-precision CPU leapfrog. The pair
// geometry on the device is the cell-local shadow, so the residual is that
// rounding, not a second integrator.
TEST(GpuLj, FusedLeapfrogMatchesTheCpuFor100Steps) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  const tilemd::ref::System origin = LatticeBox();
  const int n = static_cast<int>(origin.x.size());
  std::vector<double> vx(static_cast<size_t>(n));
  std::vector<double> vy(static_cast<size_t>(n));
  std::vector<double> vz(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] = 0.02 * ((i % 5) - 2);
    vy[static_cast<size_t>(i)] = 0.015 * ((i % 4) - 1);
    vz[static_cast<size_t>(i)] = 0.01 * ((i % 3) - 1);
  }
  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  tilemd::ref::System cpu_system = origin;
  tilemd::ref::Dynamics cpu = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::PrepareLeapfrog(cpu_system, cpu, dt, mass);
  for (int step = 0; step < 100; ++step) {
    tilemd::ref::LeapfrogStep(cpu_system, cpu, dt, mass);
  }

  const bool rebuilds[] = {true, false};
  for (bool rebuild_every_step : rebuilds) {
    tilemd::HostSystem host = FromReference(origin, 0.3, 0, rebuild_every_step);
    host.use_cluster_tiles = true;
    host.dt = dt;
    host.mass = mass;
    host.vx = vx;
    host.vy = vy;
    host.vz = vz;
    tilemd::Engine engine(std::move(host));
    engine.Run(100);
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> gvx;
    std::vector<double> gvy;
    std::vector<double> gvz;
    engine.DownloadPositions(x, y, z);
    engine.DownloadVelocities(gvx, gvy, gvz);
    const double max_x =
        std::max(MaxAbsDelta(x, cpu.x), std::max(MaxAbsDelta(y, cpu.y), MaxAbsDelta(z, cpu.z)));
    const double max_v = std::max(MaxAbsDelta(gvx, cpu.vx),
                                  std::max(MaxAbsDelta(gvy, cpu.vy), MaxAbsDelta(gvz, cpu.vz)));
    std::cout << std::scientific << std::setprecision(8) << "fused 100 rebuild " << std::boolalpha
              << rebuild_every_step << " moved " << MaxAbsDelta(cpu.x, origin.x) << " max |dx| "
              << max_x << " max |dv| " << max_v << "\n";
    ASSERT_EQ(x.size(), static_cast<size_t>(n));
    EXPECT_GT(MaxAbsDelta(cpu.x, origin.x), 1.0e-3);
    EXPECT_TRUE(std::isfinite(max_x));
    EXPECT_TRUE(std::isfinite(max_v));
    EXPECT_LT(max_x, 1.0e-7) << "rebuild_every_step " << rebuild_every_step;
    EXPECT_LT(max_v, 1.0e-7) << "rebuild_every_step " << rebuild_every_step;
  }
}

TEST(GpuLj, FusedThermoMatchesTheCpuLeapfrog) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  const tilemd::ref::System origin = LatticeBox();
  const int n = static_cast<int>(origin.x.size());
  std::vector<double> vx(static_cast<size_t>(n));
  std::vector<double> vy(static_cast<size_t>(n));
  std::vector<double> vz(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] = 0.02 * ((i % 5) - 2);
    vy[static_cast<size_t>(i)] = 0.015 * ((i % 4) - 1);
    vz[static_cast<size_t>(i)] = 0.01 * ((i % 3) - 1);
  }
  constexpr double dt = 0.005;
  constexpr double mass = 1.0;
  constexpr int every = 2;
  constexpr int nstep = 8;
  const std::vector<int> expect_steps = tilemd::TileThermoSteps(nstep, every);
  std::vector<double> cpu_ke;
  std::vector<double> cpu_pe;
  std::vector<std::array<double, 6>> cpu_w;
  tilemd::ref::System cpu_system = origin;
  tilemd::ref::Dynamics cpu = tilemd::ref::MakeDynamics(origin, vx, vy, vz);
  tilemd::ref::PrepareLeapfrog(cpu_system, cpu, dt, mass);
  auto record = [&](int step) {
    std::vector<double> ivx;
    std::vector<double> ivy;
    std::vector<double> ivz;
    tilemd::ref::IntegerTimeVelocity(cpu_system, cpu, dt, mass, ivx, ivy, ivz);
    tilemd::ref::System viewed = origin;
    viewed.x = cpu.x;
    viewed.y = cpu.y;
    viewed.z = cpu.z;
    const tilemd::ref::Observables obs = tilemd::ref::Compute(viewed);
    double ke = 0.0;
    for (int i = 0; i < n; ++i) {
      ke += 0.5 * (ivx[static_cast<size_t>(i)] * ivx[static_cast<size_t>(i)] +
                   ivy[static_cast<size_t>(i)] * ivy[static_cast<size_t>(i)] +
                   ivz[static_cast<size_t>(i)] * ivz[static_cast<size_t>(i)]);
    }
    cpu_ke.push_back(ke);
    cpu_pe.push_back(obs.potential_energy);
    std::array<double, 6> virial{};
    for (int axis = 0; axis < 6; ++axis) virial[static_cast<size_t>(axis)] = obs.virial[axis];
    cpu_w.push_back(virial);
    (void)step;
  };
  for (int step = 0; step < nstep; ++step) {
    if (tilemd::TileRecordsThermo(step, every)) record(step);
    tilemd::ref::LeapfrogStep(cpu_system, cpu, dt, mass);
  }
  if (tilemd::TileRecordsThermo(nstep, every)) record(nstep);
  ASSERT_EQ(cpu_ke.size(), expect_steps.size());

  const bool rebuilds[] = {true, false};
  for (bool rebuild_every_step : rebuilds) {
    tilemd::HostSystem host = FromReference(origin, 0.3, 0, rebuild_every_step);
    host.use_cluster_tiles = true;
    host.dt = dt;
    host.mass = mass;
    host.thermo_every = every;
    host.vx = vx;
    host.vy = vy;
    host.vz = vz;
    tilemd::Engine engine(std::move(host));
    engine.Run(nstep);
    const std::vector<tilemd::ThermoSample> got = engine.DrainThermo();
    ASSERT_EQ(got.size(), expect_steps.size()) << "rebuild_every_step " << rebuild_every_step;
    double max_pe = 0.0;
    double max_ke = 0.0;
    double max_w = 0.0;
    for (size_t s = 0; s < got.size(); ++s) {
      EXPECT_EQ(got[s].step, expect_steps[s]) << "rebuild_every_step " << rebuild_every_step;
      max_pe = std::max(max_pe, std::abs(got[s].potential_energy - cpu_pe[s]));
      max_ke = std::max(max_ke, std::abs(got[s].kinetic_energy - cpu_ke[s]));
      for (int axis = 0; axis < 6; ++axis) {
        max_w =
            std::max(max_w, std::abs(got[s].virial[axis] - cpu_w[s][static_cast<size_t>(axis)]));
      }
    }
    std::cout << std::scientific << std::setprecision(8) << "fused thermo rebuild "
              << std::boolalpha << rebuild_every_step << " max |dpe| " << max_pe << " max |dke| "
              << max_ke << " max |dw| " << max_w << "\n";
    EXPECT_LT(max_pe, 1.0e-6) << "rebuild_every_step " << rebuild_every_step;
    EXPECT_LT(max_ke, 1.0e-6) << "rebuild_every_step " << rebuild_every_step;
    EXPECT_LT(max_w, 1.0e-6) << "rebuild_every_step " << rebuild_every_step;
  }
}
