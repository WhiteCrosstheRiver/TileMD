#include "io/input.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "io/run.hpp"

namespace {

const char* kMelt = R"(
units          lj
box            fcc 2 2 2 0.8442
types          1
lj             0 0  epsilon 1.0  sigma 1.0  cutoff 2.5
lj_shift       force
mass           0 1.0
velocity       1.44 seed 99
skin           0.3
timestep       0.005
ensemble       nve
thermo_every   100
dump_every     0
graph_unroll   50
run            10
)";

double Momentum(const std::vector<double>& v) {
  double sum = 0.0;
  for (double value : v) sum += value;
  return sum;
}

}  // namespace

TEST(Input, FccMeltMatchesTheSketch) {
  const tilemd::SimulationInput input = tilemd::ParseInput(kMelt, ".");
  const int n = tilemd::ref::NumAtoms(input.system);
  EXPECT_EQ(n, 32);
  const double volume = input.system.box[0] * input.system.box[1] * input.system.box[2];
  EXPECT_NEAR(static_cast<double>(n) / volume, 0.8442, 1.0e-12);
  EXPECT_NEAR(Momentum(input.vx), 0.0, 1.0e-10);
  EXPECT_NEAR(Momentum(input.vy), 0.0, 1.0e-10);
  EXPECT_NEAR(Momentum(input.vz), 0.0, 1.0e-10);
  EXPECT_EQ(input.system.shift, tilemd::ref::Shift::Force);
  EXPECT_FALSE(input.langevin);
  EXPECT_EQ(input.steps, 10);
  EXPECT_DOUBLE_EQ(input.skin, 0.3);
  EXPECT_DOUBLE_EQ(input.timestep, 0.005);
  EXPECT_EQ(input.thermo_every, 100);
  EXPECT_EQ(input.graph_unroll, 50);
  EXPECT_DOUBLE_EQ(input.mass, 1.0);

  tilemd::ref::System system = input.system;
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  for (int step = 0; step < 5; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, input.timestep, input.mass);
  }
  EXPECT_TRUE(std::isfinite(state.x[0]));
  EXPECT_GT(std::abs(state.vx[0]) + std::abs(state.vy[0]) + std::abs(state.vz[0]), 0.0);
}

TEST(Input, VelocitiesRepeatForTheSameSeed) {
  std::string text = kMelt;
  const auto langevin = text.find("ensemble       nve");
  text.replace(langevin, std::string("ensemble       nve").size(),
               "ensemble       langevin 1.0 0.5 7");
  const tilemd::SimulationInput first = tilemd::ParseInput(text, ".");
  const tilemd::SimulationInput second = tilemd::ParseInput(text, ".");
  ASSERT_TRUE(first.langevin);
  EXPECT_DOUBLE_EQ(first.thermo.temperature, 1.0);
  EXPECT_DOUBLE_EQ(first.thermo.gamma, 0.5);
  EXPECT_EQ(first.thermo.seed, 7u);
  EXPECT_EQ(first.vx, second.vx);
  std::string other = text;
  const auto seed = other.find("seed 99");
  other.replace(seed, 7, "seed 100");
  const tilemd::SimulationInput changed = tilemd::ParseInput(other, ".");
  EXPECT_NE(first.vx, changed.vx);
}

TEST(Input, ReferenceRunMatchesVerlet) {
  std::string text = kMelt;
  const auto steps = text.find("run            10");
  text.replace(steps, std::string("run            10").size(), "run            20");
  const auto thermo = text.find("thermo_every   100");
  text.replace(thermo, std::string("thermo_every   100").size(), "thermo_every   10");
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input);
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0].step, 0);
  EXPECT_EQ(rows[1].step, 10);
  EXPECT_EQ(rows[2].step, 20);
  const double e0 = rows[0].total_per_atom;
  const double e1 = rows[2].total_per_atom;
  std::cout << "cpu input nve e0 " << e0 << " e20 " << e1 << "\n";
  // A fresh T=1.44 lattice oscillates. This bound records that swing.
  const double swing = std::abs(e1 - e0) / std::abs(e0);
  EXPECT_GT(swing, 8.0e-4);
  EXPECT_LT(swing, 2.0e-3);
  const int n = tilemd::ref::NumAtoms(input.system);
  const double kinetic0 = rows[0].kinetic_per_atom * static_cast<double>(n);
  const double dof = static_cast<double>(3 * n - 3);
  EXPECT_NEAR(rows[0].temperature, 2.0 * kinetic0 / dof, 1.0e-12);
  tilemd::ref::System system = input.system;
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  for (int step = 0; step < 20; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, input.timestep, input.mass);
  }
  system.x = state.x;
  system.y = state.y;
  system.z = state.z;
  const tilemd::ref::Observables obs = tilemd::ref::Compute(system);
  double kinetic = 0.0;
  for (int i = 0; i < n; ++i) {
    kinetic += 0.5 * input.mass *
               (state.vx[static_cast<size_t>(i)] * state.vx[static_cast<size_t>(i)] +
                state.vy[static_cast<size_t>(i)] * state.vy[static_cast<size_t>(i)] +
                state.vz[static_cast<size_t>(i)] * state.vz[static_cast<size_t>(i)]);
  }
  const double total = (obs.potential_energy + kinetic) / static_cast<double>(n);
  EXPECT_NEAR(rows[2].total_per_atom, total, 1.0e-12);
}

TEST(Input, ZeroTemperatureLangevinCools) {
  std::string text = kMelt;
  const auto ensemble = text.find("ensemble       nve");
  text.replace(ensemble, std::string("ensemble       nve").size(),
               "ensemble       langevin 0 2.0 3");
  const auto steps = text.find("run            10");
  text.replace(steps, std::string("run            10").size(), "run            40");
  const auto thermo = text.find("thermo_every   100");
  text.replace(thermo, std::string("thermo_every   100").size(), "thermo_every   40");
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input);
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_LT(rows[1].kinetic_per_atom, 0.5 * rows[0].kinetic_per_atom);
}

TEST(Input, UnknownKeyIsRejected) {
  EXPECT_THROW(tilemd::ParseInput("nope 1\n", "."), std::runtime_error);
}

TEST(Input, DumpWritesAFrameOnTheRequestedSteps) {
  std::string text = kMelt;
  const auto dump = text.find("dump_every     0");
  text.replace(dump, std::string("dump_every     0").size(), "dump_every     10");
  const auto steps = text.find("run            10");
  text.replace(steps, std::string("run            10").size(), "run            20");
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "tilemd_input_dump.extxyz";
  std::filesystem::remove(path);
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input, path.string());
  EXPECT_EQ(rows.size(), 1u);
  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  int frames = 0;
  bool saw0 = false;
  bool saw10 = false;
  bool saw20 = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line == "32") ++frames;
    if (line.find("step=0 ") != std::string::npos) saw0 = true;
    if (line.find("step=10 ") != std::string::npos) saw10 = true;
    if (line.find("step=20 ") != std::string::npos) saw20 = true;
  }
  EXPECT_EQ(frames, 3);
  EXPECT_TRUE(saw0);
  EXPECT_TRUE(saw10);
  EXPECT_TRUE(saw20);
  in.close();
  const tilemd::ref::ExtxyzFrame first = tilemd::ref::ReadExtxyzFrame(path.string(), 0);
  const tilemd::ref::ExtxyzFrame later = tilemd::ref::ReadExtxyzFrame(path.string(), 2);
  ASSERT_EQ(tilemd::ref::NumAtoms(first.system), 32);
  ASSERT_EQ(first.vx.size(), 32u);
  tilemd::ref::System system = input.system;
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, input.vx, input.vy, input.vz);
  double max_dx = 0.0;
  double max_dv = 0.0;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(first.system.type[static_cast<size_t>(i)], 0);
    max_dx = std::max(
        max_dx, std::abs(first.system.x[static_cast<size_t>(i)] - state.x[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(first.system.y[static_cast<size_t>(i)] - state.y[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(first.system.z[static_cast<size_t>(i)] - state.z[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(first.vx[static_cast<size_t>(i)] - state.vx[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(first.vy[static_cast<size_t>(i)] - state.vy[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(first.vz[static_cast<size_t>(i)] - state.vz[static_cast<size_t>(i)]));
  }
  EXPECT_LT(max_dx, 1.0e-12);
  EXPECT_LT(max_dv, 1.0e-12);
  for (int step = 0; step < 20; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, input.timestep, input.mass);
  }
  max_dx = 0.0;
  max_dv = 0.0;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(later.system.type[static_cast<size_t>(i)], 0);
    max_dx = std::max(
        max_dx, std::abs(later.system.x[static_cast<size_t>(i)] - state.x[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(later.system.y[static_cast<size_t>(i)] - state.y[static_cast<size_t>(i)]));
    max_dx = std::max(
        max_dx, std::abs(later.system.z[static_cast<size_t>(i)] - state.z[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(later.vx[static_cast<size_t>(i)] - state.vx[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(later.vy[static_cast<size_t>(i)] - state.vy[static_cast<size_t>(i)]));
    max_dv = std::max(
        max_dv, std::abs(later.vz[static_cast<size_t>(i)] - state.vz[static_cast<size_t>(i)]));
  }
  EXPECT_LT(max_dx, 1.0e-12);
  EXPECT_LT(max_dv, 1.0e-12);
  std::filesystem::remove(path);
}

TEST(Input, StructureLoadsPositions) {
  const std::filesystem::path dir = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "fixtures";
  const std::string text =
      "structure cutoff_500.extxyz\n"
      "lj 0 0 epsilon 1.0 sigma 1.0 cutoff 2.5\n";
  const tilemd::SimulationInput input = tilemd::ParseInput(text, dir.string());
  EXPECT_EQ(tilemd::ref::NumAtoms(input.system), 500);
  EXPECT_EQ(Momentum(input.vx), 0.0);
}

TEST(Input, StructureKeepsVelocitiesWhenNoVelocityLine) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "tilemd_input_vel.extxyz";
  {
    std::ofstream out(path);
    out << "1\n";
    out << "Lattice=\"8 0 0 0 8 0 0 0 8\" Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 "
           "shift=none pbc=\"T T T\"\n";
    out << "A 1.25 2.5 3.75 0.125 -0.25 0.5 0\n";
  }
  const std::string text = "structure " + path.string() +
                           "\n"
                           "lj 0 0 epsilon 1.0 sigma 1.0 cutoff 2.5\n";
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  EXPECT_DOUBLE_EQ(input.vx[0], 0.125);
  EXPECT_DOUBLE_EQ(input.vy[0], -0.25);
  EXPECT_DOUBLE_EQ(input.vz[0], 0.5);
  std::filesystem::remove(path);
}

void CopperVelocity(int n, std::vector<double>* vx, std::vector<double>* vy,
                    std::vector<double>* vz) {
  vx->assign(static_cast<size_t>(n), 0.0);
  vy->assign(static_cast<size_t>(n), 0.0);
  vz->assign(static_cast<size_t>(n), 0.0);
  double mx = 0.0;
  double my = 0.0;
  double mz = 0.0;
  for (int i = 0; i < n; ++i) {
    (*vx)[static_cast<size_t>(i)] = 0.8 * ((i % 7) - 3);
    (*vy)[static_cast<size_t>(i)] = 0.5 * ((i % 5) - 2);
    (*vz)[static_cast<size_t>(i)] = 0.4 * ((i % 3) - 1);
    mx += (*vx)[static_cast<size_t>(i)];
    my += (*vy)[static_cast<size_t>(i)];
    mz += (*vz)[static_cast<size_t>(i)];
  }
  mx /= static_cast<double>(n);
  my /= static_cast<double>(n);
  mz /= static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    (*vx)[static_cast<size_t>(i)] -= mx;
    (*vy)[static_cast<size_t>(i)] -= my;
    (*vz)[static_cast<size_t>(i)] -= mz;
  }
}

TEST(Input, EamRunMatchesLammpsCopper) {
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
  double energy0 = 0.0;
  initial >> n >> box[0] >> box[1] >> box[2] >> energy0;
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n));
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
  CopperVelocity(n, &vx, &vy, &vz);
  const std::filesystem::path xyz =
      std::filesystem::temp_directory_path() / "tilemd_cu_input.extxyz";
  {
    std::ofstream out(xyz);
    out.setf(std::ios::scientific);
    out.precision(17);
    out << n << "\n";
    out << "Lattice=\"" << box[0] << " 0 0 0 " << box[1] << " 0 0 0 " << box[2]
        << "\" Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 shift=none pbc=\"T T T\"\n";
    for (int i = 0; i < n; ++i) {
      out << "A " << x[static_cast<size_t>(i)] << " " << y[static_cast<size_t>(i)] << " "
          << z[static_cast<size_t>(i)] << " " << vx[static_cast<size_t>(i)] << " "
          << vy[static_cast<size_t>(i)] << " " << vz[static_cast<size_t>(i)] << " 0\n";
    }
  }
  const std::filesystem::path dump =
      std::filesystem::temp_directory_path() / "tilemd_cu_input.dump.extxyz";
  std::filesystem::remove(dump);
  const std::string text = "units metal\nstructure " + xyz.string() + "\neam " + table.string() +
                           "\ntimestep 0.001\nthermo_every 100\ndump_every 100\nrun 100\n";
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  EXPECT_DOUBLE_EQ(input.mass, 63.55);
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input, dump.string());
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_NEAR(rows[0].potential_per_atom * static_cast<double>(n), energy0,
              1.0e-8 * (1.0 + std::abs(energy0)));
  const tilemd::ref::ExtxyzFrame later = tilemd::ref::ReadExtxyzFrame(dump.string(), 1);
  std::ifstream in(ref);
  int n_ref = 0;
  double box_ref[3];
  double dt_ref = 0.0;
  int steps_ref = 0;
  double energy = 0.0;
  in >> n_ref >> box_ref[0] >> box_ref[1] >> box_ref[2] >> dt_ref >> steps_ref >> energy;
  ASSERT_EQ(n_ref, n);
  ASSERT_EQ(steps_ref, 100);
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
    const double dx = later.system.x[static_cast<size_t>(i)] - rx -
                      box[0] * std::rint((later.system.x[static_cast<size_t>(i)] - rx) / box[0]);
    const double dy = later.system.y[static_cast<size_t>(i)] - ry -
                      box[1] * std::rint((later.system.y[static_cast<size_t>(i)] - ry) / box[1]);
    const double dz = later.system.z[static_cast<size_t>(i)] - rz -
                      box[2] * std::rint((later.system.z[static_cast<size_t>(i)] - rz) / box[2]);
    max_dx = std::max(max_dx, std::sqrt(dx * dx + dy * dy + dz * dz));
    const double dvx = later.vx[static_cast<size_t>(i)] - rvx;
    const double dvy = later.vy[static_cast<size_t>(i)] - rvy;
    const double dvz = later.vz[static_cast<size_t>(i)] - rvz;
    max_dv = std::max(max_dv, std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz));
  }
  ASSERT_TRUE(static_cast<bool>(in));
  std::cout << std::scientific << std::setprecision(8) << "input eam nve max |dx| " << max_dx
            << " max |dv| " << max_dv << "\n";
  EXPECT_LT(max_dx, 1.0e-8);
  EXPECT_LT(max_dv, 1.0e-8);
  EXPECT_NEAR(rows[1].potential_per_atom * static_cast<double>(n), energy,
              1.0e-6 * (1.0 + std::abs(energy)));
  std::filesystem::remove(xyz);
  std::filesystem::remove(dump);
}

TEST(Input, EamDriftSlopeMatchesLammps) {
  const std::filesystem::path path = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                     "validation" / "lammps_ref" / "cu_u3_nve_drift.txt";
  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string header;
  std::getline(in, header);
  std::vector<double> time;
  std::vector<double> tilemd_energy;
  std::vector<double> lammps_energy;
  int step = 0;
  double sample_time = 0.0;
  double tilemd_value = 0.0;
  double lammps_value = 0.0;
  while (in >> step >> sample_time >> tilemd_value >> lammps_value) {
    time.push_back(sample_time);
    tilemd_energy.push_back(tilemd_value);
    lammps_energy.push_back(lammps_value);
  }
  ASSERT_EQ(time.size(), 1001u);
  EXPECT_EQ(step, 100000);
  double max_gap = 0.0;
  double span = 0.0;
  for (size_t i = 0; i < time.size(); ++i) {
    max_gap = std::max(max_gap, std::abs(tilemd_energy[i] - lammps_energy[i]));
    span = std::max(span, std::abs(tilemd_energy[i] - tilemd_energy[0]));
  }
  const auto fit = [](const std::vector<double>& xs, const std::vector<double>& ys) {
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
      mean_x += xs[i];
      mean_y += ys[i];
    }
    mean_x /= static_cast<double>(xs.size());
    mean_y /= static_cast<double>(ys.size());
    double numer = 0.0;
    double denom = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
      numer += (xs[i] - mean_x) * (ys[i] - mean_y);
      denom += (xs[i] - mean_x) * (xs[i] - mean_x);
    }
    return numer / denom;
  };
  const double tilemd_slope = fit(time, tilemd_energy);
  const double lammps_slope = fit(time, lammps_energy);
  EXPECT_LT(max_gap, 1.0e-12);
  EXPECT_LT(span, 2.0e-6);
  EXPECT_NEAR(tilemd_slope, -1.93926963e-9, 1.0e-14);
  EXPECT_NEAR(lammps_slope, tilemd_slope, 1.0e-16);
}

std::string InstalledCuNiTable() {
  const std::filesystem::path bundled =
      std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" / "potentials" / "CuNi.eam.alloy";
  if (std::filesystem::exists(bundled)) return bundled.string();
  const std::filesystem::path installed(
      "C:/ProgramData/anaconda3/Lib/site-packages/lammps/share/lammps/potentials/CuNi.eam.alloy");
  if (std::filesystem::exists(installed)) return installed.string();
  return {};
}

TEST(Input, SetflRunMatchesLammpsCuNi) {
  const std::string table = InstalledCuNiTable();
  const std::filesystem::path ref = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                    "validation" / "lammps_ref" / "cuni_setfl_nve_100.txt";
  if (table.empty() || !std::filesystem::exists(ref)) {
    GTEST_SKIP() << "CuNi setfl table is not available";
  }
  const double x0[4] = {10.0, 12.2, 10.4, 13.1};
  const double y0[4] = {10.0, 10.1, 12.5, 12.0};
  const double z0[4] = {10.0, 10.3, 10.2, 11.4};
  const int type0[4] = {0, 1, 0, 1};
  const double vx0[4] = {-1.2158939747208246, -0.41589397472082434, 0.3841060252791757,
                         1.1841060252791757};
  const double vy0[4] = {-0.75993373420051535, -0.25993373420051535, 0.24006626579948462,
                         0.74006626579948465};
  const double vz0[4] = {-0.29602650631979388, 0.10397349368020617, 0.50397349368020616,
                         -0.29602650631979388};
  const std::filesystem::path xyz =
      std::filesystem::temp_directory_path() / "tilemd_cuni_input.extxyz";
  {
    std::ofstream out(xyz);
    out.setf(std::ios::scientific);
    out.precision(17);
    out << "4\n";
    out << "Lattice=\"20 0 0 0 20 0 0 0 20\" Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 "
           "shift=none pbc=\"T T T\"\n";
    for (int i = 0; i < 4; ++i) {
      const char species = static_cast<char>('A' + type0[i]);
      out << species << " " << x0[i] << " " << y0[i] << " " << z0[i] << " " << vx0[i] << " "
          << vy0[i] << " " << vz0[i] << " " << type0[i] << "\n";
    }
  }
  const std::filesystem::path dump =
      std::filesystem::temp_directory_path() / "tilemd_cuni_input.dump.extxyz";
  std::filesystem::remove(dump);
  const std::string text = "units metal\nstructure " + xyz.string() + "\neam " + table +
                           "\ntimestep 0.001\nthermo_every 100\ndump_every 100\nrun 100\n";
  const tilemd::SimulationInput input = tilemd::ParseInput(text, ".");
  EXPECT_TRUE(input.setfl);
  EXPECT_EQ(input.mass_by_type.size(), 2u);
  EXPECT_NEAR(input.mass, 58.689, 1.0e-9);
  EXPECT_NEAR(input.mass_by_type[1], 63.546, 1.0e-9);
  const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input, dump.string());
  ASSERT_EQ(rows.size(), 2u);
  std::ifstream in(ref);
  int n = 0;
  double box[3];
  double dt_ref = 0.0;
  int steps_ref = 0;
  double energy = 0.0;
  in >> n >> box[0] >> box[1] >> box[2] >> dt_ref >> steps_ref >> energy;
  ASSERT_EQ(n, 4);
  ASSERT_EQ(steps_ref, 100);
  const tilemd::ref::ExtxyzFrame later = tilemd::ref::ReadExtxyzFrame(dump.string(), 1);
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
    const double dx = later.system.x[static_cast<size_t>(i)] - rx -
                      box[0] * std::rint((later.system.x[static_cast<size_t>(i)] - rx) / box[0]);
    const double dy = later.system.y[static_cast<size_t>(i)] - ry -
                      box[1] * std::rint((later.system.y[static_cast<size_t>(i)] - ry) / box[1]);
    const double dz = later.system.z[static_cast<size_t>(i)] - rz -
                      box[2] * std::rint((later.system.z[static_cast<size_t>(i)] - rz) / box[2]);
    max_dx = std::max(max_dx, std::sqrt(dx * dx + dy * dy + dz * dz));
    const double dvx = later.vx[static_cast<size_t>(i)] - rvx;
    const double dvy = later.vy[static_cast<size_t>(i)] - rvy;
    const double dvz = later.vz[static_cast<size_t>(i)] - rvz;
    max_dv = std::max(max_dv, std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz));
  }
  ASSERT_TRUE(static_cast<bool>(in));
  std::cout << std::scientific << std::setprecision(8) << "input setfl nve max |dx| " << max_dx
            << " max |dv| " << max_dv << " energy " << rows[1].potential_per_atom * 4.0
            << " lammps " << energy << "\n";
  EXPECT_LT(max_dx, 1.0e-8);
  EXPECT_LT(max_dv, 1.0e-8);
  EXPECT_NEAR(rows[1].potential_per_atom * 4.0, energy, 1.0e-6 * (1.0 + std::abs(energy)));
  std::filesystem::remove(xyz);
  std::filesystem::remove(dump);
}

TEST(Input, SetflDriftAgreesBeforeTheGasSeparates) {
  const std::filesystem::path path = std::filesystem::path(TILEMD_SOURCE_DIR) / "tests" /
                                     "validation" / "lammps_ref" / "cuni_setfl_nve_drift.txt";
  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string header;
  std::getline(in, header);
  int step = 0;
  double time = 0.0;
  double tilemd_energy = 0.0;
  double lammps_energy = 0.0;
  int count = 0;
  int last_step = 0;
  double early_gap = 0.0;
  double max_gap = 0.0;
  while (in >> step >> time >> tilemd_energy >> lammps_energy) {
    ++count;
    last_step = step;
    const double gap = std::abs(tilemd_energy - lammps_energy);
    max_gap = std::max(max_gap, gap);
    if (step <= 3000) early_gap = std::max(early_gap, gap);
  }
  EXPECT_EQ(count, 1001);
  EXPECT_EQ(last_step, 100000);
  EXPECT_LT(early_gap, 1.0e-12);
  std::cout << std::scientific << std::setprecision(8) << "setfl drift early gap " << early_gap
            << " max gap " << max_gap << "\n";
}
