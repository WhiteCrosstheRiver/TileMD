#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "lj_reference.hpp"

namespace {

using tilemd::ref::Compute;
using tilemd::ref::Observables;
using tilemd::ref::PairCoeff;
using tilemd::ref::ReadLammpsDump;
using tilemd::ref::Shift;
using tilemd::ref::System;
using tilemd::ref::WriteExtxyz;
using tilemd::ref::WriteObservablesJson;

constexpr double kRhoMelt = 0.8442;
constexpr double kEpsilon = 1.0;
constexpr double kSigma = 1.0;
constexpr double kCutoff = 2.5;
constexpr std::uint32_t kSeed = 20261018;

PairCoeff Mono(int type_i, int type_j, double epsilon, double sigma, double cutoff) {
  PairCoeff coeff;
  coeff.type_i = type_i;
  coeff.type_j = type_j;
  coeff.epsilon = epsilon;
  coeff.sigma = sigma;
  coeff.cutoff = cutoff;
  return coeff;
}

void SetMono(System& system) {
  system.shift = Shift::None;
  system.pairs = {Mono(0, 0, kEpsilon, kSigma, kCutoff)};
}

void WriteFixture(const std::filesystem::path& dir, const std::string& name, const System& system) {
  const Observables obs = Compute(system);
  WriteExtxyz((dir / (name + ".extxyz")).string(), system);
  WriteObservablesJson((dir / (name + ".json")).string(), name, system, obs);
  const auto bytes = std::filesystem::file_size(dir / (name + ".json"));
  if (bytes > 1000 * 1000) {
    throw std::runtime_error(name + " JSON exceeds 1 MB");
  }
}

System MakeFcc(int cells, double rho) {
  const double a = std::cbrt(4.0 / rho);
  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  System system;
  system.box[0] = system.box[1] = system.box[2] = cells * a;
  SetMono(system);
  for (int iz = 0; iz < cells; ++iz) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int ix = 0; ix < cells; ++ix) {
        for (const auto& site : basis) {
          system.type.push_back(0);
          system.x.push_back((ix + site[0]) * a);
          system.y.push_back((iy + site[1]) * a);
          system.z.push_back((iz + site[2]) * a);
        }
      }
    }
  }
  return system;
}

System MakeRandom(int n, double rho, std::mt19937* rng) {
  System system;
  const double length = std::cbrt(static_cast<double>(n) / rho);
  system.box[0] = system.box[1] = system.box[2] = length;
  SetMono(system);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  system.type.assign(n, 0);
  system.x.resize(n);
  system.y.resize(n);
  system.z.resize(n);
  for (int i = 0; i < n; ++i) {
    system.x[i] = unit(*rng) * length;
    system.y[i] = unit(*rng) * length;
    system.z[i] = unit(*rng) * length;
  }
  return system;
}

System MakeBoundary() {
  constexpr int kN = 500;
  constexpr double kLength = 10.0;
  constexpr double kEps = 1.0e-3;
  System system;
  system.box[0] = system.box[1] = system.box[2] = kLength;
  SetMono(system);
  const double faces[2] = {kEps, kLength - kEps};
  for (double x : faces) {
    for (double y : faces) {
      for (double z : faces) {
        system.type.push_back(0);
        system.x.push_back(x);
        system.y.push_back(y);
        system.z.push_back(z);
      }
    }
  }
  const double mid = 0.5 * kLength;
  for (int axis = 0; axis < 3; ++axis) {
    for (double a : faces) {
      for (double b : faces) {
        double coord[3] = {mid, mid, mid};
        coord[(axis + 1) % 3] = a;
        coord[(axis + 2) % 3] = b;
        system.type.push_back(0);
        system.x.push_back(coord[0]);
        system.y.push_back(coord[1]);
        system.z.push_back(coord[2]);
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (double face : faces) {
      double coord[3] = {mid, mid, mid};
      coord[axis] = face;
      system.type.push_back(0);
      system.x.push_back(coord[0]);
      system.y.push_back(coord[1]);
      system.z.push_back(coord[2]);
    }
  }
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> interior(kEps, kLength - kEps);
  while (static_cast<int>(system.x.size()) < kN) {
    system.type.push_back(0);
    system.x.push_back(interior(rng));
    system.y.push_back(interior(rng));
    system.z.push_back(interior(rng));
  }
  return system;
}

System MakeCutoff() {
  constexpr double kRc = 2.5;
  constexpr double kSpacing = 8.0;
  constexpr double kMargin = 4.0;
  constexpr int kGrid[3] = {5, 5, 10};
  System system;
  system.box[0] = kMargin + kGrid[0] * kSpacing;
  system.box[1] = kMargin + kGrid[1] * kSpacing;
  system.box[2] = kMargin + kGrid[2] * kSpacing;
  SetMono(system);
  int placed = 0;
  for (int iz = 0; iz < kGrid[2]; ++iz) {
    for (int iy = 0; iy < kGrid[1]; ++iy) {
      for (int ix = 0; ix < kGrid[0]; ++ix) {
        const double x = kMargin + ix * kSpacing;
        const double y = kMargin + iy * kSpacing;
        const double z = kMargin + iz * kSpacing;
        const double separation = (placed % 2 == 0) ? (kRc - 1.0e-4) : (kRc + 1.0e-4);
        for (int end = 0; end < 2; ++end) {
          system.type.push_back(0);
          system.x.push_back(x + (end == 0 ? 0.0 : separation));
          system.y.push_back(y);
          system.z.push_back(z);
        }
        ++placed;
      }
    }
  }
  return system;
}

System MakeTwoType() {
  constexpr int kEach = 1000;
  constexpr double kRho = 0.5;
  System system;
  const double length = std::cbrt((2.0 * kEach) / kRho);
  system.box[0] = system.box[1] = system.box[2] = length;
  system.shift = Shift::None;
  system.pairs = {Mono(0, 0, 1.0, 1.0, 2.5), Mono(1, 1, 0.5, 0.8, 2.0), Mono(0, 1, 1.5, 1.2, 3.0)};
  std::mt19937 rng(kSeed + 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int type = 0; type < 2; ++type) {
    for (int i = 0; i < kEach; ++i) {
      system.type.push_back(type);
      system.x.push_back(unit(rng) * length);
      system.y.push_back(unit(rng) * length);
      system.z.push_back(unit(rng) * length);
    }
  }
  return system;
}

void ReadLammpsThermo(const std::filesystem::path& log, int n_atoms, Observables* obs) {
  // units lj normalizes every thermo column by N (thermo_modify norm yes).
  std::ifstream in(log);
  if (!in) throw std::runtime_error("cannot open " + log.string());
  bool found = false;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream row(line);
    double step = 0.0;
    double pe = 0.0;
    double virial[6];
    if (!(row >> step >> pe >> virial[0] >> virial[1] >> virial[2] >> virial[3] >> virial[4] >>
          virial[5])) {
      continue;
    }
    if (step != 1000.0) continue;
    // The pe column is per atom under units lj. User variables such as v_wxx
    // are not rescaled by thermo_modify norm.
    obs->potential_energy = pe * n_atoms;
    for (int component = 0; component < 6; ++component) obs->virial[component] = virial[component];
    found = true;
  }
  if (!found) throw std::runtime_error("LAMMPS log has no step-1000 thermo line");
}

void WriteMelt(const std::filesystem::path& fixture_dir,
               const std::filesystem::path& validation_dir, const std::string& dump_path) {
  auto dump = ReadLammpsDump(dump_path, Shift::None, {Mono(0, 0, kEpsilon, kSigma, kCutoff)});
  if (tilemd::ref::NumAtoms(dump.system) != 4000) {
    throw std::runtime_error("melt dump does not contain 4000 atoms");
  }
  const auto log = std::filesystem::path(dump_path).parent_path() / "log.lammps";
  if (std::filesystem::exists(log)) {
    ReadLammpsThermo(log, tilemd::ref::NumAtoms(dump.system), &dump.observables);
  }
  WriteFixture(fixture_dir, "melt_4000", dump.system);
  std::filesystem::create_directories(validation_dir);
  WriteObservablesJson((validation_dir / "melt_4000_lammps.json").string(), "melt_4000_lammps",
                       dump.system, dump.observables);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr << "usage: tilemd_make_fixtures <fixture_dir> <validation_dir> [lammps_dump]\n";
      return 2;
    }
    const std::filesystem::path fixture_dir(argv[1]);
    const std::filesystem::path validation_dir(argv[2]);
    std::filesystem::create_directories(fixture_dir);
    std::mt19937 rng(kSeed);
    WriteFixture(fixture_dir, "fcc_500", MakeFcc(5, kRhoMelt));
    WriteFixture(fixture_dir, "boundary_500", MakeBoundary());
    WriteFixture(fixture_dir, "cutoff_500", MakeCutoff());
    WriteFixture(fixture_dir, "twotype_2000", MakeTwoType());
    WriteFixture(fixture_dir, "dilute_1000", MakeRandom(1000, 0.05, &rng));
    WriteFixture(fixture_dir, "dense_2000", MakeRandom(2000, 1.2, &rng));
    if (argc == 4) WriteMelt(fixture_dir, validation_dir, argv[3]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "tilemd_make_fixtures: " << error.what() << "\n";
    return 1;
  }
}
