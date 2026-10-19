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

double PairSigma(const System& system, int type_i, int type_j) {
  for (const PairCoeff& coeff : system.pairs) {
    if ((coeff.type_i == type_i && coeff.type_j == type_j) ||
        (coeff.type_i == type_j && coeff.type_j == type_i)) {
      return coeff.sigma;
    }
  }
  return kSigma;
}

double MinImage(double delta, double length) { return delta - length * std::rint(delta / length); }

bool Fits(const System& system, double x, double y, double z, int type) {
  constexpr double kMinFactor = 0.85;
  const int n = static_cast<int>(system.x.size());
  for (int i = 0; i < n; ++i) {
    const double dx = MinImage(x - system.x[i], system.box[0]);
    const double dy = MinImage(y - system.y[i], system.box[1]);
    const double dz = MinImage(z - system.z[i], system.box[2]);
    const double rmin = kMinFactor * PairSigma(system, type, system.type[i]);
    if (dx * dx + dy * dy + dz * dz < rmin * rmin) return false;
  }
  return true;
}

void RequireMinDistance(const System& system, const char* name) {
  const int n = static_cast<int>(system.x.size());
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double dx = MinImage(system.x[i] - system.x[j], system.box[0]);
      const double dy = MinImage(system.y[i] - system.y[j], system.box[1]);
      const double dz = MinImage(system.z[i] - system.z[j], system.box[2]);
      const double rmin = 0.85 * PairSigma(system, system.type[i], system.type[j]);
      if (dx * dx + dy * dy + dz * dz < rmin * rmin) {
        throw std::runtime_error(std::string(name) + " pair is closer than 0.85 sigma");
      }
    }
  }
}

bool Place(System* system, double x, double y, double z, int type) {
  if (!Fits(*system, x, y, z, type)) return false;
  system->type.push_back(type);
  system->x.push_back(x);
  system->y.push_back(y);
  system->z.push_back(z);
  return true;
}

System MakeRandom(int n, double rho, std::mt19937* rng) {
  System system;
  const double length = std::cbrt(static_cast<double>(n) / rho);
  system.box[0] = system.box[1] = system.box[2] = length;
  SetMono(system);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int attempts = 0;
  while (static_cast<int>(system.x.size()) < n) {
    if (++attempts > n * 100000) throw std::runtime_error("random fixture could not place an atom");
    Place(&system, unit(*rng) * length, unit(*rng) * length, unit(*rng) * length, 0);
  }
  return system;
}

System MakeDenseGrid(int n, double rho) {
  System system;
  const double length = std::cbrt(static_cast<double>(n) / rho);
  system.box[0] = system.box[1] = system.box[2] = length;
  SetMono(system);
  const int side = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(n))));
  const double spacing = length / static_cast<double>(side);
  for (int iz = 0; iz < side && static_cast<int>(system.x.size()) < n; ++iz) {
    for (int iy = 0; iy < side && static_cast<int>(system.x.size()) < n; ++iy) {
      for (int ix = 0; ix < side && static_cast<int>(system.x.size()) < n; ++ix) {
        if (!Place(&system, (ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing, 0)) {
          throw std::runtime_error("dense grid pair is closer than 0.85 sigma");
        }
      }
    }
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
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> span(0.0, kLength);
  const double faces[2] = {kEps, kLength - kEps};
  for (int axis = 0; axis < 3; ++axis) {
    for (double face : faces) {
      bool placed = false;
      for (int attempt = 0; attempt < 100000 && !placed; ++attempt) {
        double coord[3] = {span(rng), span(rng), span(rng)};
        coord[axis] = face;
        placed = Place(&system, coord[0], coord[1], coord[2], 0);
      }
      if (!placed) throw std::runtime_error("boundary fixture could not place a face atom");
    }
  }
  int attempts = 0;
  while (static_cast<int>(system.x.size()) < kN) {
    if (++attempts > kN * 100000)
      throw std::runtime_error("boundary fixture could not place an atom");
    Place(&system, span(rng), span(rng), span(rng), 0);
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
    int placed = 0;
    int attempts = 0;
    while (placed < kEach) {
      if (++attempts > kEach * 100000) {
        throw std::runtime_error("two-type fixture could not place an atom");
      }
      if (Place(&system, unit(rng) * length, unit(rng) * length, unit(rng) * length, type)) {
        ++placed;
      }
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
    const System boundary = MakeBoundary();
    RequireMinDistance(boundary, "boundary_500");
    WriteFixture(fixture_dir, "boundary_500", boundary);
    WriteFixture(fixture_dir, "cutoff_500", MakeCutoff());
    const System twotype = MakeTwoType();
    RequireMinDistance(twotype, "twotype_2000");
    WriteFixture(fixture_dir, "twotype_2000", twotype);
    const System dilute = MakeRandom(1000, 0.05, &rng);
    RequireMinDistance(dilute, "dilute_1000");
    WriteFixture(fixture_dir, "dilute_1000", dilute);
    // Uniform rejection jams before rho 1.2 at a core of 0.85. A cubic grid
    // still has a nearest neighbor above that core.
    const System dense = MakeDenseGrid(2000, 1.2);
    RequireMinDistance(dense, "dense_2000");
    WriteFixture(fixture_dir, "dense_2000", dense);
    if (argc == 4) WriteMelt(fixture_dir, validation_dir, argv[3]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "tilemd_make_fixtures: " << error.what() << "\n";
    return 1;
  }
}
