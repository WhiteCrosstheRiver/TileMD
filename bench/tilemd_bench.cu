// One timed run of an existing TileMD path. PLAN.md section 8.
//   tilemd_bench --path atom|fused-tile|atom-ref --n N --warmup W --steps T
// Optional: --dump path --thermo-every K
// Prints one JSON object. The timer is host wall time around the step loop
// after the warmup. Engine::Run synchronizes before it returns.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lj_reference.hpp"
#include "md/engine.hpp"

namespace {

struct Options {
  std::string path = "atom";
  int n = 32000;
  int warmup = 1000;
  int steps = 100000;
  int thermo_every = 1000;
  std::string dump;
};

Options Parse(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--path")
      opt.path = value;
    else if (key == "--n")
      opt.n = std::stoi(value);
    else if (key == "--warmup")
      opt.warmup = std::stoi(value);
    else if (key == "--steps")
      opt.steps = std::stoi(value);
    else if (key == "--thermo-every")
      opt.thermo_every = std::stoi(value);
    else if (key == "--dump")
      opt.dump = value;
    else
      throw std::runtime_error("unknown argument " + key);
  }
  if (opt.path != "atom" && opt.path != "fused-tile" && opt.path != "atom-ref") {
    throw std::runtime_error("path must be atom, fused-tile, or atom-ref");
  }
  if (opt.n < 1 || opt.warmup < 0 || opt.steps < 1) {
    throw std::runtime_error("n, warmup, and steps must be positive");
  }
  return opt;
}

tilemd::HostSystem MakeFcc(int n) {
  if (n % 4 != 0) throw std::runtime_error("fcc melt n must be a multiple of 4");
  const int cells = static_cast<int>(std::lround(std::cbrt(n / 4.0)));
  if (cells * cells * cells * 4 != n)
    throw std::runtime_error("fcc melt n is not a cubic cell count");
  constexpr double kRho = 0.8442;
  const double length = std::cbrt(static_cast<double>(n) / kRho);
  const double spacing = length / static_cast<double>(cells);
  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  tilemd::HostSystem host;
  host.box[0] = host.box[1] = host.box[2] = length;
  host.shift = tilemd::Shift::None;
  host.skin = 0.3;
  host.dt = 0.005;
  host.mass = 1.0;
  host.rebuild_every_step = false;
  host.thermo_every = 1000;
  host.graph_unroll = 50;
  tilemd::PairSpec spec;
  spec.cutoff = 2.5;
  host.pairs = {spec};
  host.type.assign(n, 0);
  host.x.resize(n);
  host.y.resize(n);
  host.z.resize(n);
  int id = 0;
  for (int ix = 0; ix < cells; ++ix) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int iz = 0; iz < cells; ++iz) {
        for (int b = 0; b < 4; ++b) {
          host.x[id] = (ix + basis[b][0]) * spacing;
          host.y[id] = (iy + basis[b][1]) * spacing;
          host.z[id] = (iz + basis[b][2]) * spacing;
          ++id;
        }
      }
    }
  }
  std::mt19937 rng(20261018);
  std::normal_distribution<double> normal(0.0, 1.0);
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
  return host;
}

tilemd::HostSystem LoadDump(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open dump: " + path);
  tilemd::HostSystem host = MakeFcc(4);
  host.x.clear();
  host.y.clear();
  host.z.clear();
  host.type.clear();
  std::string line;
  int n = -1;
  while (std::getline(in, line)) {
    if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) {
      in >> n;
      std::getline(in, line);
    } else if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) {
      for (int axis = 0; axis < 3; ++axis) {
        double lo = 0.0;
        double hi = 0.0;
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
        double x = 0.0, y = 0.0, z = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;
        in >> id >> type >> x >> y >> z >> vx >> vy >> vz;
        if (!in) throw std::runtime_error("truncated dump");
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

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void RunGpu(const Options& opt, tilemd::HostSystem host) {
  host.thermo_every = opt.thermo_every;
  host.rebuild_every_step = false;
  host.use_cluster_tiles = opt.path == "fused-tile";
  const int n = static_cast<int>(host.x.size());
  tilemd::Engine engine(host);
  if (opt.warmup > 0) engine.Run(opt.warmup);
  const int rebuilds_before = engine.RebuildCount();
  const auto start = std::chrono::steady_clock::now();
  engine.Run(opt.steps);
  const double seconds = SecondsSince(start);
  const double rate = static_cast<double>(n) * static_cast<double>(opt.steps) / seconds;
  std::cout << std::scientific << std::setprecision(8);
  std::cout << "{\"path\":\"" << opt.path << "\",\"n\":" << n << ",\"warmup\":" << opt.warmup
            << ",\"steps\":" << opt.steps << ",\"t_loop_s\":" << seconds
            << ",\"atom_steps_per_s\":" << rate
            << ",\"rebuilds\":" << (engine.RebuildCount() - rebuilds_before)
            << ",\"timer\":\"host wall clock around Engine::Run\"}\n";
}

void RunCpu(const Options& opt, const tilemd::HostSystem& host) {
  tilemd::ref::System system;
  system.box[0] = host.box[0];
  system.box[1] = host.box[1];
  system.box[2] = host.box[2];
  system.shift = tilemd::ref::Shift::None;
  system.type = host.type;
  system.x = host.x;
  system.y = host.y;
  system.z = host.z;
  tilemd::ref::PairCoeff coeff;
  coeff.cutoff = 2.5;
  system.pairs = {coeff};
  tilemd::ref::Dynamics state = tilemd::ref::MakeDynamics(system, host.vx, host.vy, host.vz);
  const int n = static_cast<int>(host.x.size());
  for (int step = 0; step < opt.warmup; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, host.dt, host.mass);
  }
  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < opt.steps; ++step) {
    tilemd::ref::VelocityVerletStep(system, state, host.dt, host.mass);
  }
  const double seconds = SecondsSince(start);
  const double rate = static_cast<double>(n) * static_cast<double>(opt.steps) / seconds;
  std::cout << std::scientific << std::setprecision(8);
  std::cout << "{\"path\":\"atom-ref\",\"n\":" << n << ",\"warmup\":" << opt.warmup
            << ",\"steps\":" << opt.steps << ",\"t_loop_s\":" << seconds
            << ",\"atom_steps_per_s\":" << rate
            << ",\"rebuilds\":0,\"timer\":\"host wall clock around VelocityVerletStep\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = Parse(argc, argv);
    tilemd::HostSystem host = opt.dump.empty() ? MakeFcc(opt.n) : LoadDump(opt.dump);
    if (opt.path == "atom-ref")
      RunCpu(opt, host);
    else
      RunGpu(opt, host);
  } catch (const std::exception& error) {
    std::cerr << "tilemd_bench: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
