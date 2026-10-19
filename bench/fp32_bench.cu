// Times the FP32 fused kick-drift. Not part of ctest and not the running baseline.
// Usage: tilemd_fp32_bench --n 864 --warmup 2 --steps 20 --tpa 1

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/lammps_dump.hpp"
#include "md/fused_atom.hpp"
#include "neighbor/cell_list.hpp"

namespace {

int Arg(int argc, char** argv, const std::string& name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) return std::stoi(argv[i + 1]);
  }
  return fallback;
}

std::string ArgString(int argc, char** argv, const std::string& name) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) return argv[i + 1];
  }
  return "";
}

void MakeFcc(int cells, double rho, std::vector<double>* x, std::vector<double>* y,
             std::vector<double>* z, double* box) {
  const double a = std::cbrt(4.0 / rho);
  *box = cells * a;
  const double basis[4][3] = {
      {0, 0, 0}, {0, 0.5 * a, 0.5 * a}, {0.5 * a, 0, 0.5 * a}, {0.5 * a, 0.5 * a, 0}};
  for (int iz = 0; iz < cells; ++iz) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int ix = 0; ix < cells; ++ix) {
        for (const auto& site : basis) {
          x->push_back(ix * a + site[0]);
          y->push_back(iy * a + site[1]);
          z->push_back(iz * a + site[2]);
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int warmup = Arg(argc, argv, "--warmup", 2);
    const int steps = Arg(argc, argv, "--steps", 20);
    const int tpa = Arg(argc, argv, "--tpa", 1);
    const std::string dump = ArgString(argc, argv, "--dump");
    std::vector<double> x, y, z, vx, vy, vz;
    double box[3] = {};
    if (!dump.empty()) {
      const tilemd::DumpFrame frame = tilemd::ReadLammpsDump(dump);
      x = frame.x;
      y = frame.y;
      z = frame.z;
      vx = frame.vx;
      vy = frame.vy;
      vz = frame.vz;
      box[0] = frame.box[0];
      box[1] = frame.box[1];
      box[2] = frame.box[2];
    } else {
      const int n_request = Arg(argc, argv, "--n", 864);
      int cells = 1;
      while (4 * cells * cells * cells < n_request) ++cells;
      if (4 * cells * cells * cells != n_request) {
        throw std::runtime_error("n must be 4*cells^3");
      }
      double length = 0;
      MakeFcc(cells, 0.8442, &x, &y, &z, &length);
      box[0] = box[1] = box[2] = length;
      const int count = static_cast<int>(x.size());
      vx.assign(count, 0.05);
      vy.assign(count, -0.02);
      vz.assign(count, 0.01);
    }
    const int n = static_cast<int>(x.size());
    const std::vector<int> order = tilemd::MortonOrder(box, x.data(), y.data(), z.data(), n, 2.8);
    tilemd::ApplyOrder(&x, order);
    tilemd::ApplyOrder(&y, order);
    tilemd::ApplyOrder(&z, order);
    tilemd::ApplyOrder(&vx, order);
    tilemd::ApplyOrder(&vy, order);
    tilemd::ApplyOrder(&vz, order);
    const tilemd::CellList list = tilemd::BuildCellList(box, x.data(), y.data(), z.data(), n, 2.8);
    std::vector<int> ix(n, 0), iy(n, 0), iz(n, 0);
    const std::vector<double> x_ref = x, y_ref = y, z_ref = z;
    tilemd::Fp32Lj pair;
    auto run = [&](int count) {
      tilemd::Fp32NveStats stats;
      tilemd::FusedAtomNve(box, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                           ix.data(), iy.data(), iz.data(), x_ref.data(), y_ref.data(),
                           z_ref.data(), list.nn.data(), list.nl.data(), n, list.max_neigh, pair,
                           0.005, 1.0, count, tpa, &stats);
      return stats;
    };
    auto run_chunked = [&](int count) {
      tilemd::Fp32NveStats last;
      int left = count;
      while (left > 0) {
        const int chunk = left < 200 ? left : 200;
        last = run(chunk);
        left -= chunk;
      }
      return last;
    };
    if (warmup > 0) run_chunked(warmup);
    const auto t0 = std::chrono::steady_clock::now();
    const tilemd::Fp32NveStats stats = run_chunked(steps);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double rate = static_cast<double>(n) * static_cast<double>(steps) / seconds;
    std::cout.setf(std::ios::scientific);
    std::cout.precision(8);
    std::cout << "{\"path\":\"fused-atom\",\"n\":" << n << ",\"steps\":" << steps
              << ",\"warmup\":" << warmup << ",\"tpa\":" << tpa << ",\"t_loop_s\":" << seconds
              << ",\"atom_steps_per_s\":" << rate << ",\"max_disp2\":" << stats.max_disp2
              << ",\"max_neigh\":" << list.max_neigh
              << ",\"note\":\"initial rc+skin cell list, not rebuilt, sync every 200 steps\"}\n";
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << "\n";
    return 2;
  }
}
