// One-shot T1-v2 check for the FP32 atom-list force. Not part of ctest.
// Usage: tilemd_fp32_check [threads_per_atom]

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "force/fp32_walk.hpp"
#include "lj_reference.hpp"
#include "md/fused_atom.hpp"

namespace {

double MinImage(double delta, double length) { return delta - length * std::rint(delta / length); }

void MakeFcc(int cells, double rho, std::vector<double>* x, std::vector<double>* y,
             std::vector<double>* z, double* box) {
  const double a = std::cbrt(4.0 / rho);
  *box = cells * a;
  const double basis[4][3] = {
      {0, 0, 0}, {0, 0.5 * a, 0.5 * a}, {0.5 * a, 0, 0.5 * a}, {0.5 * a, 0.5 * a, 0}};
  for (int iz = 0; iz < cells; ++iz) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int ix = 0; ix < cells; ++ix) {
        for (const auto& b : basis) {
          x->push_back(ix * a + b[0]);
          y->push_back(iy * a + b[1]);
          z->push_back(iz * a + b[2]);
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int threads = (argc > 1) ? std::stoi(argv[1]) : 4;
    const double cutoff = 2.5;
    const int cells = 6;
    std::vector<double> x, y, z;
    double length = 0.0;
    MakeFcc(cells, 0.8442, &x, &y, &z, &length);
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) {
      x[i] += 0.07 * std::sin(0.7 * i);
      y[i] += 0.05 * std::cos(0.3 * i);
      z[i] += 0.04 * std::sin(0.5 * i + 1.0);
      x[i] -= length * std::floor(x[i] / length);
      y[i] -= length * std::floor(y[i] / length);
      z[i] -= length * std::floor(z[i] / length);
    }
    const int max_neigh = 128;
    std::vector<int> nn(n, 0);
    std::vector<int> nl(static_cast<std::size_t>(max_neigh) * n, -1);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const double dx = MinImage(x[i] - x[j], length);
        const double dy = MinImage(y[i] - y[j], length);
        const double dz = MinImage(z[i] - z[j], length);
        if (dx * dx + dy * dy + dz * dz >= cutoff * cutoff) continue;
        if (nn[i] >= max_neigh) throw std::runtime_error("neighbor list overflow");
        nl[static_cast<std::size_t>(nn[i]) * n + i] = j;
        ++nn[i];
      }
    }

    tilemd::Fp32Lj pair;
    pair.sigma2 = 1.f;
    pair.eps4 = 4.f;
    pair.eps24 = 24.f;
    pair.rc = 2.5f;
    pair.rc2 = 6.25f;
    const double box[3] = {length, length, length};
    const tilemd::Fp32Force got = tilemd::FusedAtomForce(
        box, x.data(), y.data(), z.data(), nn.data(), nl.data(), n, max_neigh, pair, threads);

    tilemd::ref::System system;
    system.box[0] = system.box[1] = system.box[2] = length;
    system.shift = tilemd::ref::Shift::None;
    system.type.assign(n, 0);
    system.x = x;
    system.y = y;
    system.z = z;
    tilemd::ref::PairCoeff coeff;
    coeff.cutoff = cutoff;
    system.pairs = {coeff};
    const tilemd::ref::Observables ref = tilemd::ref::Compute(system);

    tilemd::ref::Observables fp32;
    fp32.fx.assign(got.fx.begin(), got.fx.end());
    fp32.fy.assign(got.fy.begin(), got.fy.end());
    fp32.fz.assign(got.fz.begin(), got.fz.end());
    fp32.potential_energy = got.potential_energy;
    for (int c = 0; c < 6; ++c) fp32.virial[c] = got.virial[c];
    const tilemd::ref::T1v2 gate = tilemd::ref::CompareT1v2(fp32, ref);
    std::vector<double> host_x = x, host_y = y, host_z = z;
    std::vector<double> host_vx(n, 0.0), host_vy(n, 0.0), host_vz(n, 0.0);
    std::vector<int> host_ix(n, 0), host_iy(n, 0), host_iz(n, 0);
    std::vector<double> vx = host_vx, vy = host_vy, vz = host_vz;
    std::vector<int> image_x = host_ix, image_y = host_iy, image_z = host_iz;
    const std::vector<double> x_ref = x, y_ref = y, z_ref = z;
    const int nve_steps = 5;
    const double dt = 0.005;
    for (int step = 0; step < nve_steps; ++step) {
      const tilemd::Fp32Walk force = tilemd::HostFp32Force(
          box, host_x.data(), host_y.data(), host_z.data(), nn.data(), nl.data(), n, pair);
      for (int i = 0; i < n; ++i) {
        tilemd::KickDriftAxis(&host_x[i], &host_vx[i], force.fx[i], dt, 1.0, length, &host_ix[i]);
        tilemd::KickDriftAxis(&host_y[i], &host_vy[i], force.fy[i], dt, 1.0, length, &host_iy[i]);
        tilemd::KickDriftAxis(&host_z[i], &host_vz[i], force.fz[i], dt, 1.0, length, &host_iz[i]);
      }
    }
    tilemd::Fp32NveStats stats;
    tilemd::FusedAtomNve(box, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                         image_x.data(), image_y.data(), image_z.data(), x_ref.data(), y_ref.data(),
                         z_ref.data(), nn.data(), nl.data(), n, max_neigh, pair, dt, 1.0, nve_steps,
                         1, &stats);
    double max_dx = 0.0;
    for (int i = 0; i < n; ++i) {
      max_dx = std::max(max_dx, std::abs(x[i] - host_x[i]));
      max_dx = std::max(max_dx, std::abs(y[i] - host_y[i]));
      max_dx = std::max(max_dx, std::abs(z[i] - host_z[i]));
    }
    const bool nve_ok = max_dx <= 1.0e-4;
    std::cout << "n=" << n << " tpa=" << threads << " force_rms_rel=" << gate.force_rms_rel
              << " force_max_rel=" << gate.force_max_rel << " energy_rel=" << gate.energy_rel
              << " virial_rel=" << gate.virial_rel << " pass=" << (gate.pass ? 1 : 0)
              << " nve_steps=" << nve_steps << " max_dx=" << max_dx
              << " max_disp2=" << stats.max_disp2 << " nve_ok=" << (nve_ok ? 1 : 0) << "\n";
    return (gate.pass && nve_ok) ? 0 : 1;
  } catch (const std::exception& err) {
    std::cerr << err.what() << "\n";
    return 2;
  }
}
