// Long validation runs that do not belong in the GoogleTest suite.
//   tilemd_long_run drift <dump> <steps> <sample_every> <out.csv>
//   tilemd_long_run rdf   <dump> <steps> <sample_every> <rmax> <nbins> <out.csv>
//
// drift uses lj_shift = force (PLAN.md T4). rdf uses lj_shift = none (T5).

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "md/engine.hpp"

namespace {

tilemd::HostSystem LoadDump(const std::string& path, tilemd::Shift shift) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open dump: " + path);
  tilemd::HostSystem host;
  host.shift = shift;
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
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;
        in >> id >> type >> x >> y >> z >> vx >> vy >> vz;
        if (!in) throw std::runtime_error("truncated dump");
        if (id < 1 || id > n) throw std::runtime_error("atom id outside 1..N");
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

double Kinetic(const std::vector<double>& vx, const std::vector<double>& vy,
               const std::vector<double>& vz) {
  double kinetic = 0.0;
  for (size_t i = 0; i < vx.size(); ++i) {
    kinetic += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
  }
  return kinetic;
}

// Ordinary least squares slope of y versus t. Units follow the caller.
double Slope(const std::vector<double>& t, const std::vector<double>& y) {
  const double n = static_cast<double>(t.size());
  double st = 0.0;
  double sy = 0.0;
  double stt = 0.0;
  double sty = 0.0;
  for (size_t i = 0; i < t.size(); ++i) {
    st += t[i];
    sy += y[i];
    stt += t[i] * t[i];
    sty += t[i] * y[i];
  }
  const double denom = n * stt - st * st;
  if (denom == 0.0) throw std::runtime_error("slope is undefined");
  return (n * sty - st * sy) / denom;
}

void SampleEnergy(tilemd::Engine& engine, int step, double dt, std::ofstream& out,
                  std::vector<double>& times, std::vector<double>& energies) {
  const tilemd::GpuForces forces = engine.ComputeForces();
  if (forces.overflow != 0) throw std::runtime_error("neighbor list overflowed");
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  engine.DownloadVelocities(vx, vy, vz);
  const double n = static_cast<double>(vx.size());
  const double ke = Kinetic(vx, vy, vz);
  const double pe = forces.potential_energy;
  const double e = (ke + pe) / n;
  const double time = step * dt;
  out << step << " " << time << " " << e << " " << pe / n << " " << ke / n << "\n";
  out.flush();
  times.push_back(time);
  energies.push_back(e);
  std::cout << "step " << step << " e/N " << e << "\n" << std::flush;
}

void RunDrift(const std::string& dump, int steps, int sample_every, const std::string& out_path) {
  if (sample_every < 1 || steps < 0) throw std::runtime_error("bad drift arguments");
  tilemd::Engine engine(LoadDump(dump, tilemd::Shift::Force));
  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("cannot write " + out_path);
  out << "# step time e_per_atom pe_per_atom ke_per_atom\n";
  std::vector<double> times;
  std::vector<double> energies;
  SampleEnergy(engine, 0, 0.005, out, times, energies);
  for (int done = 0; done < steps;) {
    const int chunk = std::min(sample_every, steps - done);
    engine.Run(chunk);
    done += chunk;
    SampleEnergy(engine, done, 0.005, out, times, energies);
  }
  const double slope = Slope(times, energies);
  std::cout << "drift_slope_per_atom_per_tau " << slope << "\n";
}

struct CellGrid {
  int ncell[3] = {};
  double box[3] = {};
  double cell[3] = {};
  std::vector<int> head;
  std::vector<int> next;
};

CellGrid BuildCells(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& z, const double* box, double rmax) {
  CellGrid grid;
  const int n = static_cast<int>(x.size());
  for (int axis = 0; axis < 3; ++axis) {
    grid.box[axis] = box[axis];
    grid.ncell[axis] = std::max(1, static_cast<int>(std::floor(box[axis] / rmax)));
    grid.cell[axis] = box[axis] / static_cast<double>(grid.ncell[axis]);
  }
  const int ntot = grid.ncell[0] * grid.ncell[1] * grid.ncell[2];
  grid.head.assign(ntot, -1);
  grid.next.assign(n, -1);
  auto wrap_cell = [](int value, int count) {
    int wrapped = value % count;
    if (wrapped < 0) wrapped += count;
    return wrapped;
  };
  for (int i = 0; i < n; ++i) {
    const int cx = wrap_cell(static_cast<int>(std::floor(x[i] / grid.cell[0])), grid.ncell[0]);
    const int cy = wrap_cell(static_cast<int>(std::floor(y[i] / grid.cell[1])), grid.ncell[1]);
    const int cz = wrap_cell(static_cast<int>(std::floor(z[i] / grid.cell[2])), grid.ncell[2]);
    const int cell = (cx * grid.ncell[1] + cy) * grid.ncell[2] + cz;
    grid.next[i] = grid.head[cell];
    grid.head[cell] = i;
  }
  return grid;
}

void RunRdf(const std::string& dump, int steps, int sample_every, double rmax, int nbins,
            const std::string& out_path) {
  if (sample_every < 1 || steps < 0 || !(rmax > 0.0) || nbins < 1) {
    throw std::runtime_error("bad rdf arguments");
  }
  tilemd::HostSystem host = LoadDump(dump, tilemd::Shift::None);
  tilemd::Engine engine(host);
  const int n = static_cast<int>(host.x.size());
  const double dr = rmax / static_cast<double>(nbins);
  const double volume = host.box[0] * host.box[1] * host.box[2];
  const double rho = static_cast<double>(n) / volume;
  std::vector<double> count(nbins, 0.0);
  int frames = 0;

  auto accumulate = [&](int step) {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    engine.DownloadPositions(x, y, z);
    const CellGrid grid = BuildCells(x, y, z, host.box, rmax);
    ++frames;
    for (int cx = 0; cx < grid.ncell[0]; ++cx) {
      for (int cy = 0; cy < grid.ncell[1]; ++cy) {
        for (int cz = 0; cz < grid.ncell[2]; ++cz) {
          const int cell = (cx * grid.ncell[1] + cy) * grid.ncell[2] + cz;
          for (int ox = -1; ox <= 1; ++ox) {
            for (int oy = -1; oy <= 1; ++oy) {
              for (int oz = -1; oz <= 1; ++oz) {
                int nx = (cx + ox) % grid.ncell[0];
                int ny = (cy + oy) % grid.ncell[1];
                int nz = (cz + oz) % grid.ncell[2];
                if (nx < 0) nx += grid.ncell[0];
                if (ny < 0) ny += grid.ncell[1];
                if (nz < 0) nz += grid.ncell[2];
                const int other = (nx * grid.ncell[1] + ny) * grid.ncell[2] + nz;
                for (int i = grid.head[cell]; i >= 0; i = grid.next[i]) {
                  for (int j = grid.head[other]; j >= 0; j = grid.next[j]) {
                    if (i == j) continue;
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
                    dx -= host.box[0] * std::rint(dx / host.box[0]);
                    dy -= host.box[1] * std::rint(dy / host.box[1]);
                    dz -= host.box[2] * std::rint(dz / host.box[2]);
                    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (!(r < rmax)) continue;
                    int bin = static_cast<int>(r / dr);
                    if (bin >= nbins) bin = nbins - 1;
                    count[bin] += 1.0;
                  }
                }
              }
            }
          }
        }
      }
    }
    std::cout << "rdf frame " << frames << " step " << step << "\n" << std::flush;
  };

  // Match LAMMPS fix ave/time Nevery=sample_every: frames at sample_every, 2*sample_every, ...
  // Step 0 is not included.
  for (int integrated = 0; integrated < steps;) {
    const int chunk = std::min(sample_every, steps - integrated);
    engine.Run(chunk);
    integrated += chunk;
    accumulate(integrated);
  }

  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("cannot write " + out_path);
  out << "# r_mid g coord\n";
  const double factor = 4.0 / 3.0 * 3.14159265358979323846 * rho * static_cast<double>(frames) *
                        static_cast<double>(n);
  double coord = 0.0;
  for (int bin = 0; bin < nbins; ++bin) {
    const double lo = bin * dr;
    const double hi = lo + dr;
    const double shell = hi * hi * hi - lo * lo * lo;
    const double g = (shell > 0.0) ? count[bin] / (factor * shell) : 0.0;
    coord += count[bin] / (static_cast<double>(frames) * static_cast<double>(n));
    out << (0.5 * (lo + hi)) << " " << g << " " << coord << "\n";
  }
  std::cout << "rdf_frames " << frames << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: tilemd_long_run drift|rdf ...\n";
      return 2;
    }
    const std::string mode = argv[1];
    if (mode == "drift") {
      if (argc != 6) {
        std::cerr << "usage: tilemd_long_run drift <dump> <steps> <sample_every> <out.csv>\n";
        return 2;
      }
      RunDrift(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), argv[5]);
      return 0;
    }
    if (mode == "rdf") {
      if (argc != 8) {
        std::cerr << "usage: tilemd_long_run rdf <dump> <steps> <sample_every> <rmax> <nbins> "
                     "<out.csv>\n";
        return 2;
      }
      RunRdf(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stod(argv[5]),
             std::stoi(argv[6]), argv[7]);
      return 0;
    }
    std::cerr << "unknown mode " << mode << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
