#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tilemd {

// Host-side description of one LJ system. The engine copies it; afterwards the
// device positions and velocities are the state that integration updates.
enum class Shift { None, Energy, Force };

struct PairSpec {
  int type_i = 0;
  int type_j = 0;
  double epsilon = 1.0;
  double sigma = 1.0;
  double cutoff = 2.5;
};

struct HostSystem {
  double box[3] = {1.0, 1.0, 1.0};
  Shift shift = Shift::None;
  double skin = 0.0;
  double mass = 1.0;
  double dt = 0.005;
  // 0 selects the density estimate from PLAN.md section 5.3.
  int max_neigh = 0;
  bool rebuild_every_step = true;
  // Atom-neighbor `Run` captures this many steps in one CUDA graph. The fused
  // tile path does the same only when it is not rebuilding every step, and
  // then rebuilds the cluster list on the device between steps. Otherwise it
  // launches one step at a time.
  int graph_unroll = 50;
  // 0 keeps energy and virial off the step path. Otherwise the device writes a
  // thermo record when the completed step is a multiple of this interval.
  int thermo_every = 0;
  // 0 does not write. Otherwise `Run` ends a segment at the next multiple and
  // appends an extended-XYZ frame in original-gid order to `dump_path`.
  int dump_every = 0;
  std::string dump_path;
  // 0 omits the column. 1 recomputes the six per-atom components at the dumped
  // positions and writes them as virial:R:6 (xx, yy, zz, xy, xz, yz).
  int dump_virial = 0;
  // Cluster tiles replace the atom neighbor list for forces and integration.
  // The default remains the atom-neighbor engine.
  bool use_cluster_tiles = false;
  // BAOAB on the atom-neighbor path. Zero friction stays on velocity Verlet.
  // The fused leapfrog does not apply this.
  bool langevin = false;
  double gamma = 0.0;
  double temperature = 1.0;
  std::uint32_t langevin_seed = 1;
  std::vector<int> type;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  std::vector<PairSpec> pairs;
};

struct ThermoSample {
  int step = 0;
  double kinetic_energy = 0.0;
  double potential_energy = 0.0;
  // xx, yy, zz, xy, xz, yz. Same full-list definition as the CPU oracle.
  double virial[6] = {};
};

struct GpuForces {
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  double potential_energy = 0.0;
  // xx, yy, zz, xy, xz, yz. Same full-list definition as the CPU oracle.
  double virial[6] = {};
  int overflow = 0;
  int max_neigh = 0;
};

// Thread-per-atom neighbor list and LJ force. Rebuilds reorder atoms by a
// stable Morton cell code. Run captures 50 steps in a CUDA graph and copies
// status back at segment boundaries of 1000 steps. Energy and virial are
// computed only on thermo steps and reduced into a device ring. Downloads
// scatter to the original global-id order.
class Engine {
 public:
  explicit Engine(HostSystem system);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Rebuild cells and the neighbor list, then compute forces.
  // grow=false leaves an overflow in place so a test can see it.
  GpuForces ComputeForces(bool grow = true);
  // Same neighbor list and pair loop as ComputeForces. block_size is 64, 128,
  // or 256. The integrator still launches 128.
  GpuForces ForcesAtBlock(int block_size);
  // One neighbor build, one warmup launch, then `repeats` force launches.
  // Returns seconds. block_size is 64, 128, or 256.
  double TimeForceKernel(int block_size, int repeats);
  void Run(int steps);

  void DownloadPositions(std::vector<double>& x, std::vector<double>& y,
                         std::vector<double>& z) const;
  void DownloadVelocities(std::vector<double>& vx, std::vector<double>& vy,
                          std::vector<double>& vz) const;

  int NumAtoms() const;
  int MaxNeigh() const;
  int RebuildCount() const;
  // Copies thermo records gathered at segment boundaries and clears them.
  std::vector<ThermoSample> DrainThermo();

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace tilemd
