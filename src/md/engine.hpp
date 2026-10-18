#pragma once

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
  std::vector<int> type;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  std::vector<PairSpec> pairs;
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
// status back at segment boundaries of 1000 steps. Downloads scatter to the
// original global-id order.
class Engine {
 public:
  explicit Engine(HostSystem system);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Rebuild cells and the neighbor list, then compute forces.
  // grow=false leaves an overflow in place so a test can see it.
  GpuForces ComputeForces(bool grow = true);
  void Run(int steps);

  void DownloadPositions(std::vector<double>& x, std::vector<double>& y,
                         std::vector<double>& z) const;
  void DownloadVelocities(std::vector<double>& vx, std::vector<double>& vy,
                          std::vector<double>& vz) const;

  int NumAtoms() const;
  int MaxNeigh() const;
  int RebuildCount() const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace tilemd
