#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tilemd::ref {

// Double-precision LJ oracle from PLAN.md section 4.1. Deliberately direct:
// one loop over every ordered pair, minimum image with std::rint (the double
// form of the rintf rule in decision D10).

enum class Shift { None, Energy, Force };

struct PairCoeff {
  int type_i = 0;
  int type_j = 0;
  double epsilon = 1.0;
  double sigma = 1.0;
  double cutoff = 2.5;
};

struct System {
  double box[3] = {1.0, 1.0, 1.0};
  Shift shift = Shift::None;
  std::vector<int> type;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<PairCoeff> pairs;
};

struct Observables {
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
  std::vector<double> energy;
  double potential_energy = 0.0;
  // xx, yy, zz, xy, xz, yz. Full-list definition, including the factor 1/2.
  double virial[6] = {};
  // Sum over unique pairs of |r_ij · F_ij|. The virial gate divides by this,
  // not by the cancelled tensor.
  double pair_abs = 0.0;
  // Same components, one row per atom. Their sums are `virial`.
  std::vector<double> wxx;
  std::vector<double> wyy;
  std::vector<double> wzz;
  std::vector<double> wxy;
  std::vector<double> wxz;
  std::vector<double> wyz;
};

int NumAtoms(const System& system);

Observables Compute(const System& system);

void WriteExtxyz(const std::string& path, const System& system);
void WriteObservablesJson(const std::string& path, const std::string& name, const System& system,
                          const Observables& obs);

// One extended-XYZ frame. Velocities are filled when Properties lists `vel`.
struct ExtxyzFrame {
  System system;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  // Empty unless the frame has a `virial:R:6` column, in xx, yy, zz, xy, xz, yz order.
  std::vector<double> wxx;
  std::vector<double> wyy;
  std::vector<double> wzz;
  std::vector<double> wxy;
  std::vector<double> wxz;
  std::vector<double> wyz;
};

// `frame` 0 is the first frame. Properties selects the columns. `pos` and
// `type` are required. A file with no Properties line is species, position, type.
ExtxyzFrame ReadExtxyzFrame(const std::string& path, int frame = 0);

System ReadExtxyz(const std::string& path);
void ReadSystemHeader(const std::string& path, System& system);
Observables ReadObservablesJson(const std::string& path);

// LAMMPS custom dump: id type x y z fx fy fz, atom ids sorted.
// Types are converted from LAMMPS's 1-based ids to 0-based.
struct LammpsDump {
  System system;
  Observables observables;
};
LammpsDump ReadLammpsDump(const std::string& path, Shift shift,
                          const std::vector<PairCoeff>& pairs);

double MinImage(double delta, double length);

// Moving state for the Stage-1 integrators. Positions stay wrapped in [0, L).
// Image counters hold the periods removed by wrapping. Leapfrog stores
// v(t − dt/2); velocity Verlet stores v(t).
struct Dynamics {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  std::vector<int> imgx;
  std::vector<int> imgy;
  std::vector<int> imgz;
};

Dynamics MakeDynamics(const System& system, const std::vector<double>& vx,
                      const std::vector<double>& vy, const std::vector<double>& vz);

void VelocityVerletStep(System& system, Dynamics& state, double dt, double mass);

// Rewrites velocities as v(−dt/2) = v(0) − ½ dt f(0) / m.
void PrepareLeapfrog(System& system, Dynamics& state, double dt, double mass);
void LeapfrogStep(System& system, Dynamics& state, double dt, double mass);

// v(t) = v(t − dt/2) + ½ dt f(t) / m, using the positions stored in `half_step`.
void IntegerTimeVelocity(const System& system, const Dynamics& half_step, double dt, double mass,
                         std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz);

// Random123 Philox4x32-10. Ten rounds, key bumped after each round.
struct PhiloxWord {
  std::uint32_t w[4] = {};
};

PhiloxWord Philox4x32_10(std::uint32_t c0, std::uint32_t c1, std::uint32_t c2, std::uint32_t c3,
                         std::uint32_t k0, std::uint32_t k1);

// Three standard normals from one Philox word. Counter (gid, step low, step high, 0),
// key (seed, 0). The fourth normal is discarded.
void StandardNormals(std::uint32_t seed, std::uint64_t step, int gid, double out[3]);

// BAOAB Langevin on the velocity-Verlet state. gamma == 0 calls VelocityVerletStep.
// The Ornstein–Uhlenbeck draw is three standard normals from one Philox word.
// Counter is (gid, step low, step high, 0) and the key is (seed, 0).
struct Langevin {
  double gamma = 1.0;
  double temperature = 1.0;
  std::uint32_t seed = 1;
};

void LangevinBaoabStep(System& system, Dynamics& state, double dt, double mass,
                       const Langevin& thermo, std::uint64_t step);

// REVIEW_AND_PLAN_v2 F4. Double paths keep the older max-error ratio.
struct T1v2 {
  double force_rms_rel = 0.0;
  double force_max_rel = 0.0;
  double energy_rel = 0.0;
  double virial_rel = 0.0;
  bool pass = false;
};

T1v2 CompareT1v2(const Observables& got, const Observables& reference);

// |slope_tile| <= max(2 |slope_lammps|, 3 se_lammps).
bool PassesT4v2(double slope_tile, double slope_lammps, double se_lammps);

}  // namespace tilemd::ref
