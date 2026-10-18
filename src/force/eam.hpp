#pragma once

#include <string>
#include <vector>

#include "force/pair_attention.hpp"

namespace tilemd {

// LAMMPS metal units convert eV/Å and g/mol into Å/ps. Velocity Verlet uses
// v += dt * f / m * kMetalFtm2v. Reduced LJ units omit this factor.
// kMetalBoltz is eV/K. kMetalNktv2p converts eV/Å^3 into bar.
inline constexpr double kMetalMvv2e = 1.0364269e-4;
inline constexpr double kMetalFtm2v = 1.0 / kMetalMvv2e;
inline constexpr double kMetalBoltz = 8.617343e-5;
inline constexpr double kMetalNktv2p = 1.602176637e6;

// Single-element funcfl table, read and splined the way LAMMPS pair_style eam
// does. Metal units: distance in Angstrom, energy in eV. Pass 0 sums the
// electron density. Pass 1 applies (F'(rho_i) + F'(rho_j)) rho'(r) + phi'(r).
class EamFuncfl {
 public:
  static constexpr int kNumPasses = 2;

  explicit EamFuncfl(const std::string& path);

  double Cutoff() const { return cut_; }
  double Mass() const { return mass_; }
  void Prepare(int n);

  PairOutput PairScalar(int pass, double r2, int type_i, int type_j, int i, int j);
  double AtomUpdate(int pass, int i, double value);

 private:
  struct Spline {
    double c[7] = {};
  };

  void BuildSpline(int n, double delta, const double* values, Spline* spline) const;
  void Sample(const Spline* spline, int n, double x, double inv_dx, int m_max, double* value,
              double* derivative) const;

  double mass_ = 0.0;
  double cut_ = 0.0;
  double cut2_ = 0.0;
  double dr_ = 0.0;
  double drho_ = 0.0;
  double rdr_ = 0.0;
  double rdrho_ = 0.0;
  double rhomax_ = 0.0;
  int nr_ = 0;
  int nrho_ = 0;
  std::vector<Spline> rhor_;
  std::vector<Spline> z2r_;
  std::vector<Spline> frho_;
  std::vector<double> fp_;
};

}  // namespace tilemd
