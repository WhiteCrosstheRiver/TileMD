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

// One cubic piece of a LAMMPS EAM spline. c[6] is the tabulated value.
struct EamKnot {
  double c[7] = {};
};

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
  void BuildSpline(int n, double delta, const double* values, EamKnot* spline) const;
  void Sample(const EamKnot* spline, int n, double x, double inv_dx, int m_max, double* value,
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
  std::vector<EamKnot> rhor_;
  std::vector<EamKnot> z2r_;
  std::vector<EamKnot> frho_;
  std::vector<double> fp_;
};

// Multi-element DYNAMO setfl table, read the way LAMMPS pair_style eam/alloy
// does. Three comment lines, then one density and embedding table per element.
// r*phi is stored for each pair with i >= j. Atom type i maps to file element i.
// Call SetAtoms before EvaluateAttention. Pass 0 adds the neighbor element's
// density. Pass 1 uses F'(rho_i)*rho'_j + F'(rho_j)*rho'_i + phi'.
class EamSetfl {
 public:
  static constexpr int kNumPasses = 2;

  explicit EamSetfl(const std::string& path);

  double Cutoff() const { return cut_; }
  int NumElements() const { return static_cast<int>(elements_.size()); }
  const std::string& Element(int index) const { return elements_.at(static_cast<size_t>(index)); }
  double Mass(int index) const { return mass_.at(static_cast<size_t>(index)); }
  void SetAtoms(const int* type, int n);
  void Prepare(int n);

  PairOutput PairScalar(int pass, double r2, int type_i, int type_j, int i, int j);
  double AtomUpdate(int pass, int i, double value);

 private:
  int PairIndex(int type_i, int type_j) const;

  double cut_ = 0.0;
  double cut2_ = 0.0;
  double dr_ = 0.0;
  double drho_ = 0.0;
  double rdr_ = 0.0;
  double rdrho_ = 0.0;
  double rhomax_ = 0.0;
  int nr_ = 0;
  int nrho_ = 0;
  std::vector<std::string> elements_;
  std::vector<double> mass_;
  std::vector<std::vector<EamKnot>> rhor_;
  std::vector<std::vector<EamKnot>> frho_;
  std::vector<std::vector<EamKnot>> z2r_;
  std::vector<int> type_;
  std::vector<double> fp_;
};

}  // namespace tilemd
