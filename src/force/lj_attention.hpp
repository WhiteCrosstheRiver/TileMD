#pragma once

#include <stdexcept>

#include "force/pair_attention.hpp"

namespace tilemd {

enum class AttentionShift { None, Energy, Force };

struct AttentionPair {
  int type_i = 0;
  int type_j = 0;
  double epsilon = 1.0;
  double sigma = 1.0;
  double cutoff = 2.5;
};

// One-pass Lennard-Jones. The scalar is s(r^2) and the aggregated values are
// s * r_j and s * 1, so the reconstructed force is s (r_i - r_j).
class LjAttention {
 public:
  static constexpr int kNumPasses = 1;

  LjAttention(const AttentionPair* pairs, int npair, AttentionShift shift) : shift_(shift) {
    if (npair < 1) throw std::invalid_argument("at least one pair coefficient is required");
    for (int p = 0; p < npair; ++p) {
      Store(pairs[p]);
      if (pairs[p].type_i != pairs[p].type_j) {
        AttentionPair swapped = pairs[p];
        swapped.type_i = pairs[p].type_j;
        swapped.type_j = pairs[p].type_i;
        Store(swapped);
      }
    }
  }

  double Cutoff() const { return max_cut_; }
  void Prepare(int) {}

  PairOutput PairScalar(int, double r2, int type_i, int type_j, int, int) const {
    const Row& row = table_[type_i][type_j];
    if (!row.defined) throw std::invalid_argument("missing pair coefficient");
    if (!(r2 < row.rc2)) return {};
    const double inv_r2 = 1.0 / r2;
    const double sr2 = row.sigma2 * inv_r2;
    const double sr6 = sr2 * sr2 * sr2;
    double s = row.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
    double u = row.eps4 * sr6 * (sr6 - 1.0) - row.u_shift;
    if (shift_ == AttentionShift::Force) {
      const double r = std::sqrt(r2);
      u += (r - row.rc) * row.s_rc * row.rc;
      s -= row.s_rc * row.rc / r;
    }
    PairOutput out;
    out.force = s;
    out.energy = u;
    return out;
  }

  double AtomUpdate(int, int, double) const { return 0.0; }

 private:
  struct Row {
    bool defined = false;
    double sigma2 = 0.0;
    double eps4 = 0.0;
    double eps24 = 0.0;
    double rc = 0.0;
    double rc2 = 0.0;
    double u_shift = 0.0;
    double s_rc = 0.0;
  };

  void Store(const AttentionPair& coeff) {
    if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kMaxTypes ||
        coeff.type_j >= kMaxTypes) {
      throw std::invalid_argument("pair type is outside 0..7");
    }
    if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
      throw std::invalid_argument("sigma and cutoff must be positive");
    }
    Row& row = table_[coeff.type_i][coeff.type_j];
    row.defined = true;
    row.sigma2 = coeff.sigma * coeff.sigma;
    row.eps4 = 4.0 * coeff.epsilon;
    row.eps24 = 24.0 * coeff.epsilon;
    row.rc = coeff.cutoff;
    row.rc2 = coeff.cutoff * coeff.cutoff;
    const double inv_rc2 = 1.0 / row.rc2;
    const double sr2 = row.sigma2 * inv_rc2;
    const double sr6 = sr2 * sr2 * sr2;
    const double u_c = row.eps4 * sr6 * (sr6 - 1.0);
    row.s_rc = row.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
    row.u_shift = (shift_ == AttentionShift::None) ? 0.0 : u_c;
    if (coeff.cutoff > max_cut_) max_cut_ = coeff.cutoff;
  }

  static constexpr int kMaxTypes = 8;
  AttentionShift shift_ = AttentionShift::None;
  double max_cut_ = 0.0;
  Row table_[kMaxTypes][kMaxTypes] = {};
};

}  // namespace tilemd
