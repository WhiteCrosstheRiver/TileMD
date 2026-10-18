#pragma once

#include <stdexcept>

#include "force/lj_scalar.hpp"
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
  static constexpr int kMaxTypes = 8;

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
    const LjRow& row = table_[type_i][type_j];
    if (!row.defined) throw std::invalid_argument("missing pair coefficient");
    if (!(r2 < row.rc2)) return {};
    PairOutput out;
    LjEvaluate(row, r2, shift_ == AttentionShift::Force ? 1 : 0, &out.force, &out.energy);
    return out;
  }

  // Row-major type_i * kMaxTypes + type_j. The device kernel reads this table.
  void CopyTable(LjRow out[kMaxTypes * kMaxTypes]) const {
    for (int ti = 0; ti < kMaxTypes; ++ti) {
      for (int tj = 0; tj < kMaxTypes; ++tj) out[ti * kMaxTypes + tj] = table_[ti][tj];
    }
  }

  int ForceShift() const { return shift_ == AttentionShift::Force ? 1 : 0; }

  double AtomUpdate(int, int, double) const { return 0.0; }

 private:
  void Store(const AttentionPair& coeff) {
    if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kMaxTypes ||
        coeff.type_j >= kMaxTypes) {
      throw std::invalid_argument("pair type is outside 0..7");
    }
    if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
      throw std::invalid_argument("sigma and cutoff must be positive");
    }
    LjRow& row = table_[coeff.type_i][coeff.type_j];
    row.defined = 1;
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

  AttentionShift shift_ = AttentionShift::None;
  double max_cut_ = 0.0;
  LjRow table_[kMaxTypes][kMaxTypes] = {};
};

// One warp per i-cluster on the host-built cluster list. Pair coordinates are
// the double-precision slot positions, not `pos4`. `skin` only widens the list.
// The host reduction of energy and virial walks atoms in index order.
AttentionResult RunLjAttentionGpu(const double* box, const double* x, const double* y,
                                  const double* z, const int* type, int n,
                                  const AttentionPair* pairs, int npair, AttentionShift shift,
                                  double skin = 0.3);

}  // namespace tilemd
