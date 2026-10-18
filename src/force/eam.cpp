#include "force/eam.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tilemd {
namespace {

constexpr double kHartreeToEv = 27.2;
constexpr double kBohrToAngstrom = 0.529;

std::vector<double> ReadDoubles(std::istream& in, int count) {
  std::vector<double> values(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    in >> values[static_cast<size_t>(i)];
    if (!in) throw std::runtime_error("EAM table ended early");
  }
  return values;
}

}  // namespace

EamFuncfl::EamFuncfl(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open EAM file " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("EAM file is missing its title");
  if (!std::getline(in, line)) throw std::runtime_error("EAM file is missing its mass line");
  {
    std::istringstream header(line);
    int atomic_number = 0;
    header >> atomic_number >> mass_;
    if (!header || !(mass_ > 0.0)) throw std::runtime_error("EAM mass line is invalid");
  }
  int nrho = 0;
  int nr = 0;
  double drho = 0.0;
  double dr = 0.0;
  double cut = 0.0;
  in >> nrho >> drho >> nr >> dr >> cut;
  if (!in || nrho < 5 || nr < 5 || !(drho > 0.0) || !(dr > 0.0) || !(cut > 0.0)) {
    throw std::runtime_error("EAM grid header is invalid");
  }
  const std::vector<double> frho = ReadDoubles(in, nrho);
  const std::vector<double> zr = ReadDoubles(in, nr);
  const std::vector<double> rhor = ReadDoubles(in, nr);

  nrho_ = nrho;
  nr_ = nr;
  drho_ = drho;
  dr_ = dr;
  rdrho_ = 1.0 / drho;
  rdr_ = 1.0 / dr;
  cut_ = cut;
  cut2_ = cut * cut;
  rhomax_ = (nrho - 1) * drho;

  std::vector<double> z2r(static_cast<size_t>(nr) + 1);
  std::vector<double> density(static_cast<size_t>(nr) + 1);
  std::vector<double> embedding(static_cast<size_t>(nrho) + 1);
  for (int m = 1; m <= nr; ++m) {
    const double z = zr[static_cast<size_t>(m - 1)];
    z2r[static_cast<size_t>(m)] = kHartreeToEv * kBohrToAngstrom * z * z;
    density[static_cast<size_t>(m)] = rhor[static_cast<size_t>(m - 1)];
  }
  for (int m = 1; m <= nrho; ++m)
    embedding[static_cast<size_t>(m)] = frho[static_cast<size_t>(m - 1)];

  // A single funcfl file is already on the grid LAMMPS would reinterpolate
  // onto, so the tabulated values are copied and then splined.
  rhor_.assign(static_cast<size_t>(nr) + 1, {});
  z2r_.assign(static_cast<size_t>(nr) + 1, {});
  frho_.assign(static_cast<size_t>(nrho) + 1, {});
  BuildSpline(nr, dr, density.data(), rhor_.data());
  BuildSpline(nr, dr, z2r.data(), z2r_.data());
  BuildSpline(nrho, drho, embedding.data(), frho_.data());
}

void EamFuncfl::Prepare(int n) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  fp_.assign(static_cast<size_t>(n), 0.0);
}

void EamFuncfl::BuildSpline(int n, double delta, const double* values, Spline* spline) const {
  for (int m = 1; m <= n; ++m) spline[m].c[6] = values[m];
  spline[1].c[5] = spline[2].c[6] - spline[1].c[6];
  spline[2].c[5] = 0.5 * (spline[3].c[6] - spline[1].c[6]);
  spline[n - 1].c[5] = 0.5 * (spline[n].c[6] - spline[n - 2].c[6]);
  spline[n].c[5] = spline[n].c[6] - spline[n - 1].c[6];
  for (int m = 3; m <= n - 2; ++m) {
    spline[m].c[5] = ((spline[m - 2].c[6] - spline[m + 2].c[6]) +
                      8.0 * (spline[m + 1].c[6] - spline[m - 1].c[6])) /
                     12.0;
  }
  for (int m = 1; m <= n - 1; ++m) {
    spline[m].c[4] =
        3.0 * (spline[m + 1].c[6] - spline[m].c[6]) - 2.0 * spline[m].c[5] - spline[m + 1].c[5];
    spline[m].c[3] =
        spline[m].c[5] + spline[m + 1].c[5] - 2.0 * (spline[m + 1].c[6] - spline[m].c[6]);
  }
  spline[n].c[4] = 0.0;
  spline[n].c[3] = 0.0;
  for (int m = 1; m <= n; ++m) {
    spline[m].c[2] = spline[m].c[5] / delta;
    spline[m].c[1] = 2.0 * spline[m].c[4] / delta;
    spline[m].c[0] = 3.0 * spline[m].c[3] / delta;
  }
}

void EamFuncfl::Sample(const Spline* spline, int n, double x, double inv_dx, int m_floor,
                       double* value, double* derivative) const {
  double p = x * inv_dx + 1.0;
  int m = static_cast<int>(p);
  if (m < m_floor) m = m_floor;
  if (m > n - 1) m = n - 1;
  p -= static_cast<double>(m);
  if (p > 1.0) p = 1.0;
  const double* c = spline[m].c;
  *derivative = (c[0] * p + c[1]) * p + c[2];
  *value = ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

PairOutput EamFuncfl::PairScalar(int pass, double r2, int, int, int i, int j) {
  if (!(r2 < cut2_) || !(r2 > 0.0)) return {};
  const double r = std::sqrt(r2);
  double z2 = 0.0;
  double z2p = 0.0;
  double rho = 0.0;
  double rhop = 0.0;
  Sample(rhor_.data(), nr_, r, rdr_, 1, &rho, &rhop);
  if (pass == 0) {
    PairOutput out;
    out.value = rho;
    return out;
  }
  Sample(z2r_.data(), nr_, r, rdr_, 1, &z2, &z2p);
  const double recip = 1.0 / r;
  const double phi = z2 * recip;
  const double phip = z2p * recip - phi * recip;
  const double psip =
      fp_[static_cast<size_t>(i)] * rhop + fp_[static_cast<size_t>(j)] * rhop + phip;
  PairOutput out;
  out.force = -psip * recip;
  out.energy = phi;
  return out;
}

double EamFuncfl::AtomUpdate(int pass, int i, double value) {
  if (pass != 0) return 0.0;
  double embedding = 0.0;
  double derivative = 0.0;
  Sample(frho_.data(), nrho_, value, rdrho_, 1, &embedding, &derivative);
  if (value > rhomax_) embedding += derivative * (value - rhomax_);
  fp_[static_cast<size_t>(i)] = derivative;
  return embedding;
}

}  // namespace tilemd
