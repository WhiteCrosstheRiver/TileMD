#include "force/eam.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "force/eam_eval.hpp"

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

void BuildKnots(int n, double delta, const double* values, EamKnot* spline) {
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

std::string NextContentLine(std::istream& in) {
  std::string line;
  while (std::getline(in, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    if (line.find_first_not_of(" \t\r") != std::string::npos) return line;
  }
  throw std::runtime_error("EAM setfl file ended early");
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

void EamFuncfl::BuildSpline(int n, double delta, const double* values, EamKnot* spline) const {
  BuildKnots(n, delta, values, spline);
}

void EamFuncfl::Sample(const EamKnot* spline, int n, double x, double inv_dx, int m_floor,
                       double* value, double* derivative) const {
  SampleKnots(spline, n, x, inv_dx, m_floor, value, derivative);
}

PairOutput EamFuncfl::PairScalar(int pass, double r2, int, int, int i, int j) {
  return EamFuncflPair(rhor_.data(), z2r_.data(), fp_.data(), nr_, rdr_, cut2_, pass, r2, i, j);
}

double EamFuncfl::AtomUpdate(int pass, int i, double value) {
  if (pass != 0) return 0.0;
  double derivative = 0.0;
  const double embedding = EamEmbed(frho_.data(), nrho_, rdrho_, rhomax_, value, &derivative);
  fp_[static_cast<size_t>(i)] = derivative;
  return embedding;
}

EamSetfl::EamSetfl(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open EAM file " + path);
  std::string line;
  for (int i = 0; i < 3; ++i) {
    if (!std::getline(in, line)) throw std::runtime_error("EAM setfl file is missing its header");
  }
  {
    const std::string elements = NextContentLine(in);
    std::istringstream header(elements);
    int count = 0;
    header >> count;
    if (!header || count < 1) throw std::runtime_error("EAM setfl element count is invalid");
    elements_.resize(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      if (!(header >> elements_[static_cast<size_t>(i)])) {
        throw std::runtime_error("EAM setfl element name is missing");
      }
    }
  }
  int nrho = 0;
  int nr = 0;
  double drho = 0.0;
  double dr = 0.0;
  double cut = 0.0;
  {
    std::istringstream grid(NextContentLine(in));
    grid >> nrho >> drho >> nr >> dr >> cut;
    if (!grid || nrho < 5 || nr < 5 || !(drho > 0.0) || !(dr > 0.0) || !(cut > 0.0)) {
      throw std::runtime_error("EAM setfl grid header is invalid");
    }
  }
  const int nelement = static_cast<int>(elements_.size());
  mass_.assign(static_cast<size_t>(nelement), 0.0);
  frho_.assign(static_cast<size_t>(nelement), {});
  rhor_.assign(static_cast<size_t>(nelement), {});
  for (int element = 0; element < nelement; ++element) {
    std::istringstream mass_line(NextContentLine(in));
    int atomic_number = 0;
    mass_line >> atomic_number >> mass_[static_cast<size_t>(element)];
    if (!mass_line || !(mass_[static_cast<size_t>(element)] > 0.0)) {
      throw std::runtime_error("EAM setfl mass line is invalid");
    }
    const std::vector<double> embedding = ReadDoubles(in, nrho);
    const std::vector<double> density = ReadDoubles(in, nr);
    std::vector<double> embedding_table(static_cast<size_t>(nrho) + 1);
    std::vector<double> density_table(static_cast<size_t>(nr) + 1);
    for (int m = 1; m <= nrho; ++m)
      embedding_table[static_cast<size_t>(m)] = embedding[static_cast<size_t>(m - 1)];
    for (int m = 1; m <= nr; ++m)
      density_table[static_cast<size_t>(m)] = density[static_cast<size_t>(m - 1)];
    frho_[static_cast<size_t>(element)].assign(static_cast<size_t>(nrho) + 1, {});
    rhor_[static_cast<size_t>(element)].assign(static_cast<size_t>(nr) + 1, {});
    BuildKnots(nrho, drho, embedding_table.data(), frho_[static_cast<size_t>(element)].data());
    BuildKnots(nr, dr, density_table.data(), rhor_[static_cast<size_t>(element)].data());
  }
  const int npair = nelement * (nelement + 1) / 2;
  z2r_.assign(static_cast<size_t>(npair), {});
  int pair = 0;
  for (int i = 0; i < nelement; ++i) {
    for (int j = 0; j <= i; ++j) {
      const std::vector<double> rphi = ReadDoubles(in, nr);
      std::vector<double> table(static_cast<size_t>(nr) + 1);
      for (int m = 1; m <= nr; ++m)
        table[static_cast<size_t>(m)] = rphi[static_cast<size_t>(m - 1)];
      z2r_[static_cast<size_t>(pair)].assign(static_cast<size_t>(nr) + 1, {});
      BuildKnots(nr, dr, table.data(), z2r_[static_cast<size_t>(pair)].data());
      ++pair;
    }
  }
  nrho_ = nrho;
  nr_ = nr;
  drho_ = drho;
  dr_ = dr;
  rdrho_ = 1.0 / drho;
  rdr_ = 1.0 / dr;
  cut_ = cut;
  cut2_ = cut * cut;
  rhomax_ = (nrho - 1) * drho;
}

void EamSetfl::SetAtoms(const int* type, int n) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  type_.assign(type, type + n);
  for (int value : type_) {
    if (value < 0 || value >= NumElements()) {
      throw std::invalid_argument("atom type is outside the setfl elements");
    }
  }
}

void EamSetfl::Prepare(int n) {
  if (n < 0) throw std::invalid_argument("atom count must be non-negative");
  if (static_cast<int>(type_.size()) != n) {
    throw std::invalid_argument("setfl atom types have not been set");
  }
  fp_.assign(static_cast<size_t>(n), 0.0);
}

int EamSetfl::PairIndex(int type_i, int type_j) const {
  int row = type_i;
  int col = type_j;
  if (row < col) {
    row = type_j;
    col = type_i;
  }
  return row * (row + 1) / 2 + col;
}

PairOutput EamSetfl::PairScalar(int pass, double r2, int type_i, int type_j, int i, int j) {
  return EamSetflPair(rhor_[static_cast<size_t>(type_j)].data(),
                      rhor_[static_cast<size_t>(type_i)].data(),
                      z2r_[static_cast<size_t>(PairIndex(type_i, type_j))].data(), fp_.data(), nr_,
                      rdr_, cut2_, pass, r2, i, j);
}

double EamSetfl::AtomUpdate(int pass, int i, double value) {
  if (pass != 0) return 0.0;
  const int element = type_[static_cast<size_t>(i)];
  double derivative = 0.0;
  const double embedding = EamEmbed(frho_[static_cast<size_t>(element)].data(), nrho_, rdrho_,
                                    rhomax_, value, &derivative);
  fp_[static_cast<size_t>(i)] = derivative;
  return embedding;
}

}  // namespace tilemd
