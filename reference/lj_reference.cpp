#include "lj_reference.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tilemd::ref {
namespace {

constexpr int kMaxTypes = 8;

struct Prepared {
  bool defined = false;
  double sigma2 = 0.0;
  double eps4 = 0.0;
  double eps24 = 0.0;
  double rc = 0.0;
  double rc2 = 0.0;
  double u_shift = 0.0;
  double s_rc = 0.0;
};

void PrepareOne(Prepared* row, const PairCoeff& coeff, Shift shift) {
  if (coeff.type_i < 0 || coeff.type_j < 0 || coeff.type_i >= kMaxTypes ||
      coeff.type_j >= kMaxTypes) {
    throw std::runtime_error("pair type is outside 0..7");
  }
  if (!(coeff.cutoff > 0.0) || !(coeff.sigma > 0.0)) {
    throw std::runtime_error("sigma and cutoff must be positive");
  }
  Prepared& p = row[coeff.type_j];
  p.defined = true;
  p.sigma2 = coeff.sigma * coeff.sigma;
  p.eps4 = 4.0 * coeff.epsilon;
  p.eps24 = 24.0 * coeff.epsilon;
  p.rc = coeff.cutoff;
  p.rc2 = coeff.cutoff * coeff.cutoff;
  const double inv_rc2 = 1.0 / p.rc2;
  const double sr2 = p.sigma2 * inv_rc2;
  const double sr6 = sr2 * sr2 * sr2;
  const double u_c = p.eps4 * sr6 * (sr6 - 1.0);
  p.s_rc = p.eps24 * inv_rc2 * sr6 * (2.0 * sr6 - 1.0);
  p.u_shift = (shift == Shift::None) ? 0.0 : u_c;
}

std::string ShiftName(Shift shift) {
  switch (shift) {
    case Shift::None:
      return "none";
    case Shift::Energy:
      return "energy";
    case Shift::Force:
      return "force";
  }
  throw std::runtime_error("unknown shift");
}

Shift ParseShift(const std::string& text) {
  if (text == "none") return Shift::None;
  if (text == "energy") return Shift::Energy;
  if (text == "force") return Shift::Force;
  throw std::runtime_error("lj_shift must be none, energy, or force");
}

char SpeciesForType(int type) {
  if (type < 0 || type > 25) throw std::runtime_error("type does not map to a species letter");
  return static_cast<char>('A' + type);
}

int TypeForSpecies(char species) {
  if (species < 'A' || species > 'Z') throw std::runtime_error("species must be A-Z");
  return species - 'A';
}

// Subset reader for the JSON this file writes. It is not a general parser.
std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

const char* SkipWs(const char* p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
  return p;
}

double ReadNumber(const char*& p) {
  p = SkipWs(p);
  char* end = nullptr;
  const double value = std::strtod(p, &end);
  if (end == p) throw std::runtime_error("expected a number in JSON");
  p = end;
  return value;
}

std::string ReadString(const char*& p) {
  p = SkipWs(p);
  if (*p != '"') throw std::runtime_error("expected a string in JSON");
  ++p;
  std::string out;
  while (*p && *p != '"') out.push_back(*p++);
  if (*p != '"') throw std::runtime_error("unterminated string in JSON");
  ++p;
  return out;
}

const char* FindKey(const char* json, const char* key) {
  const std::string pattern = std::string("\"") + key + "\"";
  const char* p = std::strstr(json, pattern.c_str());
  if (!p) throw std::runtime_error(std::string("missing JSON key ") + key);
  p += pattern.size();
  p = SkipWs(p);
  if (*p != ':') throw std::runtime_error(std::string("missing colon after ") + key);
  return p + 1;
}

}  // namespace

double MinImage(double delta, double length) { return delta - length * std::rint(delta / length); }

int NumAtoms(const System& system) { return static_cast<int>(system.x.size()); }

Observables Compute(const System& system) {
  const int n = NumAtoms(system);
  if (static_cast<int>(system.y.size()) != n || static_cast<int>(system.z.size()) != n ||
      static_cast<int>(system.type.size()) != n) {
    throw std::runtime_error("position and type arrays differ in length");
  }
  if (!(system.box[0] > 0.0) || !(system.box[1] > 0.0) || !(system.box[2] > 0.0)) {
    throw std::runtime_error("box lengths must be positive");
  }

  Prepared table[kMaxTypes][kMaxTypes] = {};
  if (system.pairs.empty()) throw std::runtime_error("no pair coefficients");
  for (const PairCoeff& coeff : system.pairs) {
    PrepareOne(table[coeff.type_i], coeff, system.shift);
    if (coeff.type_i != coeff.type_j) {
      PairCoeff swapped = coeff;
      swapped.type_i = coeff.type_j;
      swapped.type_j = coeff.type_i;
      PrepareOne(table[swapped.type_i], swapped, system.shift);
    }
  }

  Observables out;
  out.fx.assign(n, 0.0);
  out.fy.assign(n, 0.0);
  out.fz.assign(n, 0.0);
  out.energy.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    const int ti = system.type[i];
    if (ti < 0 || ti >= kMaxTypes) throw std::runtime_error("atom type is outside 0..7");
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      const int tj = system.type[j];
      if (tj < 0 || tj >= kMaxTypes) throw std::runtime_error("atom type is outside 0..7");
      const Prepared& p = table[ti][tj];
      if (!p.defined) throw std::runtime_error("missing pair coefficient");

      const double dx = MinImage(system.x[i] - system.x[j], system.box[0]);
      const double dy = MinImage(system.y[i] - system.y[j], system.box[1]);
      const double dz = MinImage(system.z[i] - system.z[j], system.box[2]);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= p.rc2) continue;

      const double inv_r2 = 1.0 / r2;
      const double sr2 = p.sigma2 * inv_r2;
      const double sr6 = sr2 * sr2 * sr2;
      double s = p.eps24 * inv_r2 * sr6 * (2.0 * sr6 - 1.0);
      double u = p.eps4 * sr6 * (sr6 - 1.0) - p.u_shift;
      if (system.shift == Shift::Force) {
        const double r = std::sqrt(r2);
        // U_sf = U - U(rc) - (r - rc) U'(rc), with U'(rc) = -s(rc^2) * rc.
        u += (r - p.rc) * p.s_rc * p.rc;
        s -= p.s_rc * p.rc / r;
      }
      const double fx = s * dx;
      const double fy = s * dy;
      const double fz = s * dz;
      out.fx[i] += fx;
      out.fy[i] += fy;
      out.fz[i] += fz;
      out.energy[i] += 0.5 * u;
      out.potential_energy += 0.5 * u;
      out.virial[0] += 0.5 * dx * fx;
      out.virial[1] += 0.5 * dy * fy;
      out.virial[2] += 0.5 * dz * fz;
      out.virial[3] += 0.5 * dx * fy;
      out.virial[4] += 0.5 * dx * fz;
      out.virial[5] += 0.5 * dy * fz;
    }
  }
  return out;
}

void WriteExtxyz(const std::string& path, const System& system) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  const int n = NumAtoms(system);
  out.setf(std::ios::scientific);
  out.precision(17);
  out << n << "\n";
  out << "Lattice=\"" << system.box[0] << " 0 0 0 " << system.box[1] << " 0 0 0 " << system.box[2]
      << "\" Properties=species:S:1:pos:R:3:type:I:1 shift=" << ShiftName(system.shift)
      << " pbc=\"T T T\"\n";
  for (int i = 0; i < n; ++i) {
    out << SpeciesForType(system.type[i]) << " " << system.x[i] << " " << system.y[i] << " "
        << system.z[i] << " " << system.type[i] << "\n";
  }
}

void WriteObservablesJson(const std::string& path, const std::string& name, const System& system,
                          const Observables& obs) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  const int n = NumAtoms(system);
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "{\n";
  out << "  \"name\": \"" << name << "\",\n";
  out << "  \"n\": " << n << ",\n";
  out << "  \"box\": [" << system.box[0] << ", " << system.box[1] << ", " << system.box[2]
      << "],\n";
  out << "  \"lj_shift\": \"" << ShiftName(system.shift) << "\",\n";
  out << "  \"pairs\": [\n";
  for (size_t k = 0; k < system.pairs.size(); ++k) {
    const PairCoeff& c = system.pairs[k];
    out << "    {\"i\": " << c.type_i << ", \"j\": " << c.type_j << ", \"epsilon\": " << c.epsilon
        << ", \"sigma\": " << c.sigma << ", \"cutoff\": " << c.cutoff << "}";
    out << (k + 1 == system.pairs.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"potential_energy\": " << obs.potential_energy << ",\n";
  out << "  \"virial\": [";
  for (int k = 0; k < 6; ++k) out << (k ? ", " : "") << obs.virial[k];
  out << "],\n";
  out << "  \"energy\": [";
  if (static_cast<int>(obs.energy.size()) == n) {
    for (int i = 0; i < n; ++i) out << (i ? ", " : "") << obs.energy[i];
  }
  out << "],\n";
  out << "  \"forces\": [\n";
  for (int i = 0; i < n; ++i) {
    out << "    [" << obs.fx[i] << ", " << obs.fy[i] << ", " << obs.fz[i] << "]";
    out << (i + 1 == n ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

System ReadExtxyz(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  System system;
  int n = 0;
  in >> n;
  if (!in || n < 0) throw std::runtime_error("bad atom count in " + path);
  std::string comment;
  std::getline(in, comment);
  std::getline(in, comment);
  const auto lattice = comment.find("Lattice=\"");
  if (lattice == std::string::npos) throw std::runtime_error("extxyz is missing Lattice");
  {
    std::istringstream lattice_in(comment.substr(lattice + 9));
    double m[9];
    for (double& value : m) lattice_in >> value;
    if (!lattice_in) throw std::runtime_error("failed to parse Lattice");
    system.box[0] = m[0];
    system.box[1] = m[4];
    system.box[2] = m[8];
  }
  const auto shift = comment.find("shift=");
  if (shift == std::string::npos) throw std::runtime_error("extxyz is missing shift");
  {
    std::string word;
    std::istringstream shift_in(comment.substr(shift + 6));
    shift_in >> word;
    system.shift = ParseShift(word);
  }
  system.type.resize(n);
  system.x.resize(n);
  system.y.resize(n);
  system.z.resize(n);
  for (int i = 0; i < n; ++i) {
    std::string species;
    int type = 0;
    in >> species >> system.x[i] >> system.y[i] >> system.z[i] >> type;
    if (!in || species.size() != 1) throw std::runtime_error("bad extxyz atom line");
    system.type[i] = type;
    if (TypeForSpecies(species[0]) != type) throw std::runtime_error("species/type mismatch");
  }
  return system;
}

void ReadSystemHeader(const std::string& path, System& system) {
  const std::string text = ReadFile(path);
  const char* shift = FindKey(text.c_str(), "lj_shift");
  system.shift = ParseShift(ReadString(shift));
  const char* box = FindKey(text.c_str(), "box");
  box = SkipWs(box);
  if (*box != '[') throw std::runtime_error("box is not an array");
  ++box;
  for (int axis = 0; axis < 3; ++axis) {
    system.box[axis] = ReadNumber(box);
    box = SkipWs(box);
    if (axis < 2) {
      if (*box != ',') throw std::runtime_error("box entry missing");
      ++box;
    }
  }
  const char* pairs = FindKey(text.c_str(), "pairs");
  pairs = SkipWs(pairs);
  if (*pairs != '[') throw std::runtime_error("pairs is not an array");
  ++pairs;
  system.pairs.clear();
  while (true) {
    pairs = SkipWs(pairs);
    if (*pairs == ']') break;
    if (*pairs != '{') throw std::runtime_error("pair entry is not an object");
    const char* end = std::strchr(pairs, '}');
    if (!end) throw std::runtime_error("pair entry is not closed");
    const std::string object(pairs, end + 1);
    PairCoeff coeff;
    const char* i = FindKey(object.c_str(), "i");
    coeff.type_i = static_cast<int>(ReadNumber(i));
    const char* j = FindKey(object.c_str(), "j");
    coeff.type_j = static_cast<int>(ReadNumber(j));
    const char* epsilon = FindKey(object.c_str(), "epsilon");
    coeff.epsilon = ReadNumber(epsilon);
    const char* sigma = FindKey(object.c_str(), "sigma");
    coeff.sigma = ReadNumber(sigma);
    const char* cutoff = FindKey(object.c_str(), "cutoff");
    coeff.cutoff = ReadNumber(cutoff);
    system.pairs.push_back(coeff);
    pairs = end + 1;
    pairs = SkipWs(pairs);
    if (*pairs == ',') ++pairs;
  }
}

Observables ReadObservablesJson(const std::string& path) {
  const std::string text = ReadFile(path);
  Observables obs;
  const char* pe = FindKey(text.c_str(), "potential_energy");
  obs.potential_energy = ReadNumber(pe);
  const char* virial = FindKey(text.c_str(), "virial");
  virial = SkipWs(virial);
  if (*virial != '[') throw std::runtime_error("virial is not an array");
  ++virial;
  for (int k = 0; k < 6; ++k) {
    obs.virial[k] = ReadNumber(virial);
    virial = SkipWs(virial);
    if (k < 5) {
      if (*virial != ',') throw std::runtime_error("virial entry missing");
      ++virial;
    }
  }
  const char* forces = FindKey(text.c_str(), "forces");
  forces = SkipWs(forces);
  if (*forces != '[') throw std::runtime_error("forces is not an array");
  ++forces;
  while (true) {
    forces = SkipWs(forces);
    if (*forces == ']') break;
    if (*forces != '[') throw std::runtime_error("force row is not an array");
    ++forces;
    obs.fx.push_back(ReadNumber(forces));
    forces = SkipWs(forces);
    if (*forces != ',') throw std::runtime_error("force row missing a comma");
    ++forces;
    obs.fy.push_back(ReadNumber(forces));
    forces = SkipWs(forces);
    if (*forces != ',') throw std::runtime_error("force row missing a comma");
    ++forces;
    obs.fz.push_back(ReadNumber(forces));
    forces = SkipWs(forces);
    if (*forces != ']') throw std::runtime_error("force row is not closed");
    ++forces;
    forces = SkipWs(forces);
    if (*forces == ',') ++forces;
  }
  return obs;
}

LammpsDump ReadLammpsDump(const std::string& path, Shift shift,
                          const std::vector<PairCoeff>& pairs) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  LammpsDump dump;
  dump.system.shift = shift;
  dump.system.pairs = pairs;
  std::string line;
  int n = -1;
  while (std::getline(in, line)) {
    if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) {
      in >> n;
      std::getline(in, line);
    } else if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) {
      double lo, hi;
      for (int axis = 0; axis < 3; ++axis) {
        in >> lo >> hi;
        dump.system.box[axis] = hi - lo;
        std::getline(in, line);
      }
    } else if (line.rfind("ITEM: ATOMS", 0) == 0) {
      if (n < 0) throw std::runtime_error("dump has no atom count");
      dump.system.type.assign(n, 0);
      dump.system.x.assign(n, 0.0);
      dump.system.y.assign(n, 0.0);
      dump.system.z.assign(n, 0.0);
      dump.observables.fx.assign(n, 0.0);
      dump.observables.fy.assign(n, 0.0);
      dump.observables.fz.assign(n, 0.0);
      for (int row = 0; row < n; ++row) {
        int id = 0;
        int type = 0;
        double x, y, z, fx, fy, fz;
        in >> id >> type >> x >> y >> z >> fx >> fy >> fz;
        if (!in) throw std::runtime_error("truncated LAMMPS dump");
        if (id < 1 || id > n) throw std::runtime_error("LAMMPS id is outside 1..N");
        const int slot = id - 1;
        dump.system.type[slot] = type - 1;
        auto wrap = [](double value, double length) {
          double wrapped = value - length * std::floor(value / length);
          if (wrapped >= length) wrapped -= length;
          if (wrapped < 0.0) wrapped += length;
          return wrapped;
        };
        dump.system.x[slot] = wrap(x, dump.system.box[0]);
        dump.system.y[slot] = wrap(y, dump.system.box[1]);
        dump.system.z[slot] = wrap(z, dump.system.box[2]);
        dump.observables.fx[slot] = fx;
        dump.observables.fy[slot] = fy;
        dump.observables.fz[slot] = fz;
      }
    }
  }
  if (n < 0 || NumAtoms(dump.system) != n) throw std::runtime_error("dump did not contain atoms");
  return dump;
}

}  // namespace tilemd::ref
