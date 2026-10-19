#include "lj_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "neighbor/cluster_order.hpp"

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
  out.wxx.assign(n, 0.0);
  out.wyy.assign(n, 0.0);
  out.wzz.assign(n, 0.0);
  out.wxy.assign(n, 0.0);
  out.wxz.assign(n, 0.0);
  out.wyz.assign(n, 0.0);

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
      out.pair_abs += 0.5 * std::abs(dx * fx + dy * fy + dz * fz);
      const double wxx = 0.5 * dx * fx;
      const double wyy = 0.5 * dy * fy;
      const double wzz = 0.5 * dz * fz;
      const double wxy = 0.5 * dx * fy;
      const double wxz = 0.5 * dx * fz;
      const double wyz = 0.5 * dy * fz;
      out.wxx[i] += wxx;
      out.wyy[i] += wyy;
      out.wzz[i] += wzz;
      out.wxy[i] += wxy;
      out.wxz[i] += wxz;
      out.wyz[i] += wyz;
      out.virial[0] += wxx;
      out.virial[1] += wyy;
      out.virial[2] += wzz;
      out.virial[3] += wxy;
      out.virial[4] += wxz;
      out.virial[5] += wyz;
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

namespace {

enum class XyzField { Species, Position, Velocity, Type, Virial, Text, Number };

struct XyzColumn {
  XyzField field = XyzField::Number;
  int count = 0;
};

std::vector<XyzColumn> ColumnsFromComment(const std::string& comment) {
  std::string spec = "species:S:1:pos:R:3:type:I:1";
  const auto at = comment.find("Properties=");
  if (at != std::string::npos) {
    const std::string rest = comment.substr(at + std::string("Properties=").size());
    const auto end = rest.find_first_of(" \t\"");
    spec = end == std::string::npos ? rest : rest.substr(0, end);
  }
  std::vector<std::string> parts;
  std::string token;
  for (char ch : spec) {
    if (ch == ':') {
      parts.push_back(token);
      token.clear();
    } else {
      token.push_back(ch);
    }
  }
  if (!token.empty()) parts.push_back(token);
  if (parts.empty() || parts.size() % 3 != 0) {
    throw std::runtime_error("extxyz Properties list is incomplete");
  }
  std::vector<XyzColumn> columns;
  bool saw_position = false;
  bool saw_type = false;
  for (size_t i = 0; i < parts.size(); i += 3) {
    int count = 0;
    try {
      count = std::stoi(parts[i + 2]);
    } catch (const std::exception&) {
      throw std::runtime_error("extxyz property count is not an integer");
    }
    if (count < 1) throw std::runtime_error("extxyz property count must be positive");
    XyzColumn column;
    column.count = count;
    const std::string& name = parts[i];
    const std::string& kind = parts[i + 1];
    if (kind == "S") {
      column.field = (name == "species") ? XyzField::Species : XyzField::Text;
    } else if (kind == "R" || kind == "I") {
      if (name == "pos" && count == 3)
        column.field = XyzField::Position;
      else if (name == "vel" && count == 3)
        column.field = XyzField::Velocity;
      else if (name == "type" && count == 1)
        column.field = XyzField::Type;
      else if (name == "virial" && count == 6)
        column.field = XyzField::Virial;
      else
        column.field = XyzField::Number;
    } else {
      throw std::runtime_error("unsupported extxyz property kind");
    }
    if (column.field == XyzField::Position) saw_position = true;
    if (column.field == XyzField::Type) saw_type = true;
    columns.push_back(column);
  }
  if (!saw_position || !saw_type) throw std::runtime_error("extxyz needs pos and type columns");
  return columns;
}

void ReadAtomLine(std::istream& in, const std::vector<XyzColumn>& columns, int i,
                  ExtxyzFrame* frame) {
  std::string species;
  int type = 0;
  bool saw_species = false;
  for (const XyzColumn& column : columns) {
    if (column.field == XyzField::Species || column.field == XyzField::Text) {
      for (int k = 0; k < column.count; ++k) {
        std::string word;
        in >> word;
        if (column.field == XyzField::Species && k == 0) species = word;
      }
      if (column.field == XyzField::Species) saw_species = true;
    } else if (column.field == XyzField::Position) {
      in >> frame->system.x[static_cast<size_t>(i)] >> frame->system.y[static_cast<size_t>(i)] >>
          frame->system.z[static_cast<size_t>(i)];
    } else if (column.field == XyzField::Velocity) {
      in >> frame->vx[static_cast<size_t>(i)] >> frame->vy[static_cast<size_t>(i)] >>
          frame->vz[static_cast<size_t>(i)];
    } else if (column.field == XyzField::Type) {
      in >> type;
    } else if (column.field == XyzField::Virial) {
      in >> frame->wxx[static_cast<size_t>(i)] >> frame->wyy[static_cast<size_t>(i)] >>
          frame->wzz[static_cast<size_t>(i)] >> frame->wxy[static_cast<size_t>(i)] >>
          frame->wxz[static_cast<size_t>(i)] >> frame->wyz[static_cast<size_t>(i)];
    } else {
      for (int k = 0; k < column.count; ++k) {
        double ignored = 0.0;
        in >> ignored;
      }
    }
  }
  if (!in || !saw_species || species.size() != 1) throw std::runtime_error("bad extxyz atom line");
  frame->system.type[static_cast<size_t>(i)] = type;
  if (TypeForSpecies(species[0]) != type) throw std::runtime_error("species/type mismatch");
}

}  // namespace

ExtxyzFrame ReadExtxyzFrame(const std::string& path, int frame) {
  if (frame < 0) throw std::runtime_error("extxyz frame must be non-negative");
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  ExtxyzFrame out;
  for (int index = 0; index <= frame; ++index) {
    int n = 0;
    in >> n;
    if (!in || n < 0) throw std::runtime_error("bad atom count in " + path);
    std::string comment;
    std::getline(in, comment);
    if (!std::getline(in, comment)) throw std::runtime_error("extxyz frame is missing a comment");
    if (index != frame) {
      for (int i = 0; i < n; ++i) {
        std::string skipped;
        if (!std::getline(in, skipped)) throw std::runtime_error("extxyz frame is truncated");
      }
      continue;
    }
    const auto lattice = comment.find("Lattice=\"");
    if (lattice == std::string::npos) throw std::runtime_error("extxyz is missing Lattice");
    {
      std::istringstream lattice_in(comment.substr(lattice + 9));
      double m[9];
      for (double& value : m) lattice_in >> value;
      if (!lattice_in) throw std::runtime_error("failed to parse Lattice");
      out.system.box[0] = m[0];
      out.system.box[1] = m[4];
      out.system.box[2] = m[8];
    }
    const auto shift = comment.find("shift=");
    if (shift == std::string::npos) throw std::runtime_error("extxyz is missing shift");
    {
      std::string word;
      std::istringstream shift_in(comment.substr(shift + 6));
      shift_in >> word;
      out.system.shift = ParseShift(word);
    }
    const std::vector<XyzColumn> columns = ColumnsFromComment(comment);
    bool has_velocity = false;
    bool has_virial = false;
    for (const XyzColumn& column : columns) {
      if (column.field == XyzField::Velocity) has_velocity = true;
      if (column.field == XyzField::Virial) has_virial = true;
    }
    out.system.type.resize(static_cast<size_t>(n));
    out.system.x.resize(static_cast<size_t>(n));
    out.system.y.resize(static_cast<size_t>(n));
    out.system.z.resize(static_cast<size_t>(n));
    if (has_velocity) {
      out.vx.assign(static_cast<size_t>(n), 0.0);
      out.vy.assign(static_cast<size_t>(n), 0.0);
      out.vz.assign(static_cast<size_t>(n), 0.0);
    }
    if (has_virial) {
      out.wxx.assign(static_cast<size_t>(n), 0.0);
      out.wyy.assign(static_cast<size_t>(n), 0.0);
      out.wzz.assign(static_cast<size_t>(n), 0.0);
      out.wxy.assign(static_cast<size_t>(n), 0.0);
      out.wxz.assign(static_cast<size_t>(n), 0.0);
      out.wyz.assign(static_cast<size_t>(n), 0.0);
    }
    for (int i = 0; i < n; ++i) ReadAtomLine(in, columns, i, &out);
  }
  return out;
}

System ReadExtxyz(const std::string& path) { return ReadExtxyzFrame(path, 0).system; }

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

namespace {

double WrapCoordinate(double value, double length, int* image) {
  return tilemd::WrapImage(value, length, image);
}

Observables ForcesAt(System& system, const Dynamics& state) {
  system.x = state.x;
  system.y = state.y;
  system.z = state.z;
  return Compute(system);
}

void CheckDynamics(const System& system, const Dynamics& state, double dt, double mass) {
  if (!(mass > 0.0)) throw std::runtime_error("mass must be positive");
  if (!(dt > 0.0)) throw std::runtime_error("dt must be positive");
  const int n = NumAtoms(system);
  const bool sized =
      static_cast<int>(state.x.size()) == n && static_cast<int>(state.y.size()) == n &&
      static_cast<int>(state.z.size()) == n && static_cast<int>(state.vx.size()) == n &&
      static_cast<int>(state.vy.size()) == n && static_cast<int>(state.vz.size()) == n &&
      static_cast<int>(state.imgx.size()) == n && static_cast<int>(state.imgy.size()) == n &&
      static_cast<int>(state.imgz.size()) == n;
  if (!sized) throw std::runtime_error("dynamics size does not match the system");
}

}  // namespace

Dynamics MakeDynamics(const System& system, const std::vector<double>& vx,
                      const std::vector<double>& vy, const std::vector<double>& vz) {
  const int n = NumAtoms(system);
  if (static_cast<int>(vx.size()) != n || static_cast<int>(vy.size()) != n ||
      static_cast<int>(vz.size()) != n) {
    throw std::runtime_error("velocity size does not match the system");
  }
  Dynamics state;
  state.x = system.x;
  state.y = system.y;
  state.z = system.z;
  state.vx = vx;
  state.vy = vy;
  state.vz = vz;
  state.imgx.assign(static_cast<size_t>(n), 0);
  state.imgy.assign(static_cast<size_t>(n), 0);
  state.imgz.assign(static_cast<size_t>(n), 0);
  return state;
}

void VelocityVerletStep(System& system, Dynamics& state, double dt, double mass) {
  CheckDynamics(system, state, dt, mass);
  const Observables first = ForcesAt(system, state);
  const double half = 0.5 * dt / mass;
  const int n = NumAtoms(system);
  for (int i = 0; i < n; ++i) {
    state.vx[static_cast<size_t>(i)] += half * first.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] += half * first.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] += half * first.fz[static_cast<size_t>(i)];
    state.x[static_cast<size_t>(i)] =
        WrapCoordinate(state.x[static_cast<size_t>(i)] + dt * state.vx[static_cast<size_t>(i)],
                       system.box[0], &state.imgx[static_cast<size_t>(i)]);
    state.y[static_cast<size_t>(i)] =
        WrapCoordinate(state.y[static_cast<size_t>(i)] + dt * state.vy[static_cast<size_t>(i)],
                       system.box[1], &state.imgy[static_cast<size_t>(i)]);
    state.z[static_cast<size_t>(i)] =
        WrapCoordinate(state.z[static_cast<size_t>(i)] + dt * state.vz[static_cast<size_t>(i)],
                       system.box[2], &state.imgz[static_cast<size_t>(i)]);
  }
  const Observables second = ForcesAt(system, state);
  for (int i = 0; i < n; ++i) {
    state.vx[static_cast<size_t>(i)] += half * second.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] += half * second.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] += half * second.fz[static_cast<size_t>(i)];
  }
}

void PrepareLeapfrog(System& system, Dynamics& state, double dt, double mass) {
  CheckDynamics(system, state, dt, mass);
  const Observables force = ForcesAt(system, state);
  const double half = 0.5 * dt / mass;
  const int n = NumAtoms(system);
  for (int i = 0; i < n; ++i) {
    state.vx[static_cast<size_t>(i)] -= half * force.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] -= half * force.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] -= half * force.fz[static_cast<size_t>(i)];
  }
}

void LeapfrogStep(System& system, Dynamics& state, double dt, double mass) {
  CheckDynamics(system, state, dt, mass);
  const Observables force = ForcesAt(system, state);
  const double scale = dt / mass;
  const int n = NumAtoms(system);
  for (int i = 0; i < n; ++i) {
    state.vx[static_cast<size_t>(i)] += scale * force.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] += scale * force.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] += scale * force.fz[static_cast<size_t>(i)];
    state.x[static_cast<size_t>(i)] =
        WrapCoordinate(state.x[static_cast<size_t>(i)] + dt * state.vx[static_cast<size_t>(i)],
                       system.box[0], &state.imgx[static_cast<size_t>(i)]);
    state.y[static_cast<size_t>(i)] =
        WrapCoordinate(state.y[static_cast<size_t>(i)] + dt * state.vy[static_cast<size_t>(i)],
                       system.box[1], &state.imgy[static_cast<size_t>(i)]);
    state.z[static_cast<size_t>(i)] =
        WrapCoordinate(state.z[static_cast<size_t>(i)] + dt * state.vz[static_cast<size_t>(i)],
                       system.box[2], &state.imgz[static_cast<size_t>(i)]);
  }
}

void IntegerTimeVelocity(const System& system, const Dynamics& half_step, double dt, double mass,
                         std::vector<double>& vx, std::vector<double>& vy,
                         std::vector<double>& vz) {
  CheckDynamics(system, half_step, dt, mass);
  System copy = system;
  const Observables force = ForcesAt(copy, half_step);
  const double half = 0.5 * dt / mass;
  const int n = NumAtoms(system);
  vx.resize(static_cast<size_t>(n));
  vy.resize(static_cast<size_t>(n));
  vz.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    vx[static_cast<size_t>(i)] =
        half_step.vx[static_cast<size_t>(i)] + half * force.fx[static_cast<size_t>(i)];
    vy[static_cast<size_t>(i)] =
        half_step.vy[static_cast<size_t>(i)] + half * force.fy[static_cast<size_t>(i)];
    vz[static_cast<size_t>(i)] =
        half_step.vz[static_cast<size_t>(i)] + half * force.fz[static_cast<size_t>(i)];
  }
}

namespace {

std::uint32_t MulHi(std::uint32_t a, std::uint32_t b, std::uint32_t* lo) {
  const std::uint64_t product = static_cast<std::uint64_t>(a) * b;
  *lo = static_cast<std::uint32_t>(product);
  return static_cast<std::uint32_t>(product >> 32);
}

}  // namespace

void StandardNormals(std::uint32_t seed, std::uint64_t step, int gid, double out[3]) {
  const PhiloxWord word =
      Philox4x32_10(static_cast<std::uint32_t>(gid), static_cast<std::uint32_t>(step),
                    static_cast<std::uint32_t>(step >> 32), 0u, seed, 0u);
  const std::uint32_t bits[4] = {word.w[0], word.w[1], word.w[2], word.w[3]};
  double normal[4];
  for (int pair = 0; pair < 2; ++pair) {
    const double u1 = (static_cast<double>(bits[2 * pair]) + 0.5) * (1.0 / 4294967296.0);
    const double u2 = (static_cast<double>(bits[2 * pair + 1]) + 0.5) * (1.0 / 4294967296.0);
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 6.28318530717958647692 * u2;
    normal[2 * pair] = radius * std::cos(angle);
    normal[2 * pair + 1] = radius * std::sin(angle);
  }
  out[0] = normal[0];
  out[1] = normal[1];
  out[2] = normal[2];
}

PhiloxWord Philox4x32_10(std::uint32_t c0, std::uint32_t c1, std::uint32_t c2, std::uint32_t c3,
                         std::uint32_t k0, std::uint32_t k1) {
  std::uint32_t ctr[4] = {c0, c1, c2, c3};
  std::uint32_t key0 = k0;
  std::uint32_t key1 = k1;
  for (int round = 0; round < 10; ++round) {
    std::uint32_t lo0 = 0;
    std::uint32_t lo1 = 0;
    const std::uint32_t hi0 = MulHi(0xD2511F53u, ctr[0], &lo0);
    const std::uint32_t hi1 = MulHi(0xCD9E8D57u, ctr[2], &lo1);
    const std::uint32_t next0 = hi1 ^ ctr[1] ^ key0;
    const std::uint32_t next1 = lo1;
    const std::uint32_t next2 = hi0 ^ ctr[3] ^ key1;
    const std::uint32_t next3 = lo0;
    ctr[0] = next0;
    ctr[1] = next1;
    ctr[2] = next2;
    ctr[3] = next3;
    key0 += 0x9E3779B9u;
    key1 += 0xBB67AE85u;
  }
  PhiloxWord out;
  out.w[0] = ctr[0];
  out.w[1] = ctr[1];
  out.w[2] = ctr[2];
  out.w[3] = ctr[3];
  return out;
}

void LangevinBaoabStep(System& system, Dynamics& state, double dt, double mass,
                       const Langevin& thermo, std::uint64_t step) {
  CheckDynamics(system, state, dt, mass);
  if (!(thermo.gamma >= 0.0)) throw std::runtime_error("friction must be non-negative");
  if (!(thermo.temperature >= 0.0)) throw std::runtime_error("temperature must be non-negative");
  if (!(thermo.gamma > 0.0)) {
    VelocityVerletStep(system, state, dt, mass);
    return;
  }
  const Observables first = ForcesAt(system, state);
  const double half = 0.5 * dt / mass;
  const double half_dt = 0.5 * dt;
  const int n = NumAtoms(system);
  for (int i = 0; i < n; ++i) {
    state.vx[static_cast<size_t>(i)] += half * first.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] += half * first.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] += half * first.fz[static_cast<size_t>(i)];
    state.x[static_cast<size_t>(i)] =
        WrapCoordinate(state.x[static_cast<size_t>(i)] + half_dt * state.vx[static_cast<size_t>(i)],
                       system.box[0], &state.imgx[static_cast<size_t>(i)]);
    state.y[static_cast<size_t>(i)] =
        WrapCoordinate(state.y[static_cast<size_t>(i)] + half_dt * state.vy[static_cast<size_t>(i)],
                       system.box[1], &state.imgy[static_cast<size_t>(i)]);
    state.z[static_cast<size_t>(i)] =
        WrapCoordinate(state.z[static_cast<size_t>(i)] + half_dt * state.vz[static_cast<size_t>(i)],
                       system.box[2], &state.imgz[static_cast<size_t>(i)]);
  }
  const double decay = std::exp(-thermo.gamma * dt);
  const double noise = std::sqrt(thermo.temperature * (1.0 - decay * decay) / mass);
  for (int i = 0; i < n; ++i) {
    double draw[3] = {};
    if (noise > 0.0) StandardNormals(thermo.seed, step, i, draw);
    state.vx[static_cast<size_t>(i)] = decay * state.vx[static_cast<size_t>(i)] + noise * draw[0];
    state.vy[static_cast<size_t>(i)] = decay * state.vy[static_cast<size_t>(i)] + noise * draw[1];
    state.vz[static_cast<size_t>(i)] = decay * state.vz[static_cast<size_t>(i)] + noise * draw[2];
    state.x[static_cast<size_t>(i)] =
        WrapCoordinate(state.x[static_cast<size_t>(i)] + half_dt * state.vx[static_cast<size_t>(i)],
                       system.box[0], &state.imgx[static_cast<size_t>(i)]);
    state.y[static_cast<size_t>(i)] =
        WrapCoordinate(state.y[static_cast<size_t>(i)] + half_dt * state.vy[static_cast<size_t>(i)],
                       system.box[1], &state.imgy[static_cast<size_t>(i)]);
    state.z[static_cast<size_t>(i)] =
        WrapCoordinate(state.z[static_cast<size_t>(i)] + half_dt * state.vz[static_cast<size_t>(i)],
                       system.box[2], &state.imgz[static_cast<size_t>(i)]);
  }
  const Observables second = ForcesAt(system, state);
  for (int i = 0; i < n; ++i) {
    state.vx[static_cast<size_t>(i)] += half * second.fx[static_cast<size_t>(i)];
    state.vy[static_cast<size_t>(i)] += half * second.fy[static_cast<size_t>(i)];
    state.vz[static_cast<size_t>(i)] += half * second.fz[static_cast<size_t>(i)];
  }
}

T1v2 CompareT1v2(const Observables& got, const Observables& reference) {
  const int n = static_cast<int>(reference.fx.size());
  if (n < 1 || static_cast<int>(got.fx.size()) != n) {
    throw std::runtime_error("T1-v2 force vectors have different lengths");
  }
  double sum_delta = 0.0;
  double sum_ref = 0.0;
  double max_delta = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dx = got.fx[static_cast<size_t>(i)] - reference.fx[static_cast<size_t>(i)];
    const double dy = got.fy[static_cast<size_t>(i)] - reference.fy[static_cast<size_t>(i)];
    const double dz = got.fz[static_cast<size_t>(i)] - reference.fz[static_cast<size_t>(i)];
    const double d2 = dx * dx + dy * dy + dz * dz;
    sum_delta += d2;
    const double fx = reference.fx[static_cast<size_t>(i)];
    const double fy = reference.fy[static_cast<size_t>(i)];
    const double fz = reference.fz[static_cast<size_t>(i)];
    sum_ref += fx * fx + fy * fy + fz * fz;
    max_delta = std::max(max_delta, std::sqrt(d2));
  }
  const double rms = std::sqrt(sum_ref / static_cast<double>(n));
  const double scale = std::max(rms, 1.0e-30);
  T1v2 out;
  out.force_rms_rel = std::sqrt(sum_delta / static_cast<double>(n)) / scale;
  out.force_max_rel = max_delta / scale;
  const double energy_scale = std::max(std::abs(reference.potential_energy), 1.0e-30);
  out.energy_rel = std::abs(got.potential_energy - reference.potential_energy) / energy_scale;
  const double virial_scale = std::max(reference.pair_abs, 1.0e-30);
  double virial_delta = 0.0;
  for (int k = 0; k < 6; ++k) {
    virial_delta = std::max(virial_delta, std::abs(got.virial[k] - reference.virial[k]));
  }
  out.virial_rel = virial_delta / virial_scale;
  out.pass = out.force_rms_rel <= 1.0e-5 && out.force_max_rel <= 2.0e-4 &&
             out.energy_rel <= 1.0e-6 && out.virial_rel <= 1.0e-5;
  return out;
}

bool PassesT4v2(double slope_tile, double slope_lammps, double se_lammps) {
  return std::abs(slope_tile) <= std::max(2.0 * std::abs(slope_lammps), 3.0 * se_lammps);
}

}  // namespace tilemd::ref
