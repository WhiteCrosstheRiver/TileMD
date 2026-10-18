#include "io/input.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "force/eam.hpp"

namespace tilemd {
namespace {

struct Command {
  std::string key;
  std::vector<std::string> args;
};

std::vector<Command> Lex(const std::string& text) {
  std::vector<Command> commands;
  std::istringstream lines(text);
  std::string line;
  int number = 0;
  while (std::getline(lines, line)) {
    ++number;
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    std::istringstream words(line);
    Command command;
    if (!(words >> command.key)) continue;
    std::string word;
    while (words >> word) command.args.push_back(word);
    (void)number;
    commands.push_back(command);
  }
  return commands;
}

const Command* Find(const std::vector<Command>& commands, const std::string& key) {
  const Command* found = nullptr;
  for (const Command& command : commands) {
    if (command.key != key) continue;
    if (found != nullptr) throw std::runtime_error("repeated input key " + key);
    found = &command;
  }
  return found;
}

double Number(const Command& command, int index) {
  if (index < 0 || index >= static_cast<int>(command.args.size())) {
    throw std::runtime_error("missing number for " + command.key);
  }
  try {
    size_t used = 0;
    const double value = std::stod(command.args[static_cast<size_t>(index)], &used);
    if (used != command.args[static_cast<size_t>(index)].size()) {
      throw std::runtime_error("bad number for " + command.key);
    }
    return value;
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("bad number for " + command.key);
  }
}

int Integer(const Command& command, int index) {
  const double value = Number(command, index);
  if (value != std::floor(value))
    throw std::runtime_error("expected an integer for " + command.key);
  return static_cast<int>(value);
}

void RequireCount(const Command& command, int count) {
  if (static_cast<int>(command.args.size()) != count) {
    throw std::runtime_error("wrong number of values for " + command.key);
  }
}

void FillFcc(ref::System& system, int nx, int ny, int nz, double density) {
  if (nx < 1 || ny < 1 || nz < 1) throw std::runtime_error("fcc counts must be positive");
  if (!(density > 0.0)) throw std::runtime_error("density must be positive");
  const double a = std::cbrt(4.0 / density);
  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  system.box[0] = nx * a;
  system.box[1] = ny * a;
  system.box[2] = nz * a;
  system.type.clear();
  system.x.clear();
  system.y.clear();
  system.z.clear();
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        for (int corner = 0; corner < 4; ++corner) {
          system.type.push_back(0);
          system.x.push_back((ix + basis[corner][0]) * a);
          system.y.push_back((iy + basis[corner][1]) * a);
          system.z.push_back((iz + basis[corner][2]) * a);
        }
      }
    }
  }
}

bool LooksLikeSetfl(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  for (int i = 0; i < 4; ++i) {
    if (!std::getline(in, line)) return false;
  }
  std::istringstream header(line);
  int count = 0;
  std::string name;
  if (!(header >> count >> name) || count < 1) return false;
  char* end = nullptr;
  std::strtod(name.c_str(), &end);
  return end == name.c_str() || *end != '\0';
}

double AtomMass(const SimulationInput& input, int atom) {
  if (input.mass_by_type.empty()) return input.mass;
  const int type = input.system.type.at(static_cast<size_t>(atom));
  return input.mass_by_type.at(static_cast<size_t>(type));
}

void FillVelocities(SimulationInput& input, double temperature, std::uint32_t seed) {
  if (!(temperature >= 0.0)) throw std::runtime_error("temperature must be non-negative");
  const int n = ref::NumAtoms(input.system);
  if (n < 1) throw std::runtime_error("velocity requested before any atoms");
  input.vx.assign(static_cast<size_t>(n), 0.0);
  input.vy.assign(static_cast<size_t>(n), 0.0);
  input.vz.assign(static_cast<size_t>(n), 0.0);
  const double thermal = input.metal ? temperature * kMetalBoltz * kMetalFtm2v : temperature;
  for (int i = 0; i < n; ++i) {
    const double mass = AtomMass(input, i);
    if (!(mass > 0.0)) throw std::runtime_error("mass must be positive");
    const double scale = std::sqrt(thermal / mass);
    const ref::PhiloxWord word =
        ref::Philox4x32_10(static_cast<std::uint32_t>(i), 0u, 0u, 0u, seed, 0u);
    const std::uint32_t bits[4] = {word.w[0], word.w[1], word.w[2], word.w[3]};
    double normal[3];
    for (int pair = 0; pair < 2; ++pair) {
      const double u1 = (static_cast<double>(bits[2 * pair]) + 0.5) * (1.0 / 4294967296.0);
      const double u2 = (static_cast<double>(bits[2 * pair + 1]) + 0.5) * (1.0 / 4294967296.0);
      const double radius = std::sqrt(-2.0 * std::log(u1));
      const double angle = 6.28318530717958647692 * u2;
      if (pair == 0) {
        normal[0] = radius * std::cos(angle);
        normal[1] = radius * std::sin(angle);
      } else {
        normal[2] = radius * std::cos(angle);
      }
    }
    input.vx[static_cast<size_t>(i)] = scale * normal[0];
    input.vy[static_cast<size_t>(i)] = scale * normal[1];
    input.vz[static_cast<size_t>(i)] = scale * normal[2];
  }
  if (!input.metal) {
    double mean_x = 0.0;
    double mean_y = 0.0;
    double mean_z = 0.0;
    for (int i = 0; i < n; ++i) {
      mean_x += input.vx[static_cast<size_t>(i)];
      mean_y += input.vy[static_cast<size_t>(i)];
      mean_z += input.vz[static_cast<size_t>(i)];
    }
    mean_x /= static_cast<double>(n);
    mean_y /= static_cast<double>(n);
    mean_z /= static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
      input.vx[static_cast<size_t>(i)] -= mean_x;
      input.vy[static_cast<size_t>(i)] -= mean_y;
      input.vz[static_cast<size_t>(i)] -= mean_z;
    }
    return;
  }
  double momentum_x = 0.0;
  double momentum_y = 0.0;
  double momentum_z = 0.0;
  double total_mass = 0.0;
  for (int i = 0; i < n; ++i) {
    const double mass = AtomMass(input, i);
    momentum_x += mass * input.vx[static_cast<size_t>(i)];
    momentum_y += mass * input.vy[static_cast<size_t>(i)];
    momentum_z += mass * input.vz[static_cast<size_t>(i)];
    total_mass += mass;
  }
  const double shift_x = momentum_x / total_mass;
  const double shift_y = momentum_y / total_mass;
  const double shift_z = momentum_z / total_mass;
  for (int i = 0; i < n; ++i) {
    input.vx[static_cast<size_t>(i)] -= shift_x;
    input.vy[static_cast<size_t>(i)] -= shift_y;
    input.vz[static_cast<size_t>(i)] -= shift_z;
  }
}

}  // namespace

SimulationInput ParseInput(const std::string& text, const std::string& base_dir) {
  const std::vector<Command> commands = Lex(text);
  const char* known[] = {"units",        "box",      "structure",    "types",      "lj",
                         "lj_shift",     "eam",      "mass",         "velocity",   "skin",
                         "timestep",     "ensemble", "thermo_every", "dump_every", "dump_virial",
                         "graph_unroll", "run"};
  for (const Command& command : commands) {
    bool match = false;
    for (const char* key : known) {
      if (command.key == key) match = true;
    }
    if (!match) throw std::runtime_error("unknown input key " + command.key);
  }

  SimulationInput input;
  bool metal_units = false;
  if (const Command* units = Find(commands, "units")) {
    RequireCount(*units, 1);
    if (units->args[0] == "lj")
      metal_units = false;
    else if (units->args[0] == "metal")
      metal_units = true;
    else
      throw std::runtime_error("units must be lj or metal");
  }
  if (const Command* eam = Find(commands, "eam")) {
    RequireCount(*eam, 1);
    input.eam_path = (std::filesystem::path(base_dir) / eam->args[0]).string();
    input.metal = true;
  }
  if (input.metal && !metal_units) throw std::runtime_error("eam requires units metal");
  if (metal_units && !input.metal) throw std::runtime_error("units metal needs an eam file");

  const Command* box = Find(commands, "box");
  const Command* structure = Find(commands, "structure");
  if ((box == nullptr) == (structure == nullptr)) {
    throw std::runtime_error("input needs one of box or structure");
  }
  std::vector<double> file_vx;
  std::vector<double> file_vy;
  std::vector<double> file_vz;
  if (box != nullptr) {
    RequireCount(*box, 5);
    if (box->args[0] != "fcc") throw std::runtime_error("stage 1 box is fcc");
    FillFcc(input.system, Integer(*box, 1), Integer(*box, 2), Integer(*box, 3), Number(*box, 4));
  } else {
    RequireCount(*structure, 1);
    const std::filesystem::path file = std::filesystem::path(base_dir) / structure->args[0];
    ref::ExtxyzFrame loaded = ref::ReadExtxyzFrame(file.string());
    input.system.box[0] = loaded.system.box[0];
    input.system.box[1] = loaded.system.box[1];
    input.system.box[2] = loaded.system.box[2];
    input.system.type = std::move(loaded.system.type);
    input.system.x = std::move(loaded.system.x);
    input.system.y = std::move(loaded.system.y);
    input.system.z = std::move(loaded.system.z);
    file_vx = std::move(loaded.vx);
    file_vy = std::move(loaded.vy);
    file_vz = std::move(loaded.vz);
  }

  if (const Command* types = Find(commands, "types")) {
    RequireCount(*types, 1);
    const int count = Integer(*types, 0);
    if (count < 1) throw std::runtime_error("type count must be positive");
    for (int type : input.system.type) {
      if (type < 0 || type >= count)
        throw std::runtime_error("atom type is outside the declared range");
    }
  }

  input.system.pairs.clear();
  for (const Command& command : commands) {
    if (command.key != "lj") continue;
    RequireCount(command, 8);
    if (command.args[2] != "epsilon" || command.args[4] != "sigma" || command.args[6] != "cutoff") {
      throw std::runtime_error("lj line is missing epsilon, sigma, or cutoff");
    }
    ref::PairCoeff coeff;
    coeff.type_i = Integer(command, 0);
    coeff.type_j = Integer(command, 1);
    coeff.epsilon = Number(command, 3);
    coeff.sigma = Number(command, 5);
    coeff.cutoff = Number(command, 7);
    input.system.pairs.push_back(coeff);
  }
  if (input.metal && !input.system.pairs.empty()) {
    throw std::runtime_error("eam input does not take an lj line");
  }
  if (!input.metal && input.system.pairs.empty())
    throw std::runtime_error("input needs an lj line");

  if (const Command* shift = Find(commands, "lj_shift")) {
    if (input.metal) throw std::runtime_error("lj_shift does not apply to eam");
    RequireCount(*shift, 1);
    if (shift->args[0] == "none")
      input.system.shift = ref::Shift::None;
    else if (shift->args[0] == "energy")
      input.system.shift = ref::Shift::Energy;
    else if (shift->args[0] == "force")
      input.system.shift = ref::Shift::Force;
    else
      throw std::runtime_error("lj_shift must be none, energy, or force");
  }

  bool mass_set = false;
  if (const Command* mass = Find(commands, "mass")) {
    RequireCount(*mass, 2);
    if (Integer(*mass, 0) != 0) throw std::runtime_error("stage 1 has one mass, on type 0");
    input.mass = Number(*mass, 1);
    if (!(input.mass > 0.0)) throw std::runtime_error("mass must be positive");
    mass_set = true;
  }
  if (input.metal && LooksLikeSetfl(input.eam_path)) {
    if (mass_set) throw std::runtime_error("setfl masses come from the table");
    const EamSetfl table(input.eam_path);
    input.setfl = true;
    input.mass_by_type.resize(static_cast<size_t>(table.NumElements()));
    for (int element = 0; element < table.NumElements(); ++element) {
      input.mass_by_type[static_cast<size_t>(element)] = table.Mass(element);
    }
    input.mass = input.mass_by_type[0];
    for (int type : input.system.type) {
      if (type < 0 || type >= table.NumElements()) {
        throw std::runtime_error("atom type is outside the setfl elements");
      }
    }
  } else if (input.metal && !mass_set) {
    input.mass = EamFuncfl(input.eam_path).Mass();
  }

  const int n = ref::NumAtoms(input.system);
  input.vx.assign(static_cast<size_t>(n), 0.0);
  input.vy.assign(static_cast<size_t>(n), 0.0);
  input.vz.assign(static_cast<size_t>(n), 0.0);
  if (const Command* velocity = Find(commands, "velocity")) {
    RequireCount(*velocity, 3);
    if (velocity->args[1] != "seed") throw std::runtime_error("velocity line needs a seed");
    FillVelocities(input, Number(*velocity, 0), static_cast<std::uint32_t>(Integer(*velocity, 2)));
  } else if (!file_vx.empty()) {
    if (static_cast<int>(file_vx.size()) != n) {
      throw std::runtime_error("structure velocity count does not match the atoms");
    }
    input.vx = std::move(file_vx);
    input.vy = std::move(file_vy);
    input.vz = std::move(file_vz);
  }

  if (const Command* skin = Find(commands, "skin")) {
    RequireCount(*skin, 1);
    input.skin = Number(*skin, 0);
    if (!(input.skin >= 0.0)) throw std::runtime_error("skin must be non-negative");
  }
  if (const Command* timestep = Find(commands, "timestep")) {
    RequireCount(*timestep, 1);
    input.timestep = Number(*timestep, 0);
    if (!(input.timestep > 0.0)) throw std::runtime_error("timestep must be positive");
  }
  if (const Command* ensemble = Find(commands, "ensemble")) {
    if (ensemble->args.empty()) throw std::runtime_error("ensemble needs a name");
    if (ensemble->args[0] == "nve") {
      RequireCount(*ensemble, 1);
      input.langevin = false;
    } else if (ensemble->args[0] == "langevin") {
      RequireCount(*ensemble, 4);
      input.langevin = true;
      input.thermo.temperature = Number(*ensemble, 1);
      input.thermo.gamma = Number(*ensemble, 2);
      input.thermo.seed = static_cast<std::uint32_t>(Integer(*ensemble, 3));
    } else {
      throw std::runtime_error("ensemble must be nve or langevin");
    }
  }
  if (const Command* thermo = Find(commands, "thermo_every")) {
    RequireCount(*thermo, 1);
    input.thermo_every = Integer(*thermo, 0);
    if (input.thermo_every < 0) throw std::runtime_error("thermo_every must be non-negative");
  }
  if (const Command* dump = Find(commands, "dump_every")) {
    RequireCount(*dump, 1);
    input.dump_every = Integer(*dump, 0);
    if (input.dump_every < 0) throw std::runtime_error("dump_every must be non-negative");
  }
  if (const Command* virial = Find(commands, "dump_virial")) {
    RequireCount(*virial, 1);
    input.dump_virial = Integer(*virial, 0);
    if (input.dump_virial != 0 && input.dump_virial != 1) {
      throw std::runtime_error("dump_virial must be 0 or 1");
    }
  }
  if (const Command* graph = Find(commands, "graph_unroll")) {
    RequireCount(*graph, 1);
    input.graph_unroll = Integer(*graph, 0);
    if (input.graph_unroll < 1) throw std::runtime_error("graph_unroll must be positive");
  }
  if (const Command* run = Find(commands, "run")) {
    RequireCount(*run, 1);
    input.steps = Integer(*run, 0);
    if (input.steps < 0) throw std::runtime_error("run length must be non-negative");
  }
  return input;
}

SimulationInput ParseInputFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream body;
  body << in.rdbuf();
  const std::filesystem::path file(path);
  const std::string base = file.has_parent_path() ? file.parent_path().string() : ".";
  return ParseInput(body.str(), base);
}

}  // namespace tilemd
