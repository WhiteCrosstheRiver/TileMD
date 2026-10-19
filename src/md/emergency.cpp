#include "md/emergency.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemd {

std::string NonFiniteMessage(int step, int gid) {
  return "non-finite force or position during integration at step " + std::to_string(step) +
         " atom " + std::to_string(gid);
}

void WriteEmergencyFrame(const std::string& path, const double* box, const double* x,
                         const double* y, const double* z, const double* vx, const double* vy,
                         const double* vz, const int* type, const int* gid, int n, int step,
                         bool append, const char* shift, const double* wxx, const double* wyy,
                         const double* wzz, const double* wxy, const double* wxz,
                         const double* wyz) {
  const bool virial = wxx != nullptr;
  if (virial &&
      (wyy == nullptr || wzz == nullptr || wxy == nullptr || wxz == nullptr || wyz == nullptr)) {
    throw std::runtime_error("per-atom virial needs six components");
  }
  std::ofstream out(path, append ? std::ios::app : std::ios::out);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.setf(std::ios::scientific);
  out.precision(17);
  const char* properties = virial ? "species:S:1:pos:R:3:vel:R:3:type:I:1:virial:R:6"
                                  : "species:S:1:pos:R:3:vel:R:3:type:I:1";
  out << n << "\n";
  out << "Lattice=\"" << box[0] << " 0 0 0 " << box[1] << " 0 0 0 " << box[2]
      << "\" Properties=" << properties << " step=" << step
      << " shift=" << (shift == nullptr ? "none" : shift) << " pbc=\"T T T\"\n";
  std::vector<int> slot_of(static_cast<size_t>(n), -1);
  for (int i = 0; i < n; ++i) {
    const int original = gid[i];
    if (original >= 0 && original < n) slot_of[static_cast<size_t>(original)] = i;
  }
  for (int original = 0; original < n; ++original) {
    const int slot = slot_of[static_cast<size_t>(original)];
    const int i = slot >= 0 ? slot : original;
    const int atom_type = type[i];
    if (atom_type < 0 || atom_type > 25) {
      throw std::runtime_error("type does not map to a species letter");
    }
    const char species = static_cast<char>('A' + atom_type);
    out << species << " " << x[i] << " " << y[i] << " " << z[i] << " " << vx[i] << " " << vy[i]
        << " " << vz[i] << " " << atom_type;
    if (virial) {
      out << " " << wxx[i] << " " << wyy[i] << " " << wzz[i] << " " << wxy[i] << " " << wxz[i]
          << " " << wyz[i];
    }
    out << "\n";
  }
}

}  // namespace tilemd
