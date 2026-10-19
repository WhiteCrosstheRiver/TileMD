#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemd {

// One LAMMPS dump frame: id type x y z vx vy vz. Ids are 1-based.
struct DumpFrame {
  double box[3] = {};
  std::vector<int> type;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
};

inline DumpFrame ReadLammpsDump(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open dump: " + path);
  DumpFrame frame;
  std::string line;
  int n = -1;
  while (std::getline(in, line)) {
    if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) {
      in >> n;
      std::getline(in, line);
    } else if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) {
      for (int axis = 0; axis < 3; ++axis) {
        double lo = 0.0;
        double hi = 0.0;
        in >> lo >> hi;
        frame.box[axis] = hi - lo;
        std::getline(in, line);
      }
    } else if (line.rfind("ITEM: ATOMS", 0) == 0) {
      if (n < 1) throw std::runtime_error("dump has no atom count");
      frame.type.assign(n, 0);
      frame.x.assign(n, 0.0);
      frame.y.assign(n, 0.0);
      frame.z.assign(n, 0.0);
      frame.vx.assign(n, 0.0);
      frame.vy.assign(n, 0.0);
      frame.vz.assign(n, 0.0);
      for (int row = 0; row < n; ++row) {
        int id = 0;
        int type = 0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;
        in >> id >> type >> x >> y >> z >> vx >> vy >> vz;
        if (!in) throw std::runtime_error("truncated dump");
        const int slot = id - 1;
        if (slot < 0 || slot >= n) throw std::runtime_error("dump id is outside 1..n");
        frame.type[slot] = type - 1;
        frame.x[slot] = x;
        frame.y[slot] = y;
        frame.z[slot] = z;
        frame.vx[slot] = vx;
        frame.vy[slot] = vy;
        frame.vz[slot] = vz;
      }
      return frame;
    }
  }
  throw std::runtime_error("dump has no atoms");
}

}  // namespace tilemd
