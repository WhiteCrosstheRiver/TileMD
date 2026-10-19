#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tilemd {

// Column-major neighbor list. `nl[k * n + i]` is neighbor k of slot i.
struct CellList {
  int max_neigh = 1;
  std::vector<int> nn;
  std::vector<int> nl;
};

// Pairs closer than `reach`, including the minimum image. The cell edge is at
// least `reach`, and a wrapped stencil is searched. Each row is sorted by slot.
inline CellList BuildCellList(const double* box, const double* x, const double* y, const double* z,
                              int n, double reach) {
  if (n <= 0) throw std::runtime_error("cell list needs at least one atom");
  if (!(reach > 0.0)) throw std::runtime_error("neighbor reach must be positive");
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) throw std::runtime_error("box lengths must be positive");
  }

  int ncell[3];
  double edge[3];
  int span[3];
  for (int axis = 0; axis < 3; ++axis) {
    ncell[axis] = std::max(1, static_cast<int>(std::floor(box[axis] / reach)));
    edge[axis] = box[axis] / static_cast<double>(ncell[axis]);
    span[axis] = std::max(1, static_cast<int>(std::ceil(reach / edge[axis] - 1.0e-12)));
  }

  auto cell_of = [&](double coordinate, int axis) {
    int cell = static_cast<int>(std::floor(coordinate / edge[axis]));
    if (cell < 0) cell = 0;
    if (cell >= ncell[axis]) cell = ncell[axis] - 1;
    return cell;
  };
  auto flat = [&](int cx, int cy, int cz) { return (cz * ncell[1] + cy) * ncell[0] + cx; };

  const int nflat = ncell[0] * ncell[1] * ncell[2];
  std::vector<int> cursor(nflat, 0);
  std::vector<int> slot_cell(n);
  for (int i = 0; i < n; ++i) {
    const int cx = cell_of(x[i], 0);
    const int cy = cell_of(y[i], 1);
    const int cz = cell_of(z[i], 2);
    slot_cell[i] = flat(cx, cy, cz);
    ++cursor[slot_cell[i]];
  }
  std::vector<int> start(nflat + 1, 0);
  for (int cell = 0; cell < nflat; ++cell) start[cell + 1] = start[cell] + cursor[cell];
  std::vector<int> atoms(n);
  std::fill(cursor.begin(), cursor.end(), 0);
  for (int i = 0; i < n; ++i) {
    const int cell = slot_cell[i];
    atoms[start[cell] + cursor[cell]] = i;
    ++cursor[cell];
  }

  const double reach2 = reach * reach;
  const double* coordinates[3] = {x, y, z};
  std::vector<std::vector<int>> rows(n);
  std::vector<int> seen(n, -1);
  for (int i = 0; i < n; ++i) {
    const int home = slot_cell[i];
    const int hx = home % ncell[0];
    const int hy = (home / ncell[0]) % ncell[1];
    const int hz = home / (ncell[0] * ncell[1]);
    for (int oz = -span[2]; oz <= span[2]; ++oz) {
      for (int oy = -span[1]; oy <= span[1]; ++oy) {
        for (int ox = -span[0]; ox <= span[0]; ++ox) {
          auto wrap = [](int cell, int offset, int count) {
            int wrapped = cell + offset;
            wrapped %= count;
            if (wrapped < 0) wrapped += count;
            return wrapped;
          };
          const int cx = wrap(hx, ox, ncell[0]);
          const int cy = wrap(hy, oy, ncell[1]);
          const int cz = wrap(hz, oz, ncell[2]);
          const int cell = flat(cx, cy, cz);
          for (int cursor_atom = start[cell]; cursor_atom < start[cell + 1]; ++cursor_atom) {
            const int j = atoms[cursor_atom];
            if (j == i || seen[j] == i) continue;
            seen[j] = i;
            double r2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
              double delta = coordinates[axis][i] - coordinates[axis][j];
              delta -= box[axis] * std::rint(delta / box[axis]);
              r2 += delta * delta;
            }
            if (r2 < reach2) rows[i].push_back(j);
          }
        }
      }
    }
    std::sort(rows[i].begin(), rows[i].end());
  }

  CellList out;
  out.nn.assign(n, 0);
  out.max_neigh = 1;
  for (int i = 0; i < n; ++i) {
    out.nn[i] = static_cast<int>(rows[i].size());
    out.max_neigh = std::max(out.max_neigh, out.nn[i]);
  }
  out.nl.assign(static_cast<std::size_t>(out.max_neigh) * n, -1);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < out.nn[i]; ++k) {
      out.nl[static_cast<std::size_t>(k) * n + i] = rows[i][k];
    }
  }
  return out;
}

}  // namespace tilemd
