#pragma once

#include <string>

namespace tilemd {

// PLAN.md section 6.2: the abort names the step and the original atom id.
std::string NonFiniteMessage(int step, int gid);

// One extended-XYZ frame in original-gid order. Species is 'A' + type.
// Virial pointers use the same index as x, y, z. All six are required together.
// A null wxx leaves the virial column out.
void WriteEmergencyFrame(const std::string& path, const double* box, const double* x,
                         const double* y, const double* z, const double* vx, const double* vy,
                         const double* vz, const int* type, const int* gid, int n, int step,
                         bool append = false, const char* shift = "none",
                         const double* wxx = nullptr, const double* wyy = nullptr,
                         const double* wzz = nullptr, const double* wxy = nullptr,
                         const double* wxz = nullptr, const double* wyz = nullptr);

}  // namespace tilemd
