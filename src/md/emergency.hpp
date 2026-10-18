#pragma once

#include <string>

namespace tilemd {

// PLAN.md section 6.2: the abort names the step and the original atom id.
std::string NonFiniteMessage(int step, int gid);

// One extended-XYZ frame in original-gid order. Species is 'A' + type.
void WriteEmergencyFrame(const std::string& path, const double* box, const double* x,
                         const double* y, const double* z, const double* vx, const double* vy,
                         const double* vz, const int* type, const int* gid, int n, int step,
                         bool append = false);

}  // namespace tilemd
