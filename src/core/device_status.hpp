#pragma once

#include <cstddef>
#include <cstdint>

namespace tilemd {

// Single device-side status word. The host reads it once per segment, never
// inside the time-step loop (PLAN.md section 5.2).
struct DeviceStatus {
  int need_rebuild;
  int overflow;
  int nan_detected;
  int rebuild_count;
  // Non-negative displacement stored as IEEE bits so a kernel can atomicMax
  // the field as an int. Ordered for values >= 0.
  float max_disp2_bits;
  // Device slot of the first non-finite atom. Meaningful when nan_detected is set.
  int nan_slot;
};

static_assert(sizeof(int) == 4, "max_disp2 bit trick assumes a 32-bit int");
static_assert(sizeof(float) == 4, "max_disp2 bit trick assumes IEEE-754 binary32");
static_assert(offsetof(DeviceStatus, max_disp2_bits) == 4 * sizeof(int),
              "max_disp2_bits must follow the four int counters with no padding");

}  // namespace tilemd
