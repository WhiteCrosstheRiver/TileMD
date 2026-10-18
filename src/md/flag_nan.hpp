#pragma once

#include "core/device_status.hpp"

#if defined(__CUDACC__)
namespace tilemd {

static __device__ __forceinline__ void FlagNonFinite(DeviceStatus* status, int slot) {
  if (status == nullptr) return;
  if (atomicCAS(&status->nan_detected, 0, 1) == 0) status->nan_slot = slot;
}

}  // namespace tilemd
#endif
