#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "core/device_status.hpp"
#include "core/version.hpp"

namespace {

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace

TEST(CpuSmoke, VersionIsNonEmpty) {
  ASSERT_NE(tilemd::kVersion, nullptr);
  EXPECT_STRNE(tilemd::kVersion, "");
}

// PLAN.md section 6.3 updates max displacement with atomicMax on the IEEE bits
// of a non-negative float. That is only valid while the bit pattern is ordered.
TEST(CpuSmoke, NonNegativeFloatBitsCompareAsInts) {
  EXPECT_LT(FloatBits(0.0f), FloatBits(0.25f));
  EXPECT_LT(FloatBits(0.25f), FloatBits(1.0f));
  EXPECT_LT(FloatBits(1.0f), FloatBits(100.0f));

  tilemd::DeviceStatus status{};
  EXPECT_EQ(status.need_rebuild, 0);
  EXPECT_EQ(status.overflow, 0);
  EXPECT_EQ(status.nan_detected, 0);
  EXPECT_EQ(status.rebuild_count, 0);
  EXPECT_EQ(status.max_disp2_bits, 0.0f);
}
