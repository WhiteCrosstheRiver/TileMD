#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/device_status.hpp"
#include "core/version.hpp"
#include "md/emergency.hpp"

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

TEST(CpuSmoke, NonFiniteAbortNamesTheAtomAndWritesTheFrame) {
  EXPECT_EQ(tilemd::NonFiniteMessage(12, 7),
            "non-finite force or position during integration at step 12 atom 7");
  const double box[3] = {2.0, 3.0, 4.0};
  const double x[2] = {0.1, 0.2};
  const double y[2] = {1.0, 1.5};
  const double z[2] = {2.0, 2.5};
  const double vx[2] = {0.01, -0.02};
  const double vy[2] = {0.0, 0.3};
  const double vz[2] = {-0.4, 0.5};
  const int type[2] = {1, 0};
  const int gid[2] = {1, 0};
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "tilemd-emergency-test.extxyz";
  tilemd::WriteEmergencyFrame(path.string(), box, x, y, z, vx, vy, vz, type, gid, 2, 12);
  std::ifstream in(path);
  std::string header;
  std::string lattice;
  std::string line0;
  std::string line1;
  std::getline(in, header);
  std::getline(in, lattice);
  std::getline(in, line0);
  std::getline(in, line1);
  in.close();
  std::filesystem::remove(path);
  EXPECT_EQ(header, "2");
  EXPECT_NE(lattice.find("step=12"), std::string::npos);
  EXPECT_EQ(line0.front(), 'A');
  EXPECT_EQ(line1.front(), 'B');
  EXPECT_EQ(line0.substr(line0.rfind(' ') + 1), "0");
  EXPECT_EQ(line1.substr(line1.rfind(' ') + 1), "1");
}
