#include "tf32_pair.hpp"

#include <cstring>

namespace tilemd::numerics {
namespace {

uint32_t Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float FromBits(uint32_t bits) {
  float value = 0.f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

float RoundTf32(float value) {
  const uint32_t bits = Bits(value);
  const uint32_t exp = (bits >> 23) & 0xffu;
  if (exp == 0u || exp == 0xffu) return value;
  constexpr int kDiscard = 13;
  const uint32_t mask = (1u << kDiscard) - 1u;
  const uint32_t low = bits & mask;
  const uint32_t half = 1u << (kDiscard - 1);
  uint32_t kept = bits & ~mask;
  const uint32_t lsb = (kept >> kDiscard) & 1u;
  if (low > half || (low == half && lsb != 0u)) kept += 1u << kDiscard;
  return FromBits(kept);
}

void PackScoreA(float x, float y, float z, float out[8]) {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  out[3] = x * x + y * y + z * z;
  out[4] = 1.f;
  out[5] = 0.f;
  out[6] = 0.f;
  out[7] = 0.f;
}

void PackScoreB(float x, float y, float z, float out[8]) {
  out[0] = -2.f * x;
  out[1] = -2.f * y;
  out[2] = -2.f * z;
  out[3] = 1.f;
  out[4] = x * x + y * y + z * z;
  out[5] = 0.f;
  out[6] = 0.f;
  out[7] = 0.f;
}

float DotFp32(const float a[8], const float b[8]) {
  float sum = 0.f;
  for (int k = 0; k < 8; ++k) sum += a[k] * b[k];
  return sum;
}

float DotTf32(const float a[8], const float b[8]) {
  float sum = 0.f;
  for (int k = 0; k < 8; ++k) sum += RoundTf32(a[k]) * RoundTf32(b[k]);
  return sum;
}

float Dot3Tf32(const float a[8], const float b[8]) {
  float sum = 0.f;
  for (int k = 0; k < 8; ++k) {
    const float a_hi = RoundTf32(a[k]);
    const float b_hi = RoundTf32(b[k]);
    const float a_lo = RoundTf32(a[k] - a_hi);
    const float b_lo = RoundTf32(b[k] - b_hi);
    sum += a_hi * b_hi;
    sum += a_hi * b_lo;
    sum += a_lo * b_hi;
  }
  return sum;
}

}  // namespace tilemd::numerics
