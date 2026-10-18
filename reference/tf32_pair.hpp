#pragma once

#include <cstdint>

namespace tilemd::numerics {

// Round a binary32 value to TF32 (8-bit exponent, 10-bit mantissa, nearest even).
float RoundTf32(float value);

// Pack the M7 score vectors. |r|^2 is a binary32 sum of squares.
void PackScoreA(float x, float y, float z, float out[8]);
void PackScoreB(float x, float y, float z, float out[8]);

// Dot products. FP32 is the SIMT product. TF32 rounds both operands once.
// 3×TF32 adds the high/high, high/low, and low/high products, each rounded to TF32.
float DotFp32(const float a[8], const float b[8]);
float DotTf32(const float a[8], const float b[8]);
float Dot3Tf32(const float a[8], const float b[8]);

}  // namespace tilemd::numerics
