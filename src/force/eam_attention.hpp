#pragma once

#include "force/eam.hpp"
#include "force/pair_attention.hpp"

namespace tilemd {

// Two-pass EAM on the device lane map. Pass 0 writes the density, the embedding
// update runs before pass 1, and pass 1 writes the force. This launches kernels.
AttentionResult RunEamFuncflGpu(const EamFuncfl& potential, const double* box, const double* x,
                                const double* y, const double* z, const int* type, int n,
                                double skin);
AttentionResult RunEamSetflGpu(const EamSetfl& potential, const double* box, const double* x,
                               const double* y, const double* z, const int* type, int n,
                               double skin);

}  // namespace tilemd
