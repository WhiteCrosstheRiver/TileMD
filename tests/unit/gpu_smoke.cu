#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

__global__ void TouchKernel() {}

}  // namespace

TEST(GpuSmoke, LaunchesOrSkips) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  TouchKernel<<<1, 1>>>();
  err = cudaDeviceSynchronize();
  if (err == cudaErrorNoKernelImageForDevice || err == cudaErrorInvalidDeviceFunction) {
    cudaGetLastError();
    GTEST_SKIP() << "No kernel image for the visible GPU. Rebuild with -DTILEMD_ARCHS=<cc>.";
  }
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}
