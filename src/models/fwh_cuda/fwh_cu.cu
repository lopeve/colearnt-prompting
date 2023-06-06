// The codes are from Armen Aghajanyan from facebook, from paper
// Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
// https://arxiv.org/abs/2012.13255

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void FastWalshHadamardKernel(const int stride, const scalar_t* in, scalar_t* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    const auto elemIdx = (idx / stride ) * (2 * stride) + (idx % stride);
    const auto tmp = in[elemIdx], tmp2 = in[elemIdx + stride];
    out[elemIdx] = tmp + tmp2;
    out[elemIdx + stride