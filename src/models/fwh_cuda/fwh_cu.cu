// The codes are from Armen Aghajanyan from facebook, from paper
// Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
// https://arxiv.org/abs/2012.13255

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void FastWalshHadamardKernel(const int stride, c