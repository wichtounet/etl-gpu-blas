//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "cpp_utils/assert.hpp"

template<typename T>
__global__ void scalar_add_kernel(T * x, size_t n, const T beta){
    auto index = threadIdx.x + blockIdx.x*blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for(; index < n; index += stride) {
        x[index] += beta;
    }
}

template<typename T>
void scalar_add_kernel_run(T * x, size_t n, size_t s, T beta){
    cpp_unused(s);

    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_add_kernel<T>, 0, 0);

    int gridSize = (n + blockSize - 1) / blockSize;

    scalar_add_kernel<T><<<gridSize, blockSize>>>(x, n, beta);

    cudaDeviceSynchronize();
}

void egblas_scalar_sadd(float* x, size_t n, size_t s, float beta){
    cpp_assert(s == 1, "Stride is not yet supported for egblas_scalar_sadd");
    cpp_unused(s);

    scalar_add_kernel_run(x, n, s, beta);
}

void egblas_scalar_dadd(double* x, size_t n, size_t s, double beta){
    cpp_assert(s == 1, "Stride is not yet supported for egblas_scalar_dadd");
    cpp_unused(s);

    scalar_add_kernel_run(x, n, s, beta);
}
