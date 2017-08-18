//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/scalar_add.hpp"

template <typename T>
__global__ void scalar_div_kernel(const T beta, T* x, size_t n, size_t s) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] = beta / x[s * index];
    }
}

template <>
__global__ void scalar_div_kernel(const cuComplex beta, cuComplex* x, size_t n, size_t s) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] = cuCdivf(beta, x[s * index]);
    }
}

template <>
__global__ void scalar_div_kernel(const cuDoubleComplex beta, cuDoubleComplex* x, size_t n, size_t s) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] = cuCdiv(beta, x[s * index]);
    }
}

template <typename T>
void scalar_div_kernel_run(T beta, T* x, size_t n, size_t s) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_div_kernel<T>, 0, 0);

    int gridSize = ((n / s) + blockSize - 1) / blockSize;

    scalar_div_kernel<T><<<gridSize, blockSize>>>(beta, x, n, s);

    cudaDeviceSynchronize();
}

void egblas_scalar_sdiv(float beta, float* x, size_t n, size_t s) {
    scalar_div_kernel_run(beta, x, n, s);
}

void egblas_scalar_ddiv(double beta, double* x, size_t n, size_t s) {
    scalar_div_kernel_run(beta, x, n, s);
}

void egblas_scalar_cdiv(cuComplex beta, cuComplex* x, size_t n, size_t s) {
    scalar_div_kernel_run(beta, x, n, s);
}

void egblas_scalar_zdiv(cuDoubleComplex beta, cuDoubleComplex* x, size_t n, size_t s) {
    scalar_div_kernel_run(beta, x, n, s);
}
