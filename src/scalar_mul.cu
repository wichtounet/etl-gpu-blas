//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/scalar_mul.hpp"

template <typename T>
__global__ void scalar_mul_kernel(T* x, size_t n, size_t s, const T beta) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] *= beta;
    }
}

template <>
__global__ void scalar_mul_kernel(cuDoubleComplex* x, size_t n, size_t s, const cuDoubleComplex beta) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] = cuCmul(x[s * index], beta);
    }
}

template <>
__global__ void scalar_mul_kernel(cuComplex* x, size_t n, size_t s, const cuComplex beta) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] = cuCmulf(x[s * index], beta);
    }
}

template <typename T>
void scalar_mul_kernel_run(T* x, size_t n, size_t s, T beta) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_mul_kernel<T>, 0, 0);

    int gridSize = ((n / s) + blockSize - 1) / blockSize;

    scalar_mul_kernel<T><<<gridSize, blockSize>>>(x, n, s, beta);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_scalar_smul(float* x, size_t n, size_t s, float beta) {
    scalar_mul_kernel_run(x, n, s, beta);
}

void egblas_scalar_dmul(double* x, size_t n, size_t s, double beta) {
    scalar_mul_kernel_run(x, n, s, beta);
}

void egblas_scalar_cmul(cuComplex* x, size_t n, size_t s, cuComplex beta) {
    scalar_mul_kernel_run(x, n, s, beta);
}

void egblas_scalar_zmul(cuDoubleComplex* x, size_t n, size_t s, cuDoubleComplex beta) {
    scalar_mul_kernel_run(x, n, s, beta);
}
