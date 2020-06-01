//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/scalar_add.hpp"

template <typename T>
__global__ void scalar_add_kernel(T* x, size_t n, size_t s, const T beta) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        x[s * index] += beta;
    }
}

template <typename T>
__global__ void scalar_add_kernel_flat(T* x, size_t n, const T beta) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        x[index] += beta;
    }
}

template <>
__global__ void scalar_add_kernel(cuDoubleComplex* x, size_t n, size_t s, const cuDoubleComplex beta) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        x[s * index] = cuCadd(x[s * index], beta);
    }
}

template <>
__global__ void scalar_add_kernel_flat(cuDoubleComplex* x, size_t n, const cuDoubleComplex beta) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        x[index] = cuCadd(x[index], beta);
    }
}

template <>
__global__ void scalar_add_kernel(cuComplex* x, size_t n, size_t s, const cuComplex beta) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        x[s * index] = cuCaddf(x[s * index], beta);
    }
}

template <>
__global__ void scalar_add_kernel_flat(cuComplex* x, size_t n, const cuComplex beta) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        x[index] = cuCaddf(x[index], beta);
    }
}

template <typename T>
void scalar_add_kernel_run(T* x, size_t n, size_t s, T beta) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_add_kernel<T>, 0, 0);
    }

    int gridSize = (n + blockSize - 1) / blockSize;

    if (s == 1) {
        scalar_add_kernel_flat<T><<<gridSize, blockSize>>>(x, n, beta);
    } else {
        scalar_add_kernel<T><<<gridSize, blockSize>>>(x, n, s, beta);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_scalar_sadd(float* x, size_t n, size_t s, float beta) {
    scalar_add_kernel_run(x, n, s, beta);
}

void egblas_scalar_dadd(double* x, size_t n, size_t s, double beta) {
    scalar_add_kernel_run(x, n, s, beta);
}

void egblas_scalar_cadd(cuComplex* x, size_t n, size_t s, cuComplex beta) {
    scalar_add_kernel_run(x, n, s, beta);
}

void egblas_scalar_zadd(cuDoubleComplex* x, size_t n, size_t s, cuDoubleComplex beta) {
    scalar_add_kernel_run(x, n, s, beta);
}
