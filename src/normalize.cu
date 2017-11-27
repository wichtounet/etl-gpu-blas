//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <random>

#include "egblas/mean.hpp"
#include "egblas/stddev.hpp"
#include "egblas/shuffle.hpp"
#include "egblas/cuda_check.hpp"

template <typename T>
__global__ void normalize_flat_kernel1(size_t n, T* x, size_t incx, T mean) {
    auto index        = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        x[incx * index] = x[incx * index] - mean;
    }
}

template <typename T>
__global__ void normalize_flat_kernel2(size_t n, T* x, size_t incx, T s) {
    auto index        = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        x[incx * index] = x[incx * index] / s;
    }
}

template <typename T>
void normalize_flat_kernel1_run(size_t n, T* x, size_t incx, T mean) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, normalize_flat_kernel1<T>, 0, 0);
    }

    int gridSize = ((n / incx) + blockSize - 1) / blockSize;
    normalize_flat_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, mean);

    cudaDeviceSynchronize();
}

template <typename T>
void normalize_flat_kernel2_run(size_t n, T* x, size_t incx, T s) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, normalize_flat_kernel2<T>, 0, 0);
    }

    int gridSize = ((n / incx) + blockSize - 1) / blockSize;
    normalize_flat_kernel2<T><<<gridSize, blockSize>>>(n, x, incx, s);

    cudaDeviceSynchronize();
}

void egblas_snormalize_flat(size_t n, float* x, size_t incx) {
    auto m = egblas_smean(x, n, incx);
    normalize_flat_kernel1_run(n, x, incx, m);

    auto s = egblas_sstddev_mean(x, n, incx, 0.0f);
    normalize_flat_kernel2_run(n, x, incx, s);
}

void egblas_dnormalize_flat(size_t n, double* x, size_t incx) {
    auto m = egblas_dmean(x, n, incx);
    normalize_flat_kernel1_run(n, x, incx, m);

    auto s = egblas_dstddev_mean(x, n, incx, 0.0);
    normalize_flat_kernel2_run(n, x, incx, s);
}

void egblas_snormalize_sub(size_t n, float* x, size_t sub_n, size_t incx) {
    for(size_t i = 0; i < n; ++i){
        egblas_snormalize_flat(sub_n, x + i * sub_n * incx, incx);
    }
}

void egblas_dnormalize_sub(size_t n, double* x, size_t sub_n, size_t incx) {
    for(size_t i = 0; i < n; ++i){
        egblas_dnormalize_flat(sub_n, x + i * sub_n * incx, incx);
    }
}
