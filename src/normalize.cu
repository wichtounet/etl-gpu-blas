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
__global__ void normalize_sub_kernel(size_t n, size_t sub_n, T* x, size_t incx) {
    auto index        = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        // 1. compute the mean

        T m = 0;

        for(size_t i = 0; i < sub_n; ++i){
            m += x[(index * sub_n + i) * incx];
        }

        m /= T(sub_n);

        // 2. Compute the standard deviation

        T s = 0;

        for (size_t i = 0; i < sub_n; ++i) {
            s += (x[(index * sub_n + i) * incx] - m) * (x[(index * sub_n + i) * incx] - m);
        }

        s = sqrtf(s / T(sub_n));

        // 2. Normalize

        for (size_t i = 0; i < sub_n; ++i) {
            x[(index * sub_n + i) * incx] = (x[(index * sub_n + i) * incx] - m) / s;
        }
    }
}

template <typename T>
__global__ void normalize_sub_kernel1(size_t n, size_t sub_n, T* x) {
    auto index        = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        // 1. compute the mean

        T m = 0;

        for(size_t i = 0; i < sub_n; ++i){
            m += x[index * sub_n + i];
        }

        m /= T(sub_n);

        // 2. Compute the standard deviation

        T s = 0;

        for (size_t i = 0; i < sub_n; ++i) {
            s += (x[index * sub_n + i] - m) * (x[index * sub_n + i] - m);
        }

        s = sqrtf(s / T(sub_n));

        // 2. Normalize

        for (size_t i = 0; i < sub_n; ++i) {
            x[index * sub_n + i] = (x[index * sub_n + i] - m) / s;
        }
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

template <typename T>
void normalize_sub_kernel_run(size_t n, size_t sub_n, T* x, size_t incx) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, normalize_sub_kernel<T>, 0, 0);
    }

    int gridSize = ((n / incx) + blockSize - 1) / blockSize;
    normalize_sub_kernel<T><<<gridSize, blockSize>>>(n, sub_n, x, incx);

    cudaDeviceSynchronize();
}

template <typename T>
void normalize_sub_kernel1_run(size_t n, size_t sub_n, T* x) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, normalize_sub_kernel1<T>, 0, 0);
    }

    int gridSize = (n + blockSize - 1) / blockSize;
    normalize_sub_kernel1<T><<<gridSize, blockSize>>>(n, sub_n, x);

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
    if(incx == 1){
        normalize_sub_kernel1_run(n, sub_n, x);
    } else {
        normalize_sub_kernel_run(n, sub_n, x, incx);
    }
}

void egblas_dnormalize_sub(size_t n, double* x, size_t sub_n, size_t incx) {
    if(incx == 1){
        normalize_sub_kernel1_run(n, sub_n, x);
    } else {
        normalize_sub_kernel_run(n, sub_n, x, incx);
    }
}
