//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/assert.hpp"
#include "egblas/utils.hpp"
#include "egblas/batch_k_scale.hpp"
#include "egblas/cuda_check.hpp"

// 2D Version

template <typename T>
__global__ void batch_k_scale_plus2_kernel(size_t B, size_t K, const T* x, const T* gamma, const T* beta, T* y) {
    auto bk  = threadIdx.x + blockIdx.x * blockDim.x;

    if (bk < B * K) {
        const size_t b = bk / K;
        const size_t k = bk % K;

        y[b * K + k] = gamma[k] * x[b * K + k] + beta[k];
    }
}

template <typename T>
void egblas_batch_k_scale_plus2_run(size_t b, size_t k, const T* x, const T* gamma, const T* beta, T* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, batch_k_scale_plus2_kernel<T>, 0, 0);

    int gridSize = ((b * k) + blockSize - 1) / blockSize;

    batch_k_scale_plus2_kernel<<<gridSize, blockSize>>>(b, k, x, gamma, beta, y);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sbatch_k_scale_plus2(size_t b, size_t k, const float* x, const float* gamma, const float* beta, float* y) {
    egblas_batch_k_scale_plus2_run(b, k, x, gamma, beta, y);
}

void egblas_dbatch_k_scale_plus2(size_t b, size_t k, const double* x, const double* gamma, const double* beta, double* y) {
    egblas_batch_k_scale_plus2_run(b, k, x, gamma, beta, y);
}

// 4D version

template <typename T>
__global__ void batch_k_scale_plus4_kernel(size_t B, size_t K, size_t M, size_t N, const T* x, const T* gamma, const T* beta, T* y) {
    auto bkmn  = threadIdx.x + blockIdx.x * blockDim.x;

    if (bkmn < B * K * M * N) {
        const size_t k = (bkmn / (M * N)) % K;

        y[bkmn] = gamma[k] * x[bkmn] + beta[k];
    }
}

template <typename T>
void egblas_batch_k_scale_plus4_run(size_t b, size_t k, size_t m, size_t n, const T* x, const T* gamma, const T* beta, T* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, batch_k_scale_plus4_kernel<T>, 0, 0);

    int gridSize = ((b * k * m * n) + blockSize - 1) / blockSize;

    batch_k_scale_plus4_kernel<<<gridSize, blockSize>>>(b, k, m, n, x, gamma, beta, y);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sbatch_k_scale_plus4(size_t b, size_t k, size_t m, size_t n, const float* x, const float* gamma, const float* beta, float* y) {
    egblas_batch_k_scale_plus4_run(b, k, m, n, x, gamma, beta, y);
}

void egblas_dbatch_k_scale_plus4(size_t b, size_t k, size_t m, size_t n, const double* x, const double * gamma, const double* beta, double* y) {
    egblas_batch_k_scale_plus4_run(b, k, m, n, x, gamma, beta, y);
}
