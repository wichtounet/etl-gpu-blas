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
#include "egblas/sum.hpp"
#include "egblas/cuda_check.hpp"

template <typename T>
__global__ void transpose_front_kernel(size_t M, size_t N, size_t K, const T* x, T* y) {
    auto m  = threadIdx.x + blockIdx.x * blockDim.x;

    if (m < M) {
        for (size_t n = 0; n < N; ++n) {
            // y(n, m) = x(m, n) 
            // x[M, N, K]
            // y[N, M, K]

            for (size_t k = 0; k < K; ++k) {
                y[n * (M * K) + m * K + k] = x[m * (N * K) + n * K + k];
            }
        }
    }
}

void egblas_stranspose_front(size_t m, size_t n, size_t k, float* x, float* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &transpose_front_kernel<float>, 0, 0);

    int gridSize = (m + blockSize - 1) / blockSize;

    transpose_front_kernel<<<gridSize, blockSize>>>(m, n, k, x, y);
}

void egblas_stranspose_front(size_t m, size_t n, size_t k, double* x, double* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &transpose_front_kernel<double>, 0, 0);

    int gridSize = (m + blockSize - 1) / blockSize;

    transpose_front_kernel<<<gridSize, blockSize>>>(m, n, k, x, y);
}
