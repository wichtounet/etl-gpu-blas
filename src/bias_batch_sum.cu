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

#include "sum_reduce.hpp"

template <typename T>
__global__ void bias_batch_sum_kernel(size_t B, size_t N, const T* x, size_t incx, T* y, size_t incy) {
    auto n  = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N) {
        T sum = 0;

        for (size_t b = 0; b < B; ++b) {
            sum += x[(b * N + n) * incx];
        }

        y[incy * n] = sum;
    }
}

template <typename T>
__global__ void bias_batch_sum_kernel_flat(size_t B, size_t N, const T* x, T* y) {
    auto n  = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N) {
        T sum = 0;

        for (size_t b = 0; b < B; ++b) {
            sum += x[b * N + n];
        }

        y[n] = sum;
    }
}

void egblas_sbias_batch_sum(size_t b, size_t n, float* x, size_t incx, float* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        bias_batch_sum_kernel_flat<<<gridSize, blockSize>>>(b, n, x, y);
    } else {
        bias_batch_sum_kernel<<<gridSize, blockSize>>>(b, n, x, incx, y, incy);
    }
}

void egblas_dbias_batch_sum(size_t b, size_t n, double* x, size_t incx, double* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        bias_batch_sum_kernel_flat<<<gridSize, blockSize>>>(b, n, x, y);
    } else {
        bias_batch_sum_kernel<<<gridSize, blockSize>>>(b, n, x, incx, y, incy);
    }
}
