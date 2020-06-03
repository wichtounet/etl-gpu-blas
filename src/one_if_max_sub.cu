//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/assert.hpp"
#include "egblas/utils.hpp"
#include "egblas/sum.hpp"
#include "egblas/cuda_check.hpp"

template <typename T>
__global__ void one_if_max_sub_kernel(size_t B, size_t N, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto b  = threadIdx.x + blockIdx.x * blockDim.x;

    if (b < B) {
        T max = x[(b * N + 0) * incx];

        for (size_t n = 1; n < N; ++n) {
            max = x[(b * N + n) * incx] > max ? x[(b * N + n) * incx] : max;
        }

        for (size_t n = 0; n < N; ++n) {
            y[(b * N + n) * incx] = x[(b * N + n) * incx] == max ? alpha : T(0);
        }
    }
}

template <typename T>
__global__ void one_if_max_sub_kernel_flat(size_t B, size_t N, T alpha, const T* x, T* y) {
    auto b  = threadIdx.x + blockIdx.x * blockDim.x;

    if (b < B) {
        T max = x[b * N + 0];

        for (size_t n = 1; n < N; ++n) {
            max = x[b * N + n] > max ? x[b * N + n] : max;
        }

        for (size_t n = 0; n < N; ++n) {
            y[b * N + n] = x[b * N + n] == max ? alpha : T(0);
        }
    }
}

void egblas_sone_if_max_sub(size_t b, size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (b + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        one_if_max_sub_kernel_flat<<<gridSize, blockSize>>>(b, n, alpha, x, y);
    } else {
        one_if_max_sub_kernel<<<gridSize, blockSize>>>(b, n, alpha, x, incx, y, incy);
    }
}

void egblas_done_if_max_sub(size_t b, size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (b + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        one_if_max_sub_kernel_flat<<<gridSize, blockSize>>>(b, n, alpha, x, y);
    } else {
        one_if_max_sub_kernel<<<gridSize, blockSize>>>(b, n, alpha, x, incx, y, incy);
    }
}
