//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/one_if_max.hpp"
#include "egblas/max_reduce.hpp"

template <typename T>
__global__ void one_if_max_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy, T max) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = x[incx * index] == max ? alpha : T(0);
    }
}

template <typename T>
__global__ void one_if_max_kernel_flat(size_t n, T alpha, const T* x, T* y, T max) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = x[index] == max ? alpha : T(0);
    }
}

template <typename T>
void one_if_max_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy, T max) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, one_if_max_kernel<T>, 0, 0);
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        one_if_max_kernel_flat<T><<<gridSize, blockSize>>>(n, alpha, x, y, max);
    } else {
        one_if_max_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy, max);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sone_if_max(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    auto max = egblas_smax(x, n, incx);
    one_if_max_kernel_run(n, alpha, x, incx, y, incy, max);
}

void egblas_done_if_max(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    auto max = egblas_dmax(x, n, incx);
    one_if_max_kernel_run(n, alpha, x, incx, y, incy, max);
}
