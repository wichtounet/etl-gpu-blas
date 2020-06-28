//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/bias_add.hpp"

template <typename T>
__global__ void bias_add_2d_kernel(size_t m, size_t n, const T* x, size_t incx, const T* b, size_t incb, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < m * n; index += stride) {
        y[incy * index] = x[incx * index] + b[(index % n) * incb];
    }
}

template <typename T>
__global__ void bias_add_2d_kernel_flat(size_t m, size_t n, const T* x, const T* b, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < m * n; index += stride) {
        y[index] = x[index] + b[(index % n)];
    }
}

template <typename T>
__global__ void bias_add_4d_kernel(size_t m, size_t n, size_t o, size_t p, const T* x, size_t incx, const T* b, size_t incb, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < m * n * o * p; index += stride) {
        y[incy * index] = x[incx * index] + b[((index / (o * p)) % n) * incb];
    }
}

template <typename T>
__global__ void bias_add_4d_kernel_flat(size_t m, size_t n, size_t o, size_t p, size_t limit, size_t inner, const T* x, const T* b, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < limit; index += stride) {
        y[index] = x[index] + b[(index / inner) % n];
    }
}

template <typename T>
void bias_add_2d_kernel_run(size_t m, size_t n, const T* x, size_t incx, const T* b, size_t incb, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bias_add_2d_kernel<T>, 0, 0);
    }

    int gridSize = (((m * n) / incy) + blockSize - 1) / blockSize;

    if (incx == 1 && incb == 1 && incy == 1) {
        bias_add_2d_kernel_flat<T><<<gridSize, blockSize>>>(m, n, x, b, y);
    } else {
        bias_add_2d_kernel<T><<<gridSize, blockSize>>>(m, n, x, incx, b, incb, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void bias_add_4d_kernel_run(size_t m, size_t n, size_t o, size_t p, const T* x, size_t incx, const T* b, size_t incb, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bias_add_4d_kernel_flat<T>, 0, 0);
    }

    if (incx == 1 && incb == 1 && incy == 1) {
        bias_add_4d_kernel_flat<T><<<minGridSize, blockSize>>>(m, n, o, p, m * n * o * p, o * p, x, b, y);
    } else {
        bias_add_4d_kernel<T><<<minGridSize, blockSize>>>(m, n, o, p, x, incx, b, incb, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sbias_add_2d(size_t m, size_t n, const float* x, size_t incx, const float* b, size_t incb, float* y, size_t incy){
    bias_add_2d_kernel_run(m, n, x, incx, b, incb, y, incy);
}

void egblas_dbias_add_2d(size_t m, size_t n, const double* x, size_t incx, const double* b, size_t incb, double* y, size_t incy){
    bias_add_2d_kernel_run(m, n, x, incx, b, incb, y, incy);
}

void egblas_sbias_add_4d(size_t m, size_t n, size_t o, size_t p, const float* x, size_t incx, const float* b, size_t incb, float* y, size_t incy){
    bias_add_4d_kernel_run(m, n, o, p, x, incx, b, incb, y, incy);
}

void egblas_dbias_add_4d(size_t m, size_t n, size_t o, size_t p, const double* x, size_t incx, const double* b, size_t incb, double* y, size_t incy){
    bias_add_4d_kernel_run(m, n, o, p, x, incx, b, incb, y, incy);
}
