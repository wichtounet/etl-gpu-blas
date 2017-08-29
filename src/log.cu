//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/log.hpp"

template <typename T>
__global__ void log_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * logf(x[incx * index]);
    }
}

template <typename T>
__global__ void log_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = logf(x[incx * index]);
    }
}

template <typename T>
__global__ void log_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <typename T>
void log_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, log_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    log_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void log_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, log_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    log_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void log_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, log_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    log_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_slog(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        log_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        log_kernel0_run(n, y, incy);
    } else {
        log_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dlog(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        log_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        log_kernel0_run(n, y, incy);
    } else {
        log_kernel_run(n, alpha, x, incx, y, incy);
    }
}
