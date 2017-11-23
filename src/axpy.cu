//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axpy.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axpy_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * x[incx * index] + y[incy * index];
    }
}

template <typename T>
__global__ void axpy_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = x[incx * index] + y[incy * index];
    }
}

template <typename T>
__global__ void axpy_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = zero<T>();
    }
}

template <typename T>
void axpy_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy_kernel<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void axpy_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy_kernel1<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpy_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void axpy_kernel0_run(size_t n, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy_kernel0<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpy_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_saxpy(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        axpy_kernel0_run(n, y, incy);
    } else {
        axpy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_daxpy(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        axpy_kernel0_run(n, y, incy);
    } else {
        axpy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_caxpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        axpy_kernel0_run(n, y, incy);
    } else {
        axpy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zaxpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        axpy_kernel0_run(n, y, incy);
    } else {
        axpy_kernel_run(n, alpha, x, incx, y, incy);
    }
}
