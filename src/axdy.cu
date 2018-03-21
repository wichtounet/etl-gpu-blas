//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axdy.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axdy_kernel(size_t n, const T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = y[incy * index] / (alpha * x[incx * index]);
    }
}

template <typename T>
__global__ void axdy_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = y[incy * index] / x[incx * index];
    }
}

template <typename T>
void axdy_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axdy_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    axdy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axdy_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axdy_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    axdy_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_saxdy(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        axdy_kernel1_run(n, x, incx, y, incy);
    } else {
        axdy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_daxdy(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        axdy_kernel1_run(n, x, incx, y, incy);
    } else {
        axdy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_caxdy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        axdy_kernel1_run(n, x, incx, y, incy);
    } else {
        axdy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zaxdy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        axdy_kernel1_run(n, x, incx, y, incy);
    } else {
        axdy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_iaxdy(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t* y, size_t incy) {
    if (alpha == 1) {
        axdy_kernel1_run(n, x, incx, y, incy);
    } else {
        axdy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_laxdy(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t* y, size_t incy) {
    if (alpha == 1) {
        axdy_kernel1_run(n, x, incx, y, incy);
    } else {
        axdy_kernel_run(n, alpha, x, incx, y, incy);
    }
}
