//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axdbpy.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axdbpy_kernel(size_t n, const T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = (alpha * x[incx * index]) / (beta + y[incy * index]);
    }
}

template <typename T>
__global__ void axdbpy_kernel1(size_t n, const T alpha, const T* x, T beta, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[index] = (alpha * x[index]) / (beta + y[index]);
    }
}

template <typename T>
void axdbpy_kernel_run(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axdbpy_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        axdbpy_kernel1<T><<<gridSize, blockSize>>>(n, alpha, x, beta, y);
    } else {
        axdbpy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, beta, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_saxdbpy(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy) {
    axdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}

void egblas_daxdbpy(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy) {
    axdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}

void egblas_caxdbpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy) {
    axdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}

void egblas_zaxdbpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy) {
    axdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}
