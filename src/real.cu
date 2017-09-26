//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/real.hpp"

template <typename T, typename TT>
__global__ void real_kernel(size_t n, TT alpha, const T* x, size_t incx, TT* y, size_t incy);

template <>
__global__ void real_kernel(size_t n, float alpha, const cuComplex* x, size_t incx, float* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        y[incx * index] = alpha * c.x;
    }
}

template <>
__global__ void real_kernel(size_t n, double alpha, const cuDoubleComplex* x, size_t incx, double* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        y[incx * index] = alpha * c.x;
    }
}

template <typename T, typename TT>
__global__ void real_kernel1(size_t n, const T* x, size_t incx, TT* y, size_t incy);

template <>
__global__ void real_kernel1(size_t n, const cuComplex* x, size_t incx, float* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        y[incx * index] = c.x;
    }
}

template <>
__global__ void real_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, double* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        y[incx * index] = c.x;
    }
}

template <typename T>
__global__ void real_kernel0(size_t n, T* y, size_t incy){
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = 0;
    }
}

template <typename T, typename TT>
void real_kernel_run(size_t n, TT alpha, const T* x, size_t incx, TT* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, real_kernel<T, TT>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    real_kernel<T, TT><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T, typename TT>
void real_kernel1_run(size_t n, const T* x, size_t incx, TT* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, real_kernel1<T, TT>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    real_kernel1<T, TT><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void real_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, real_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    real_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_creal(size_t n, float alpha, const cuComplex* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        real_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        real_kernel0_run(n, y, incy);
    } else {
        real_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zreal(size_t n, double alpha, const cuDoubleComplex* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        real_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        real_kernel0_run(n, y, incy);
    } else {
        real_kernel_run(n, alpha, x, incx, y, incy);
    }
}
