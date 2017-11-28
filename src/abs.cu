//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/abs.hpp"
#include "complex.hpp"

template <typename T1, typename T2>
__global__ void abs_kernel(size_t n, T1 alpha, const T2* x, size_t incx, T1* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * abs(x[incx * index]);
    }
}

template <typename T1, typename T2>
__global__ void abs_kernel1(size_t n, const T2* x, size_t incx, T1* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = abs(x[incx * index]);
    }
}

template <typename T>
__global__ void abs_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = zero<T>();
    }
}

template <typename T1, typename T2>
void abs_kernel_run(size_t n, T1 alpha, const T2* x, size_t incx, T1* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, abs_kernel<T1, T2>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    abs_kernel<T1, T2><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T1, typename T2>
void abs_kernel1_run(size_t n, const T2* x, size_t incx, T1* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, abs_kernel1<T1, T2>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    abs_kernel1<T1, T2><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void abs_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, abs_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    abs_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sabs(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        abs_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        abs_kernel0_run(n, y, incy);
    } else {
        abs_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dabs(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        abs_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        abs_kernel0_run(n, y, incy);
    } else {
        abs_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_cabs(size_t n, float alpha, const cuComplex* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        abs_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        abs_kernel0_run(n, y, incy);
    } else {
        abs_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zabs(size_t n, double alpha, const cuDoubleComplex* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        abs_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        abs_kernel0_run(n, y, incy);
    } else {
        abs_kernel_run(n, alpha, x, incx, y, incy);
    }
}
