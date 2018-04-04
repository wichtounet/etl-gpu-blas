//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axpby.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axpby_kernel(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * x[incx * index] + beta * y[incy * index];
    }
}

template <typename T>
__global__ void axpby_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = x[incx * index] + y[incy * index];
    }
}

template <typename T>
__global__ void axpby_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = zero<T>();
    }
}

template <typename T>
void axpby_kernel_run(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_kernel<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpby_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, beta, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpby_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_kernel1<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpby_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpby_kernel0_run(size_t n, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_kernel0<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpby_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_saxpby(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy) {
    if (alpha == 1.0f && beta == 1.0f) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f && beta == 0.0f) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

void egblas_daxpby(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy) {
    if (alpha == 1.0 && beta == 1.0) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0 && beta == 0.0) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

void egblas_caxpby(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f && beta.x == 1.0f && beta.y == 0.0f) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f && beta.x == 1.0f && beta.y == 0.0f) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

void egblas_zaxpby(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0 && beta.x == 1.0 && beta.y == 0.0) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0 && beta.x == 1.0 && beta.y == 0.0) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

void egblas_iaxpby(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t beta, int32_t* y, size_t incy) {
    if (alpha == 1 && beta == 1) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0 && beta == 0) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

void egblas_laxpby(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t beta, int64_t* y, size_t incy) {
    if (alpha == 1 && beta == 1) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0 && beta == 0) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}
