//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axdy_3.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axdy_3_kernel(size_t n, T alpha, const T* x, size_t incx, const T* y, size_t incy, T* yy, size_t incyy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        yy[incyy * index] = x[incx * index] / (alpha * y[incy * index]);
    }
}

template <typename T>
__global__ void axdy_3_kernel_flat(size_t n, T alpha, const T* x, const T* y, T* yy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        yy[index] = x[index] / (alpha * y[index]);
    }
}

template <typename T>
__global__ void axdy_3_kernel1(size_t n, const T* x, size_t incx, const T* y, size_t incy, T* yy, size_t incyy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        yy[incyy * index] = x[incx * index] / y[incy * index];
    }
}

template <typename T>
__global__ void axdy_3_kernel1_flat(size_t n, const T* x, const T* y, T* yy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        yy[index] = x[index] / y[index];
    }
}

template <typename T>
void axdy_3_kernel_run(size_t n, T alpha, const T* x, size_t incx, const T* y, size_t incy, T* yy, size_t incyy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axdy_3_kernel<T>, 0, 0);
    }

    int gridSize = ((n / incyy) + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1 && incyy == 1) {
        axdy_3_kernel_flat<T><<<gridSize, blockSize>>>(n, alpha, x, y, yy);
    } else {
        axdy_3_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy, yy, incyy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axdy_3_kernel1_run(size_t n, const T* x, size_t incx, const T* y, size_t incy, T* yy, size_t incyy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axdy_3_kernel1<T>, 0, 0);
    }

    int gridSize = ((n / incyy) + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1 && incyy == 1) {
        axdy_3_kernel1_flat<T><<<gridSize, blockSize>>>(n, x, y, yy);
    } else {
        axdy_3_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy, yy, incyy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_saxdy_3(size_t n, float alpha, const float* x, size_t incx, const float* y, size_t incy, float* yy, size_t incyy) {
    if (alpha == 1.0f) {
        axdy_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else {
        axdy_3_kernel_run(n, alpha, x, incx, y, incy, yy, incyy);
    }
}

void egblas_daxdy_3(size_t n, double alpha, const double* x, size_t incx, const double* y, size_t incy, double* yy, size_t incyy) {
    if (alpha == 1.0) {
        axdy_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else {
        axdy_3_kernel_run(n, alpha, x, incx, y, incy, yy, incyy);
    }
}

void egblas_caxdy_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, const cuComplex* y, size_t incy, cuComplex* yy, size_t incyy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        axdy_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else {
        axdy_3_kernel_run(n, alpha, x, incx, y, incy, yy, incyy);
    }
}

void egblas_zaxdy_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, const cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        axdy_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else {
        axdy_3_kernel_run(n, alpha, x, incx, y, incy, yy, incyy);
    }
}

void egblas_iaxdy_3(size_t n, int32_t alpha, const int32_t* x, size_t incx, const int32_t* y, size_t incy, int32_t* yy, size_t incyy) {
    if (alpha == 1) {
        axdy_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else {
        axdy_3_kernel_run(n, alpha, x, incx, y, incy, yy, incyy);
    }
}

void egblas_laxdy_3(size_t n, int64_t alpha, const int64_t* x, size_t incx, const int64_t* y, size_t incy, int64_t* yy, size_t incyy) {
    if (alpha == 1) {
        axdy_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else {
        axdy_3_kernel_run(n, alpha, x, incx, y, incy, yy, incyy);
    }
}
