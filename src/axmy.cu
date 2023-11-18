//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axmy.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axmy_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * (x[incx * index] * y[incy * index]);
    }
}

template <typename T>
__global__ void axmy_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = x[incx * index] * y[incy * index];
    }
}

template <typename T>
__global__ void axmy_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = zero<T>();
    }
}

template <typename T>
void axmy_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axmy_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    axmy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axmy_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axmy_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    axmy_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axmy_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axmy_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    axmy_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

#ifdef EGBLAS_HAS_HAXMY

void egblas_haxmy(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16* y, size_t incy) {
    if (__low2float(alpha) == 1.0f && __high2float(alpha) == 1.0f) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (__low2float(alpha) == 0.0f && __high2float(alpha) == 0.0f) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

#endif

#ifdef EGBLAS_HAS_BAXMY

void egblas_baxmy(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16* y, size_t incy) {
    if (__low2float(alpha) == 1.0f && __high2float(alpha) == 1.0f) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (__low2float(alpha) == 0.0f && __high2float(alpha) == 0.0f) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

#endif

void egblas_saxmy(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_daxmy(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_caxmy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zaxmy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_iaxmy(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t* y, size_t incy) {
    if (alpha == 1) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_laxmy(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t* y, size_t incy) {
    if (alpha == 1) {
        axmy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0) {
        axmy_kernel0_run(n, y, incy);
    } else {
        axmy_kernel_run(n, alpha, x, incx, y, incy);
    }
}
