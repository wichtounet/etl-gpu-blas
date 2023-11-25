//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axpby.hpp"

#include "complex.hpp"

static constexpr int MAX_BLOCK_SIZE = 256;

template <typename T>
__global__ void axpby_kernel(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = alpha * x[incx * index] + beta * y[incy * index];
    }
}

template <typename T>
__global__ void axpby_kernel1(size_t n, T alpha, const T* x, T beta, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = alpha * x[index] + beta * y[index];
    }
}

template <typename T>
__global__ void axpby_kernel_alpha1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = x[incx * index] + y[incy * index];
    }
}

template <typename T>
__global__ void axpby_kernel_alpha0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = zero<T>();
    }
}

template <typename T>
void axpby_kernel_run(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_kernel1<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        axpby_kernel1<T><<<gridSize, blockSize>>>(n, alpha, x, beta, y);
    } else {
        axpby_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, beta, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpby_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_kernel_alpha1<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpby_kernel_alpha1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpby_kernel0_run(size_t n, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_kernel_alpha0<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    axpby_kernel_alpha0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

#ifdef EGBLAS_HAS_HAXPBY

void egblas_haxpby(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16 beta, fp16* y, size_t incy) {
    if (__low2float(alpha) == 1.0f && __high2float(alpha) == 1.0f && __low2float(beta) == 1.0f && __high2float(beta) == 1.0f) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (__low2float(alpha) == 0.0f && __high2float(alpha) == 0.0f && __low2float(beta) == 0.0f && __high2float(beta) == 0.0f) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

#endif

#ifdef EGBLAS_HAS_BAXPBY

void egblas_baxpby(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16 beta, bf16* y, size_t incy) {
    if (__low2float(alpha) == 1.0f && __high2float(alpha) == 1.0f && __low2float(beta) == 1.0f && __high2float(beta) == 1.0f) {
        axpby_kernel1_run(n, x, incx, y, incy);
    } else if (__low2float(alpha) == 0.0f && __high2float(alpha) == 0.0f && __low2float(beta) == 0.0f && __high2float(beta) == 0.0f) {
        axpby_kernel0_run(n, y, incy);
    } else {
        axpby_kernel_run(n, alpha, x, incx, beta, y, incy);
    }
}

#endif

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
