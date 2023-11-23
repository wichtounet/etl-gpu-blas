//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================
#include "egblas/axpy.hpp"
#include <iostream>

#include "complex.hpp"

static constexpr int MAX_BLOCK_SIZE = 256;

template <typename T>
__global__ void axpy_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = y[incy * index] + alpha * x[incx * index];
    }
}

template <typename T>
__global__ void axpy_kernel_flat(size_t n, T alpha, const T* x, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = y[index] + alpha * x[index];
    }
}

template <typename T>
__global__ void axpy_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = x[incx * index] + y[incy * index];
    }
}

template <typename T>
void axpy_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy_kernel<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    axpy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpy_kernel_run_flat(size_t n, T alpha, const T* x, T* y) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy_kernel_flat<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    axpy_kernel_flat<T><<<gridSize, blockSize>>>(n, alpha, x, y);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpy_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy_kernel1<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;
    axpy_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

#ifdef EGBLAS_HAS_HAXPY

void egblas_haxpy(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16* y, size_t incy) {
    if (__low2float(alpha) == 1.0f && __high2float(alpha) == 1.0f) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (__low2float(alpha) == 0.0f && __high2float(alpha) == 0.0f) {
        return;
    } else if (incx == 1 && incy == 1) {
        axpy_kernel_run_flat(n, alpha, x, y);
    } else {
        axpy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

#endif

#ifdef EGBLAS_HAS_BAXPY

void egblas_baxpy(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16* y, size_t incy) {
    if (__low2float(alpha) == 1.0f && __high2float(alpha) == 1.0f) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (__low2float(alpha) == 0.0f && __high2float(alpha) == 0.0f) {
        return;
    } else if (incx == 1 && incy == 1) {
        axpy_kernel_run_flat(n, alpha, x, y);
    } else {
        axpy_kernel_run(n, alpha, x, incx, y, incy);
    }
}

#endif

void egblas_saxpy(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_daxpy(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_caxpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_zaxpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_oaxpy(size_t n, int8_t alpha, const int8_t* x, size_t incx, int8_t* y, size_t incy) {
    if (alpha == 1) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_waxpy(size_t n, int16_t alpha, const int16_t* x, size_t incx, int16_t* y, size_t incy) {
    if (alpha == 1) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_iaxpy(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t* y, size_t incy) {
    if (alpha == 1) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_laxpy(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t* y, size_t incy) {
    if (alpha == 1) {
        axpy_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0) {
        return;
    } else {
        if (incx == 1 && incy == 1) {
            axpy_kernel_run_flat(n, alpha, x, y);
        } else {
            axpy_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}
