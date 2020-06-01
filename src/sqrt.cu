//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/sqrt.hpp"

#include "complex.hpp"

template <typename T>
__global__ void sqrt_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = alpha * sqrt(x[incx * index]);
    }
}

template <typename T>
__global__ void sqrt_kernel_flat(size_t n, T alpha, const T* x, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = alpha * sqrt(x[index]);
    }
}

template <>
__global__ void sqrt_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void sqrt_kernel_flat(size_t n, cuComplex alpha, const cuComplex* x, cuComplex* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[index];

        auto res = sqrt(c);

        y[index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void sqrt_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = cuCmul(alpha, res);
    }
}

template <>
__global__ void sqrt_kernel_flat(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, cuDoubleComplex* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[index];

        auto res = sqrt(c);

        y[index] = cuCmul(alpha, res);
    }
}

template <typename T>
__global__ void sqrt_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = sqrt(x[incx * index]);
    }
}

template <typename T>
__global__ void sqrt_kernel1_flat(size_t n, const T* x, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = sqrt(x[index]);
    }
}

template <>
__global__ void sqrt_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = res;
    }
}

template <>
__global__ void sqrt_kernel1_flat(size_t n, const cuComplex* x, cuComplex* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[index];

        auto res = sqrt(c);

        y[index] = res;
    }
}

template <>
__global__ void sqrt_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = res;
    }
}

template <>
__global__ void sqrt_kernel1_flat(size_t n, const cuDoubleComplex* x, cuDoubleComplex* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        auto c = x[index];

        auto res = sqrt(c);

        y[index] = res;
    }
}

template <typename T>
__global__ void sqrt_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = zero<T>();
    }
}

template <typename T>
__global__ void sqrt_kernel0_flat(size_t n, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = zero<T>();
    }
}

template <typename T>
void sqrt_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sqrt_kernel<T>, 0, 0);
    }

    int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        sqrt_kernel_flat<T><<<gridSize, blockSize>>>(n, alpha, x, y);
    } else {
        sqrt_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void sqrt_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sqrt_kernel1<T>, 0, 0);
    }

    int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        sqrt_kernel1_flat<T><<<gridSize, blockSize>>>(n, x, y);
    } else {
        sqrt_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void sqrt_kernel0_run(size_t n, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sqrt_kernel0<T>, 0, 0);
    }

    int gridSize = (n + blockSize - 1) / blockSize;

    if (incy == 1) {
        sqrt_kernel0_flat<T><<<gridSize, blockSize>>>(n, y);
    } else {
        sqrt_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_ssqrt(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        sqrt_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        sqrt_kernel0_run(n, y, incy);
    } else {
        sqrt_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dsqrt(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        sqrt_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        sqrt_kernel0_run(n, y, incy);
    } else {
        sqrt_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_csqrt(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        sqrt_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        sqrt_kernel0_run(n, y, incy);
    } else {
        sqrt_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zsqrt(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        sqrt_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        sqrt_kernel0_run(n, y, incy);
    } else {
        sqrt_kernel_run(n, alpha, x, incx, y, incy);
    }
}
