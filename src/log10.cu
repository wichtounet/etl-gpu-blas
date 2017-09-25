//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/log10.hpp"

template <typename T>
__global__ void log10_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * log10(x[incx * index]);
    }
}

template <>
__global__ void log10_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    auto log10 = make_cuComplex(log(10.0f), 0.0f);

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        float c_abs = hypot(c.x, c.y);
        float c_arg = atan2(c.y, c.x);

        auto logx = make_cuComplex(log(c_abs), c_arg);

        auto res = cuCdivf(logx, log10);

        y[incx * index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void log10_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    auto log10 = make_cuDoubleComplex(log(10.0), 0.0);

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        double c_abs = hypot(c.x, c.y);
        double c_arg = atan2(c.y, c.x);

        auto logx = make_cuDoubleComplex(log(c_abs), c_arg);

        auto res = cuCdiv(logx, log10);

        y[incx * index] = cuCmul(alpha, res);
    }
}

template <typename T>
__global__ void log10_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = log10(x[incx * index]);
    }
}

template <>
__global__ void log10_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    auto log10 = make_cuComplex(log(10.0f), 0.0f);

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        float c_abs = hypot(c.x, c.y);
        float c_arg = atan2(c.y, c.x);

        auto logx = make_cuComplex(log(c_abs), c_arg);

        auto res = cuCdivf(logx, log10);

        y[incx * index] = res;
    }
}

template <>
__global__ void log10_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    auto log10 = make_cuDoubleComplex(log(10.0), 0.0);

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        double c_abs = hypot(c.x, c.y);
        double c_arg = atan2(c.y, c.x);

        auto logx = make_cuDoubleComplex(log(c_abs), c_arg);

        auto res = cuCdiv(logx, log10);

        y[incx * index] = res;
    }
}

template <typename T>
__global__ void log10_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void log10_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void log10_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void log10_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, log10_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    log10_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void log10_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, log10_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    log10_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void log10_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, log10_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    log10_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_slog10(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        log10_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        log10_kernel0_run(n, y, incy);
    } else {
        log10_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dlog10(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        log10_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        log10_kernel0_run(n, y, incy);
    } else {
        log10_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_clog10(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        log10_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        log10_kernel0_run(n, y, incy);
    } else {
        log10_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zlog10(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        log10_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        log10_kernel0_run(n, y, incy);
    } else {
        log10_kernel_run(n, alpha, x, incx, y, incy);
    }
}
