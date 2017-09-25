//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/sqrt.hpp"

__device__ float abs(const cuComplex z){
    auto x = z.x;
    auto y = z.y;

    auto s = max(abs(x), abs(y));

    if(s == 0.0f){
        return 0.0f;
    }

    x = x / s;
    y = y / s;

    return s * sqrt(x * x + y * y);
}

__device__ double abs(const cuDoubleComplex z){
    auto x = z.x;
    auto y = z.y;

    auto s = max(abs(x), abs(y));

    if(s == 0.0){
        return 0.0;
    }

    x = x / s;
    y = y / s;

    return s * sqrt(x * x + y * y);
}

__device__ cuComplex sqrt(cuComplex z){
    auto x = z.x;
    auto y = z.y;

    if(x == 0.0f){
        auto t = sqrt(abs(y) / 2);
        return make_cuComplex(t, y < 0.0f ? -t : t);
    } else {
        auto t = sqrt(2 * (abs(z) + abs(x)));
        auto u = t / 2;

        if(x > 0.0f){
            return make_cuComplex(u, y / t);
        } else {
            return make_cuComplex(abs(y) / t, y < 0.0f ? -u : u);
        }
    }
}

__device__ cuDoubleComplex sqrt(cuDoubleComplex z){
    auto x = z.x;
    auto y = z.y;

    if(x == 0.0){
        auto t = sqrt(abs(y) / 2);
        return make_cuDoubleComplex(t, y < 0.0 ? -t : t);
    } else {
        auto t = sqrt(2 * (abs(z) + abs(x)));
        auto u = t / 2;

        if(x > 0.0){
            return make_cuDoubleComplex(u, y / t);
        } else {
            return make_cuDoubleComplex(abs(y) / t, y < 0.0 ? -u : u);
        }
    }
}

template <typename T>
__global__ void sqrt_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * sqrtf(x[incx * index]);
    }
}

template <>
__global__ void sqrt_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void sqrt_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = cuCmul(alpha, res);
    }
}

template <typename T>
__global__ void sqrt_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = sqrtf(x[incx * index]);
    }
}

template <>
__global__ void sqrt_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = res;
    }
}

template <>
__global__ void sqrt_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto c = x[incx * index];

        auto res = sqrt(c);

        y[incy * index] = res;
    }
}

template <typename T>
__global__ void sqrt_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void sqrt_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void sqrt_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void sqrt_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sqrt_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    sqrt_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void sqrt_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sqrt_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    sqrt_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void sqrt_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sqrt_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    sqrt_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
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
