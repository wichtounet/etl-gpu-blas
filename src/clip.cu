//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/clip.hpp"

#include "complex.hpp"

template <typename T>
__device__ T clip(T x, T min, T max){
    if(x < min){
        return min;
    } else if(x > max){
        return max;
    } else {
        return x;
    }
}

template <>
__device__ cuComplex clip(cuComplex x, cuComplex min, cuComplex max){
    if(x.x < min.x || (x.x == min.x && x.y < min.y)){
        return min;
    } else if(x.x > max.x || (x.x == max.x && x.y > max.y)){
        return max;
    } else {
        return x;
    }
}

template <>
__device__ cuDoubleComplex clip(cuDoubleComplex x, cuDoubleComplex min, cuDoubleComplex max){
    if(x.x < min.x || (x.x == min.x && x.y < min.y)){
        return min;
    } else if(x.x > max.x || (x.x == max.x && x.y > max.y)){
        return max;
    } else {
        return x;
    }
}

template <typename T>
__global__ void clip_kernel(size_t n, T alpha, const T* x, size_t incx, const T* z, size_t incz, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * clip(y[incy *index], x[incx * index], z[incz * index]);
    }
}

template <typename T>
__global__ void clip_kernel1(size_t n, const T* x, size_t incx, const T* z, size_t incz, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = clip(y[incy *index], x[incx * index], z[incz * index]);
    }
}

template <typename T>
__global__ void clip_value_kernel(size_t n, T alpha, T x, T z, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * clip(y[incy *index], x, z);
    }
}

template <typename T>
__global__ void clip_value_kernel1(size_t n, T x, T z, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = clip(y[incy *index], x, z);
    }
}

template <typename T>
__global__ void clip_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = zero<T>();
    }
}

template <typename T>
void clip_kernel_run(size_t n, T alpha, const T* x, size_t incx, const T* z, size_t incz, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, clip_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    clip_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, z, incz, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void clip_kernel1_run(size_t n, const T* x, size_t incx, const T* z, size_t incz, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, clip_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    clip_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, z, incz, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void clip_value_kernel_run(size_t n, T alpha, T x, T z, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, clip_value_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    clip_value_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, z, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void clip_value_kernel1_run(size_t n, T x, T z, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, clip_value_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    clip_value_kernel1<T><<<gridSize, blockSize>>>(n, x, z, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void clip_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, clip_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    clip_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sclip(size_t n, float alpha, const float* x, size_t incx, const float* z, size_t incz, float* y, size_t incy) {
    if (alpha == 1.0f) {
        clip_kernel1_run(n, x, incx, z, incz, y, incy);
    } else if (alpha == 0.0f) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_kernel_run(n, alpha, x, incx, z, incz, y, incy);
    }
}

void egblas_dclip(size_t n, double alpha, const double* x, size_t incx, const double* z, size_t incz, double* y, size_t incy) {
    if (alpha == 1.0) {
        clip_kernel1_run(n, x, incx, z, incz, y, incy);
    } else if (alpha == 0.0) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_kernel_run(n, alpha, x, incx, z, incz, y, incy);
    }
}

void egblas_cclip(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, const cuComplex* z, size_t incz, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        clip_kernel1_run(n, x, incx, z, incz, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_kernel_run(n, alpha, x, incx, z, incz, y, incy);
    }
}

void egblas_zclip(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, const cuDoubleComplex* z, size_t incz, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        clip_kernel1_run(n, x, incx, z, incz, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_kernel_run(n, alpha, x, incx, z, incz, y, incy);
    }
}

void egblas_sclip_value(size_t n, float alpha, float x, float z, float* y, size_t incy) {
    if (alpha == 1.0f) {
        clip_value_kernel1_run(n, x, z, y, incy);
    } else if (alpha == 0.0f) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_value_kernel_run(n, alpha, x, z, y, incy);
    }
}

void egblas_dclip_value(size_t n, double alpha, double x, double z, double* y, size_t incy) {
    if (alpha == 1.0) {
        clip_value_kernel1_run(n, x, z, y, incy);
    } else if (alpha == 0.0) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_value_kernel_run(n, alpha, x, z, y, incy);
    }
}

void egblas_cclip_value(size_t n, cuComplex alpha, cuComplex x, cuComplex z, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        clip_value_kernel1_run(n, x, z, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_value_kernel_run(n, alpha, x, z, y, incy);
    }
}

void egblas_zclip_value(size_t n, cuDoubleComplex alpha, cuDoubleComplex x, cuDoubleComplex z, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        clip_value_kernel1_run(n, x, z, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        clip_kernel0_run(n, y, incy);
    } else {
        clip_value_kernel_run(n, alpha, x, z, y, incy);
    }
}
