//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/max.hpp"
#include "complex.hpp"

__device__ cuComplex max(cuComplex x, cuComplex y){
    if (x.x > y.x) {
        return x;
    } else if (y.x > x.x) {
        return y;
    } else {
        if (x.y > y.y) {
            return x;
        } else {
            return y;
        }
    }
}

__device__ cuDoubleComplex max(cuDoubleComplex x, cuDoubleComplex y){
    if (x.x > y.x) {
        return x;
    } else if (y.x > x.x) {
        return y;
    } else {
        if (x.y > y.y) {
            return x;
        } else {
            return y;
        }
    }
}

template <typename T>
__global__ void max_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * max(x[incx * index], y[incy * index]);
    }
}

template <typename T>
__global__ void max3_kernel(size_t n, T alpha, const T* a, size_t inca, const T* b, size_t incb, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * max(a[inca * index], b[incb * index]);
    }
}

template <typename T>
__global__ void max_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = max(x[incx * index], y[incy * index]);
    }
}

template <typename T>
__global__ void max3_kernel1(size_t n, const T* a, size_t inca, const T* b, size_t incb, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = max(a[inca * index], b[incb * index]);
    }
}

template <typename T>
__global__ void max_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void max_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void max_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void max_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    max_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void max_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    max_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void max_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    max_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void max_kernel_run(size_t n, T alpha, const T* a, size_t inca, const T* b, size_t incb, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max3_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    max3_kernel<T><<<gridSize, blockSize>>>(n, alpha, a, inca, b, incb, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void max_kernel1_run(size_t n, const T* a, size_t inca, const T* b, size_t incb, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max3_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    max3_kernel1<T><<<gridSize, blockSize>>>(n, a, inca, b, incb, y, incy);

    cudaDeviceSynchronize();
}

void egblas_smax(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        max_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dmax(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        max_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_cmax(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        max_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zmax(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        max_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_smax(size_t n, float alpha, const float* a, size_t inca, const float* b, size_t incb, float* y, size_t incy) {
    if (alpha == 1.0f) {
        max_kernel1_run(n, a, inca, b, incb, y, incy);
    } else if (alpha == 0.0f) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, a, inca, b, incb, y, incy);
    }
}

void egblas_dmax(size_t n, double alpha, const double* a, size_t inca, const double* b, size_t incb, double* y, size_t incy) {
    if (alpha == 1.0) {
        max_kernel1_run(n, a, inca, b, incb, y, incy);
    } else if (alpha == 0.0) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, a, inca, b, incb, y, incy);
    }
}

void egblas_cmax(size_t n, cuComplex alpha, const cuComplex* a, size_t inca, const cuComplex* b, size_t incb, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        max_kernel1_run(n, a, inca, b, incb, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, a, inca, b, incb, y, incy);
    }
}

void egblas_zmax(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* a, size_t inca, const cuDoubleComplex* b, size_t incb, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        max_kernel1_run(n, a, inca, b, incb, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        max_kernel0_run(n, y, incy);
    } else {
        max_kernel_run(n, alpha, a, inca, b, incb, y, incy);
    }
}
