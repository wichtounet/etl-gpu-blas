//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/sigmoid.hpp"
#include <iostream>

static constexpr int MAX_BLOCK_SIZE = 256;

namespace {

template<typename T>
__device__ T logistic_sigmoid(T x){
    return T(1) / (T(1) + exp(-x));
}

template <typename T>
__global__ void sigmoid_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = alpha * logistic_sigmoid(x[incx * index]);
    }
}

template <typename T>
__global__ void sigmoid_kernel_flat(size_t n, T alpha, const T* x, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = alpha * logistic_sigmoid(x[index]);
    }
}

template <typename T>
__global__ void sigmoid_kernel_alpha1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = logistic_sigmoid(x[incx * index]);
    }
}

template <typename T>
__global__ void sigmoid_kernel_alpha1_flat(size_t n, const T* x, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[index] = logistic_sigmoid(x[index]);
    }
}

template <typename T>
__global__ void sigmoid_kernel_alpha0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = T(0);
    }
}

template <typename T>
void sigmoid_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sigmoid_kernel<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    sigmoid_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void sigmoid_kernel_run_flat(size_t n, T alpha, const T* x, T* y) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sigmoid_kernel_flat<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    sigmoid_kernel_flat<T><<<gridSize, blockSize>>>(n, alpha, x, y);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void sigmoid_kernel_alpha1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sigmoid_kernel_alpha1<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;
    sigmoid_kernel_alpha1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void sigmoid_kernel_alpha1_run_flat(size_t n, const T* x, T* y) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sigmoid_kernel_alpha1_flat<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;
    sigmoid_kernel_alpha1_flat<T><<<gridSize, blockSize>>>(n, x, y);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void sigmoid_kernel_alpha0_run(size_t n, T* y, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sigmoid_kernel_alpha0<T>, 0, 0);
        blockSize = blockSize > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : blockSize;
    }

    const int gridSize = (n + blockSize - 1) / blockSize;

    sigmoid_kernel_alpha0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

} // End of anonymous namespace

void egblas_ssigmoid(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        if (incx == 1 && incy == 1) {
            sigmoid_kernel_alpha1_run_flat(n, x, y);
        } else {
            sigmoid_kernel_alpha1_run(n, x, incx, y, incy);
        }
    } else if (alpha == 0.0f) {
        sigmoid_kernel_alpha0_run(n, y, incy);
    } else {
        if (incx == 1 && incy == 1) {
            sigmoid_kernel_run_flat(n, alpha, x, y);
        } else {
            sigmoid_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_dsigmoid(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        if (incx == 1 && incy == 1) {
            sigmoid_kernel_alpha1_run_flat(n, x, y);
        } else {
            sigmoid_kernel_alpha1_run(n, x, incx, y, incy);
        }
    } else if (alpha == 0.0) {
        sigmoid_kernel_alpha0_run(n, y, incy);
    } else {
        if (incx == 1 && incy == 1) {
            sigmoid_kernel_run_flat(n, alpha, x, y);
        } else {
            sigmoid_kernel_run(n, alpha, x, incx, y, incy);
        }
    }
}
