//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/conv_1d.hpp"

#include "complex.hpp"

template <typename T>
__global__ void conv1_valid_kernel(size_t N, size_t K, T alpha, const T* x, size_t incx, const T * k, size_t inck, T* y, size_t incy) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N - K + 1) {
        T value(0);

        for (size_t l = 0; l < K; ++l) {
            value += x[(index + l) * incx] * k[(K - 1 - l) * inck];
        }

        y[index * incy] = alpha * value;
    }
}

template <typename T>
__global__ void conv1_valid_kernel_flat(size_t N, size_t K, T alpha, const T* x, const T * k, T* y) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N - K + 1) {
        T value(0);

        for (size_t l = 0; l < K; ++l) {
            value += x[index + l] * k[K - 1 - l];
        }

        y[index] = alpha * value;
    }
}

template <typename T>
__global__ void conv1_valid_kernel1(size_t N, size_t K, const T* x, size_t incx, const T * k, size_t inck, T* y, size_t incy) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N - K + 1) {
        T value(0);

        for (size_t l = 0; l < K; ++l) {
            value += x[(index + l) * incx] * k[(K - 1 - l) * inck];
        }

        y[index * incy] = value;
    }
}

template <typename T>
__global__ void conv1_valid_kernel1_flat(size_t N, size_t K, const T* x, const T * k, T* y) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N - K + 1) {
        T value(0);

        for (size_t l = 0; l < K; ++l) {
            value += x[index + l] * k[K - 1 - l];
        }

        y[index] = value;
    }
}

template <typename T>
void conv1_valid_kernel_run(size_t N, size_t K, T alpha, const T* x, size_t incx, const T * k, size_t inck, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, conv1_valid_kernel<T>, 0, 0);

    int gridSize = ((N - K + 1) + blockSize - 1) / blockSize;

    if (alpha == T(1)) {
        if (incx == 1 && inck == 1 && incy == 1) {
            conv1_valid_kernel1_flat<T><<<gridSize, blockSize>>>(N, K, x, k, y);
        } else {
            conv1_valid_kernel1<T><<<gridSize, blockSize>>>(N, K, x, incx, k, inck, y, incy);
        }
    } else {
        if (incx == 1 && inck == 1 && incy == 1) {
            conv1_valid_kernel_flat<T><<<gridSize, blockSize>>>(N, K, alpha, x, k, y);
        } else {
            conv1_valid_kernel<T><<<gridSize, blockSize>>>(N, K, alpha, x, incx, k, inck, y, incy);
        }
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sconv1_valid(size_t N, size_t K, float alpha, const float* x, size_t incx, const float * k, size_t inck, float* y, size_t incy) {
    conv1_valid_kernel_run(N, K, alpha, x, incx, k, inck, y, incy);
}

void egblas_dconv1_valid(size_t N, size_t K, double alpha, const double* x, size_t incx, const double * k, size_t inck, double* y, size_t incy) {
    conv1_valid_kernel_run(N, K, alpha, x, incx, k, inck, y, incy);
}

template<typename T>
__device__ T max(T a, T b){
    return a > b ? a : b;
}

template<typename T>
__device__ T min(T a, T b){
    return a < b ? a : b;
}

template <typename T>
__global__ void conv1_same_kernel(size_t N, size_t K, T alpha, const T* x, size_t incx, const T * k, size_t inck, T* y, size_t incy) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        T value(0);

        size_t l_lo = max<int>(0, index - (K - 1) / 2);
        size_t l_hi = min<int>(N - 1, index + K / 2) + 1;

        for (size_t l = l_lo; l < l_hi; ++l) {
            value += x[l * incx] * k[(index - l + K / 2) * inck];
        }

        y[index * incy] = alpha * value;
    }
}

template <typename T>
__global__ void conv1_same_kernel_flat(size_t N, size_t K, T alpha, const T* x, const T * k, T* y) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        T value(0);

        size_t l_lo = max<int>(0, index - (K - 1) / 2);
        size_t l_hi = min<int>(N - 1, index + K / 2) + 1;

        for (size_t l = l_lo; l < l_hi; ++l) {
            value += x[l] * k[index - l + K / 2];
        }

        y[index] = alpha * value;
    }
}

template <typename T>
__global__ void conv1_same_kernel1(size_t N, size_t K, const T* x, size_t incx, const T * k, size_t inck, T* y, size_t incy) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        T value(0);

        size_t l_lo = max<int>(0, index - (K - 1) / 2);
        size_t l_hi = min<int>(N - 1, index + K / 2) + 1;

        for (size_t l = l_lo; l < l_hi; ++l) {
            value += x[l * incx] * k[(index - l + K / 2) * inck];
        }

        y[index * incy] = value;
    }
}

template <typename T>
__global__ void conv1_same_kernel1_flat(size_t N, size_t K, const T* x, const T * k, T* y) {
    const auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        T value(0);

        size_t l_lo = max<int>(0, index - (K - 1) / 2);
        size_t l_hi = min<int>(N - 1, index + K / 2) + 1;

        for (size_t l = l_lo; l < l_hi; ++l) {
            value += x[l] * k[index - l + K / 2];
        }

        y[index] = value;
    }
}

template <typename T>
void conv1_same_kernel_run(size_t N, size_t K, T alpha, const T* x, size_t incx, const T * k, size_t inck, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, conv1_same_kernel<T>, 0, 0);

    int gridSize = (N + blockSize - 1) / blockSize;

    if (alpha == T(1)) {
        if (incx == 1 && inck == 1 && incy == 1) {
            conv1_same_kernel1_flat<T><<<gridSize, blockSize>>>(N, K, x, k, y);
        } else {
            conv1_same_kernel1<T><<<gridSize, blockSize>>>(N, K, x, incx, k, inck, y, incy);
        }
    } else {
        if (incx == 1 && inck == 1 && incy == 1) {
            conv1_same_kernel_flat<T><<<gridSize, blockSize>>>(N, K, alpha, x, k, y);
        } else {
            conv1_same_kernel<T><<<gridSize, blockSize>>>(N, K, alpha, x, incx, k, inck, y, incy);
        }
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sconv1_same(size_t N, size_t K, float alpha, const float* x, size_t incx, const float * k, size_t inck, float* y, size_t incy) {
    conv1_same_kernel_run(N, K, alpha, x, incx, k, inck, y, incy);
}

void egblas_dconv1_same(size_t N, size_t K, double alpha, const double* x, size_t incx, const double * k, size_t inck, double* y, size_t incy) {
    conv1_same_kernel_run(N, K, alpha, x, incx, k, inck, y, incy);
}
