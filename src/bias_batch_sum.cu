//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/assert.hpp"
#include "egblas/utils.hpp"
#include "egblas/sum.hpp"
#include "egblas/cuda_check.hpp"

#include "sum_reduce.hpp"

template <bool Mean, typename T>
__global__ void bias_batch_sum_kernel(size_t B, size_t N, const T* x, size_t incx, T* y, size_t incy) {
    auto n  = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N) {
        T sum = 0;

        for (size_t b = 0; b < B; ++b) {
            sum += x[(b * N + n) * incx];
        }

        if (Mean) {
            y[incy * n] = sum / B;
        } else {
            y[incy * n] = sum;
        }
    }
}

template <bool Mean, typename T>
__global__ void bias_batch_sum_kernel_flat(size_t B, size_t N, const T* x, T* y) {
    auto n  = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N) {
        T sum = 0;

        size_t b = 0;

        for (; b + 7 < B; b += 8) {
            sum += x[(b + 0) * N + n];
            sum += x[(b + 1) * N + n];
            sum += x[(b + 2) * N + n];
            sum += x[(b + 3) * N + n];
            sum += x[(b + 4) * N + n];
            sum += x[(b + 5) * N + n];
            sum += x[(b + 6) * N + n];
            sum += x[(b + 7) * N + n];
        }

        for (; b + 3 < B; b += 4) {
            sum += x[(b + 0) * N + n];
            sum += x[(b + 1) * N + n];
            sum += x[(b + 2) * N + n];
            sum += x[(b + 3) * N + n];
        }

        for (; b + 1 < B; b += 2) {
            sum += x[(b + 0) * N + n];
            sum += x[(b + 1) * N + n];
        }

        // Note: This should be a if, but using a if makes it slower
        for (; b < B; ++b) {
            sum += x[b * N + n];
        }

        if (Mean) {
            y[n] = sum / B;
        } else {
            y[n] = sum;
        }
    }
}

void egblas_sbias_batch_sum(size_t b, size_t n, float* x, size_t incx, float* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        bias_batch_sum_kernel_flat<false><<<gridSize, blockSize>>>(b, n, x, y);
    } else {
        bias_batch_sum_kernel<false><<<gridSize, blockSize>>>(b, n, x, incx, y, incy);
    }
}

void egblas_dbias_batch_sum(size_t b, size_t n, double* x, size_t incx, double* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        bias_batch_sum_kernel_flat<false><<<gridSize, blockSize>>>(b, n, x, y);
    } else {
        bias_batch_sum_kernel<false><<<gridSize, blockSize>>>(b, n, x, incx, y, incy);
    }
}

void egblas_sbias_batch_mean(size_t b, size_t n, float* x, size_t incx, float* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        bias_batch_sum_kernel_flat<true><<<gridSize, blockSize>>>(b, n, x, y);
    } else {
        bias_batch_sum_kernel<true><<<gridSize, blockSize>>>(b, n, x, incx, y, incy);
    }
}

void egblas_dbias_batch_mean(size_t b, size_t n, double* x, size_t incx, double* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize = (n + blockSize - 1) / blockSize;

    if (incx == 1 && incy == 1) {
        bias_batch_sum_kernel_flat<true><<<gridSize, blockSize>>>(b, n, x, y);
    } else {
        bias_batch_sum_kernel<true><<<gridSize, blockSize>>>(b, n, x, incx, y, incy);
    }
}

// 4D Versions

__device__ inline void atomicAddF(float* address, float value){
#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
  atomicAdd(address,value);
#elif __CUDA_ARCH__ >= 110
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
#endif
}

__device__ inline void atomicAddF(double* address, double value){
    unsigned long long int* a = (unsigned long long int*) address;
    unsigned long long int old = *a, assumed;

    do {
        assumed = old;
        old     = atomicCAS(a, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
}

template <size_t Factor, typename T>
__global__ void bias_batch_sum4_kernel_zero(size_t Last, size_t Limit, const T* x, T* y) {
    auto base_n  = threadIdx.x + blockIdx.x * blockDim.x;

    if (base_n < Limit) {
        const size_t t = base_n & (Factor - 1);

        base_n = base_n / Factor;

        const T * xx = x + base_n * Last;

        T sum = 0;

        for (size_t o = t; o < Last; o += Factor) {
            sum += xx[o];
        }

        atomicAddF(&y[base_n], sum);
    }
}

template <size_t Factor, bool Mean, typename T>
__global__ void bias_batch_sum4_kernel_last(size_t B, size_t N, size_t W, size_t H, size_t Limit, const T* x, T* y) {
    auto base_n  = threadIdx.x + blockIdx.x * blockDim.x;

    if (base_n < Limit) {
        const size_t t = base_n & (Factor - 1);

        base_n = base_n / Factor;

        T sum = 0;

        for (size_t o = t; o < B; o += Factor) {
            sum += x[o * N + base_n];
        }

        if (Mean) {
            atomicAddF(&y[base_n], sum / (B * W * H));
        } else {
            atomicAddF(&y[base_n], sum);
        }
    }
}

// Opportunities for further reductions
// Do a wrap reduction instead of an atomic reduction
// Optimize the three Factor parameter based on the data
// Optimize blockSize based on the data

template <bool Mean, typename T>
void egblas_sbias_batch_sum4_run(size_t b, size_t n, size_t w, size_t h, T* x, T* y) {
    T* tmp_zero;
    T* tmp_first;

    cuda_check(cudaMalloc((void**)&tmp_zero, b * n * w * sizeof(T)));
    cuda_check(cudaMalloc((void**)&tmp_first, b * n * sizeof(T)));

    cudaMemset(tmp_zero, 0, b * n * w * sizeof(T));
    cudaMemset(tmp_first, 0, b * n * sizeof(T));
    cudaMemset(y, 0, n * sizeof(T));

    // Phase 0 (Bottleneck)

    int blockSize = 128;
    int gridSize = (8 * b * n * w + blockSize - 1) / blockSize;

    bias_batch_sum4_kernel_zero<8><<<gridSize, blockSize>>>(h, 8 * b * n * w, x, tmp_zero);

    // Phase 1

    blockSize = 128;
    gridSize = (8 * b * n + blockSize - 1) / blockSize;

    bias_batch_sum4_kernel_zero<8><<<gridSize, blockSize>>>(w, 8 * b * n, tmp_zero, tmp_first);

    // Phase 2

    blockSize = 128;
    gridSize = (8 * n + blockSize - 1) / blockSize;

    bias_batch_sum4_kernel_last<8, Mean><<<gridSize, blockSize>>>(b, n, w, h, 8 * n, tmp_first, y);

    cuda_check(cudaFree(tmp_zero));
    cuda_check(cudaFree(tmp_first));
}

void egblas_sbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, float* x, float* y) {
    egblas_sbias_batch_sum4_run<false>(b, n, w, h, x, y);
}

void egblas_dbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, double* x, double* y) {
    egblas_sbias_batch_sum4_run<false>(b, n, w, h, x, y);
}

void egblas_sbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, float* x, float* y) {
    egblas_sbias_batch_sum4_run<true>(b, n, w, h, x, y);
}

void egblas_dbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, double* x, double* y) {
    egblas_sbias_batch_sum4_run<true>(b, n, w, h, x, y);
}
