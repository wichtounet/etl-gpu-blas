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
__global__ void bias_batch_sum4_kernel_zero(size_t Last, size_t Limit, const T* __restrict__ x, T* __restrict__ y) {
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
__global__ void bias_batch_sum4_kernel_last(size_t B, size_t N, size_t W, size_t H, size_t Limit, const T* __restrict__ x, T* __restrict__ y) {
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

// This is the legacy version
// This is kept for further optimizations

template <bool Mean, typename T>
void egblas_sbias_batch_sum4_run(size_t b, size_t n, size_t w, size_t h, T* x, T* y) {
    const size_t big_size = b * n * w * sizeof(T) + b * n * sizeof(T);

    T* tmp;
    cuda_check(cudaMalloc((void**)&tmp, big_size));

    cudaMemset(tmp, 0, big_size);
    cudaMemset(y, 0, n * sizeof(T));

    T* tmp_zero  = tmp;
    T* tmp_first = tmp + b * n * w;

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

    cuda_check(cudaFree(tmp));
}

// The current algorithm is a two-phase algorithm
// First, we reduce the last two dimensions of the matrix
// Then, we reduce the first dimension
// Only the first reduction takes a significant amount of time

// Opportunities for further improvements in speed
// * Even though we are not expecting very large W/H, we could still probably
// do a multi-step reductions of of the first two dimensions in the same model
// as the full sum reduuction
// * We could probably profit from extra unrolling

// Reduce version

template <size_t blockSize, typename T>
__global__ void bias_batch_sum4_kernel_reduce(size_t lda, const T* __restrict__ x, T* __restrict__ y) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    const size_t tid = threadIdx.x;

    T mySum = 0.0;

    const T * p = &x[blockIdx.x * lda];

    size_t i = tid;

    while (i < lda) {
        mySum += p[i];

        i += blockSize;
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(y, shared_data);
}

template <typename T>
void invoke_reduce_kernel(size_t lda, const T* __restrict__ x, T* __restrict__ y, size_t blockSize, size_t gridSize) {
    int sharedSize = (blockSize <= 32) ? 64 * sizeof(T) : blockSize * sizeof(T);

    switch (blockSize) {
        case 512:
            bias_batch_sum4_kernel_reduce<512><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 256:
            bias_batch_sum4_kernel_reduce<256><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 128:
            bias_batch_sum4_kernel_reduce<128><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 64:
            bias_batch_sum4_kernel_reduce<64><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 32:
            bias_batch_sum4_kernel_reduce<32><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 16:
            bias_batch_sum4_kernel_reduce<16><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 8:
            bias_batch_sum4_kernel_reduce<8><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 4:
            bias_batch_sum4_kernel_reduce<4><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 2:
            bias_batch_sum4_kernel_reduce<2><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;

        case 1:
            bias_batch_sum4_kernel_reduce<1><<<gridSize, blockSize, sharedSize>>>(lda, x, y);
            break;
    }
}

template <bool Mean, typename T>
void egblas_sbias_batch_sum4_run_reduce(size_t b, size_t n, size_t w, size_t h, T* x, T* y) {
    T* tmp;
    cuda_check(cudaMalloc((void**)&tmp, b * n * sizeof(T)));

    cudaMemset(y, 0, n * sizeof(T));

    // Phase 0 (Bottleneck)

    size_t s = w * h;
    size_t baseBlockSize = 128;
    size_t blockSize= s < baseBlockSize * 2 ? nextPow2((s + 1) / 2) : baseBlockSize;
    size_t gridSize(b * n);

    invoke_reduce_kernel(w * h, x, tmp, blockSize, gridSize);

    // Phase 1 (Very quick)

    blockSize = 128;
    gridSize = (8 * n + blockSize - 1) / blockSize;

    bias_batch_sum4_kernel_last<8, Mean><<<gridSize, blockSize>>>(b, n, w, h, 8 * n, tmp, y);

    cuda_check(cudaFree(tmp));
}

void egblas_sbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, float* x, float* y) {
    egblas_sbias_batch_sum4_run_reduce<false>(b, n, w, h, x, y);
}

void egblas_dbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, double* x, double* y) {
    egblas_sbias_batch_sum4_run_reduce<false>(b, n, w, h, x, y);
}

void egblas_sbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, float* x, float* y) {
    egblas_sbias_batch_sum4_run_reduce<true>(b, n, w, h, x, y);
}

void egblas_dbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, double* x, double* y) {
    egblas_sbias_batch_sum4_run_reduce<true>(b, n, w, h, x, y);
}
