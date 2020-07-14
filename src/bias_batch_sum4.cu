//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
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

// Two-Pass Reduce version

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

// When W*H is very small, we do  a full reduction
// In the cases, the two-pass is not great because of the small W*H opportunities
// Full Reduce versioon

template <bool Mean, size_t blockSize, typename T>
__global__ void bias_batch_sum4_kernel_reduce_full_kernel_general(size_t lda1, size_t lda2, size_t limit, const T* __restrict__ x, T* __restrict__ y) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    const size_t ik = blockIdx.x; /// Index in k dimension

    const size_t tid = threadIdx.x;

    T mySum = 0.0;

    size_t i = tid;

    size_t ib = i / lda2;
    size_t ii = i % lda2;

    const T * p = x + ik * lda2;

    while (i < limit) {
        mySum += p[ib * lda1 + ii];

        i += blockSize;

        ib = i / lda2;
        ii = i % lda2;
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(y, shared_data);

    if (Mean) {
        if (tid == 0) {
            y[blockIdx.x] /= limit;
        }
    }
}

template <size_t Max, bool Mean, size_t blockSize, typename T>
__global__ void bias_batch_sum4_kernel_reduce_impl_small(size_t b, size_t n, size_t w, size_t h, size_t llimit, const T* __restrict__ x, T* __restrict__ y) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    const size_t ik = blockIdx.x; /// Index in k dimension

    const size_t tid = threadIdx.x;

    T mySum = 0.0;

    const size_t lda1 = n * w * h;
    const size_t lda2 = w * h;

    size_t i = tid;

    size_t ib = Max == 1 ? 0 : i / (lda2); // Index in b dimension
    size_t ii = Max == 1 ? i : i % (lda2); // Index in i dimension

    const T * p = x + ik * lda2;

    while (i < llimit) {
        mySum += p[ib * lda1 + ii];

        i += blockSize;
        ii += blockSize;

        if (ii >= lda2) {
            ++ib;
            ii = ii - lda2;
        }

        if /* compile-time */ (Max > 1) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }

        if /* compile-time */ (Max > 2) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }

        if /* compile-time */ (Max > 3) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }

        if /* compile-time */ (Max > 4) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(y, shared_data);

    if (Mean) {
        if (tid == 0) {
            y[blockIdx.x] /= llimit;
        }
    }
}

template <bool Mean, size_t blockSize, typename T>
void bias_batch_sum4_kernel_reduce_full(size_t b, size_t n, size_t w, size_t h, const T* __restrict__ x, T* __restrict__ y, size_t gridSize) {
    // Precompute some stuff on the CPU
    // Strangely, it seems slower for the small kernels to compute it themselves
    // So, we only pass everything to the general kernel
    const size_t lda1 = n * w * h;
    const size_t lda2 = w * h;
    const size_t limit = b * w * h;

    int sharedSize = blockSize * sizeof(T);

    if (blockSize <= w * h) {
        bias_batch_sum4_kernel_reduce_impl_small<1, Mean, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, x, y);
    } else if(blockSize <= 2 * w * h) {
        bias_batch_sum4_kernel_reduce_impl_small<2, Mean, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, x, y);
    } else if(blockSize <= 3 * w * h) {
        bias_batch_sum4_kernel_reduce_impl_small<3, Mean, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, x, y);
    } else if(blockSize <= 4 * w * h) {
        bias_batch_sum4_kernel_reduce_impl_small<4, Mean, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, x, y);
    } else if(blockSize <= 5 * w * h) {
        bias_batch_sum4_kernel_reduce_impl_small<5, Mean, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, x, y);
    } else {
        bias_batch_sum4_kernel_reduce_full_kernel_general<Mean, blockSize><<<gridSize, blockSize, sharedSize>>>(lda1, lda2, limit, x, y);
    }
}

template <bool Mean, typename T>
void invoke_reduce_kernel_full(size_t b, size_t n, size_t w, size_t h, const T* __restrict__ x, T* __restrict__ y, size_t blockSize, size_t gridSize) {
    switch (blockSize) {
        case 1024:
            bias_batch_sum4_kernel_reduce_full<Mean, 1024>(b, n, w, h, x, y, gridSize);
            break;

        case 512:
            bias_batch_sum4_kernel_reduce_full<Mean, 512>(b, n, w, h, x, y, gridSize);
            break;

        case 256:
            bias_batch_sum4_kernel_reduce_full<Mean, 256>(b, n, w, h, x, y, gridSize);
            break;

        case 128:
            bias_batch_sum4_kernel_reduce_full<Mean, 128>(b, n, w, h, x, y, gridSize);
            break;

        case 64:
            bias_batch_sum4_kernel_reduce_full<Mean, 64>(b, n, w, h, x, y, gridSize);
            break;

        case 32:
            bias_batch_sum4_kernel_reduce_full<Mean, 32>(b, n, w, h, x, y, gridSize);
            break;

        case 16:
            bias_batch_sum4_kernel_reduce_full<Mean, 16>(b, n, w, h, x, y, gridSize);
            break;

        case 8:
            bias_batch_sum4_kernel_reduce_full<Mean, 8>(b, n, w, h, x, y, gridSize);
            break;

        case 4:
            bias_batch_sum4_kernel_reduce_full<Mean, 4>(b, n, w, h, x, y, gridSize);
            break;

        case 2:
            bias_batch_sum4_kernel_reduce_full<Mean, 2>(b, n, w, h, x, y, gridSize);
            break;

        case 1:
            bias_batch_sum4_kernel_reduce_full<Mean, 1>(b, n, w, h, x, y, gridSize);
            break;
    }
}

template <bool Mean, typename T>
void egblas_sbias_batch_sum4_run_reduce_full(size_t b, size_t n, size_t w, size_t h, T* x, T* y) {
    size_t s = b * w * h;
    size_t baseBlockSize = 256;
    size_t blockSize= s < baseBlockSize * 2 ? nextPow2((s + 1) / 2) : baseBlockSize;
    size_t gridSize(n);

    invoke_reduce_kernel_full<Mean>(b, n, w, h, x, y, blockSize, gridSize);
}

// The actual API

void egblas_sbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, float* x, float* y) {
    if (w * h <= 256) {
        egblas_sbias_batch_sum4_run_reduce_full<false>(b, n, w, h, x, y);
    } else {
        egblas_sbias_batch_sum4_run_reduce<false>(b, n, w, h, x, y);
    }
}

void egblas_dbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, double* x, double* y) {
    if (w * h <= 256) {
        egblas_sbias_batch_sum4_run_reduce_full<false>(b, n, w, h, x, y);
    } else {
        egblas_sbias_batch_sum4_run_reduce<false>(b, n, w, h, x, y);
    }
}

void egblas_sbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, float* x, float* y) {
    if (w * h <= 256) {
        egblas_sbias_batch_sum4_run_reduce_full<true>(b, n, w, h, x, y);
    } else {
        egblas_sbias_batch_sum4_run_reduce<true>(b, n, w, h, x, y);
    }
}

void egblas_dbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, double* x, double* y) {
    if (w * h <= 256) {
        egblas_sbias_batch_sum4_run_reduce_full<true>(b, n, w, h, x, y);
    } else {
        egblas_sbias_batch_sum4_run_reduce<true>(b, n, w, h, x, y);
    }
}

// bias_batch_var4 version

template <size_t blockSize, typename T>
__global__ void bias_batch_var4_kernel_reduce_full_kernel_general(size_t lda1, size_t lda2, size_t limit, const T* __restrict__ a, const T* __restrict__ c, T* __restrict__ y) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    const size_t ik = blockIdx.x; /// Index in k dimension

    const size_t tid = threadIdx.x;

    T mySum = 0.0;

    size_t i = tid;

    size_t ib = i / lda2;
    size_t ii = i % lda2;

    const T * p = a + ik * lda2;

    while (i < limit) {
        mySum += (p[ib * lda1 + ii] - c[blockIdx.x]) * (p[ib * lda1 + ii] - c[blockIdx.x]);

        i += blockSize;

        ib = i / lda2;
        ii = i % lda2;
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(y, shared_data);

    if (tid == 0) {
        y[blockIdx.x] /= limit;
    }
}

template <size_t Max, size_t blockSize, typename T>
__global__ void bias_batch_var4_kernel_reduce_impl_small(size_t b, size_t n, size_t w, size_t h, size_t llimit, const T* __restrict__ a, const T* __restrict__ c, T* __restrict__ y) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    const size_t ik = blockIdx.x; /// Index in k dimension

    const size_t tid = threadIdx.x;

    T mySum = 0.0;

    const size_t lda1 = n * w * h;
    const size_t lda2 = w * h;

    size_t i = tid;

    size_t ib = Max == 1 ? 0 : i / (lda2); // Index in b dimension
    size_t ii = Max == 1 ? i : i % (lda2); // Index in i dimension

    const T * p = a + ik * lda2;

    while (i < llimit) {
        mySum += (p[ib * lda1 + ii] - c[blockIdx.x]) * (p[ib * lda1 + ii] - c[blockIdx.x]);

        i += blockSize;
        ii += blockSize;

        if (ii >= lda2) {
            ++ib;
            ii = ii - lda2;
        }

        if /* compile-time */ (Max > 1) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }

        if /* compile-time */ (Max > 2) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }

        if /* compile-time */ (Max > 3) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }

        if /* compile-time */ (Max > 4) {
            if (ii >= lda2) {
                ++ib;
                ii = ii - lda2;
            }
        }
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(y, shared_data);

    if (tid == 0) {
        y[blockIdx.x] /= llimit;
    }
}

template <size_t blockSize, typename T>
void bias_batch_var4_kernel_reduce_full(size_t b, size_t n, size_t w, size_t h, const T* __restrict__ a, const T* __restrict__ c, T* __restrict__ y, size_t gridSize) {
    // Precompute some stuff on the CPU
    // Strangely, it seems slower for the small kernels to compute it themselves
    // So, we only pass everything to the general kernel
    const size_t lda1 = n * w * h;
    const size_t lda2 = w * h;
    const size_t limit = b * w * h;

    int sharedSize = blockSize * sizeof(T);

    if (blockSize <= w * h) {
        bias_batch_var4_kernel_reduce_impl_small<1, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, a, c, y);
    } else if(blockSize <= 2 * w * h) {
        bias_batch_var4_kernel_reduce_impl_small<2, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, a, c, y);
    } else if(blockSize <= 3 * w * h) {
        bias_batch_var4_kernel_reduce_impl_small<3, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, a, c, y);
    } else if(blockSize <= 4 * w * h) {
        bias_batch_var4_kernel_reduce_impl_small<4, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, a, c, y);
    } else if(blockSize <= 5 * w * h) {
        bias_batch_var4_kernel_reduce_impl_small<5, blockSize><<<gridSize, blockSize, sharedSize>>>(b, n, w, h, limit, a, c, y);
    } else {
        bias_batch_var4_kernel_reduce_full_kernel_general<blockSize><<<gridSize, blockSize, sharedSize>>>(lda1, lda2, limit, a, c, y);
    }
}
template <typename T>
void invoke_reduce_kernel_full_var(size_t b, size_t n, size_t w, size_t h, const T* __restrict__ a, const T* __restrict__ c, T* __restrict__ y, size_t blockSize, size_t gridSize) {
    switch (blockSize) {
        case 1024:
            bias_batch_var4_kernel_reduce_full<1024>(b, n, w, h, a, c, y, gridSize);
            break;

        case 512:
            bias_batch_var4_kernel_reduce_full<512>(b, n, w, h, a, c, y, gridSize);
            break;

        case 256:
            bias_batch_var4_kernel_reduce_full<256>(b, n, w, h, a, c, y, gridSize);
            break;

        case 128:
            bias_batch_var4_kernel_reduce_full<128>(b, n, w, h, a, c, y, gridSize);
            break;

        case 64:
            bias_batch_var4_kernel_reduce_full<64>(b, n, w, h, a, c, y, gridSize);
            break;

        case 32:
            bias_batch_var4_kernel_reduce_full<32>(b, n, w, h, a, c, y, gridSize);
            break;

        case 16:
            bias_batch_var4_kernel_reduce_full<16>(b, n, w, h, a, c, y, gridSize);
            break;

        case 8:
            bias_batch_var4_kernel_reduce_full<8>(b, n, w, h, a, c, y, gridSize);
            break;

        case 4:
            bias_batch_var4_kernel_reduce_full<4>(b, n, w, h, a, c, y, gridSize);
            break;

        case 2:
            bias_batch_var4_kernel_reduce_full<2>(b, n, w, h, a, c, y, gridSize);
            break;

        case 1:
            bias_batch_var4_kernel_reduce_full<1>(b, n, w, h, a, c, y, gridSize);
            break;
    }
}

template <typename T>
void egblas_sbias_batch_var4_run_reduce_full(size_t b, size_t n, size_t w, size_t h, T* a, T* c, T* y) {
    size_t s = b * w * h;
    size_t baseBlockSize = 256;
    size_t blockSize= s < baseBlockSize * 2 ? nextPow2((s + 1) / 2) : baseBlockSize;
    size_t gridSize(n);

    invoke_reduce_kernel_full_var(b, n, w, h, a, c, y, blockSize, gridSize);
}

// Performance Note: bias_batch_var could be optimized better for large matrices

void egblas_sbias_batch_var4(size_t b, size_t n, size_t w, size_t h, float* a, float* c, float* y) {
    egblas_sbias_batch_var4_run_reduce_full(b, n, w, h, a, c, y);
}

void egblas_dbias_batch_var4(size_t b, size_t n, size_t w, size_t h, double* a, double* c, double* y){
    egblas_sbias_batch_var4_run_reduce_full(b, n, w, h, a, c, y);
}
