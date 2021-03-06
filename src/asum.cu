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

template <class T, size_t blockSize>
__global__ void asum_kernel(size_t n, const T* input, size_t incx, T* output) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + tid;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of reduction,
    // reading from global memory and writing to shared memory

    T mySum = 0;

    while (i < n) {
        mySum += fabsf(input[i * incx]);

        if (i + blockSize < n) {
            mySum += fabsf(input[(i + blockSize) * incx]);
        }

        i += gridSize;
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(output, shared_data);
}

template <class T, size_t blockSize>
__global__ void asum_kernel1(size_t n, const T* input, T* output) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + tid;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of reduction,
    // reading from global memory and writing to shared memory

    T mySum = 0;

    while (i < n) {
        mySum += fabsf(input[i]);

        if (i + blockSize < n) {
            mySum += fabsf(input[i + blockSize]);
        }

        i += gridSize;
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(output, shared_data);
}

template <typename T>
void invoke_asum_kernel(size_t n, const T* input, size_t incx, T* output, size_t numThreads, size_t numBlocks) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            if (incx == 1) {
                asum_kernel1<T, 512><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 512><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 256:
            if (incx == 1) {
                asum_kernel1<T, 256><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 256><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 128:
            if (incx == 1) {
                asum_kernel1<T, 128><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 128><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 64:
            if (incx == 1) {
                asum_kernel1<T, 64><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 64><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 32:
            if (incx == 1) {
                asum_kernel1<T, 32><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 32><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 16:
            if (incx == 1) {
                asum_kernel1<T, 16><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 16><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 8:
            if (incx == 1) {
                asum_kernel1<T, 8><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 8><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 4:
            if (incx == 1) {
                asum_kernel1<T, 4><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 4><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 2:
            if (incx == 1) {
                asum_kernel1<T, 2><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 2><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 1:
            if (incx == 1) {
                asum_kernel1<T, 1><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                asum_kernel<T, 1><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;
    }
}

template <typename T>
T asum_kernel_run(size_t n, const T* input, size_t incx) {
    T result = 0;

    const size_t cpu_threshold = 1024;

    if (n <= cpu_threshold && incx == 1) {
        T* host_data = new T[n];

        cuda_check(cudaMemcpy(host_data, input, n * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < n; i++) {
            result += fabsf(host_data[i]);
        }

        delete[] host_data;

        return result;
    }

    const size_t maxThreads    = 512;
    const size_t maxBlocks     = 64;

    // Compute the launch configuration of the kernel
    size_t numThreads = n < maxThreads * 2 ? nextPow2((n + 1) / 2) : maxThreads;
    size_t numBlocks  = std::min((n + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

    // Allocate memory on the device

    T* tmp_gpu;
    cuda_check(cudaMalloc((void**)&tmp_gpu, numBlocks * sizeof(T)));

    // Run the first reduction on GPU

    invoke_asum_kernel<T>(n, input, incx, tmp_gpu, numThreads, numBlocks);

    size_t s = numBlocks;

    // Run the following reductions on GPU

    while(s > cpu_threshold){
        // Compute again the configuration of the reduction kernel
        numThreads = s < maxThreads * 2 ? nextPow2((s + 1) / 2) : maxThreads;
        numBlocks  = std::min((s + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

        invoke_asum_kernel<T>(s, tmp_gpu, 1, tmp_gpu, numThreads, numBlocks);

        s = (s + numThreads * 2 - 1) / (numThreads * 2);
    }

    if(s > 1){
        T* host_data = new T[s];

        cuda_check(cudaMemcpy(host_data, tmp_gpu, s * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < s; i++) {
            result += host_data[i];
        }

        delete[] host_data;
    } else {
        cuda_check(cudaMemcpy(&result, tmp_gpu, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    }

    cuda_check(cudaFree(tmp_gpu));

    return result;
}

float egblas_sasum(float* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_sasum");
    egblas_unused(s);

    return asum_kernel_run(n, x, s);
}

double egblas_dasum(double* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dasum");
    egblas_unused(s);

    return asum_kernel_run(n, x, s);
}
