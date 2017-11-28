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
#include "egblas/stddev.hpp"
#include "egblas/mean.hpp"
#include "egblas/cuda_check.hpp"

#include "sum_reduce.hpp"

template <class T, size_t blockSize, bool Reduce>
__global__ void stddev_kernel(size_t n, const T* input, size_t incx, T* output, T mean) {
    extern __shared__ __align__(sizeof(T)) volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of durection,
    // reading from global memory and writing to shared memory

    T mySum = 0;

    if (Reduce) {
        // In case of reductions, this a simple sum

        while (i < n) {
            mySum += input[i * incx];

            if (i + blockSize < n) {
                mySum += input[(i + blockSize) * incx];
            }

            i += gridSize;
        }
    } else {
        // In the basic case, compute the standard deviation

        while (i < n) {
            mySum += (input[i * incx] - mean) * (input[i * incx] - mean);

            if (i + blockSize < n) {
                mySum += (input[(i + blockSize) * incx] - mean) * (input[(i + blockSize) * incx] - mean);
            }

            i += gridSize;
        }
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(output, shared_data, mySum);
}

template <typename T, bool Reduce>
void invoke_stddev_kernel(size_t n, const T* input, size_t incx, T* output, size_t numThreads, size_t numBlocks, T mean) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            stddev_kernel<T, 512, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 256:
            stddev_kernel<T, 256, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 128:
            stddev_kernel<T, 128, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 64:
            stddev_kernel<T,  64, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 32:
            stddev_kernel<T,  32, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 16:
            stddev_kernel<T,  16, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 8:
            stddev_kernel<T,   8, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 4:
            stddev_kernel<T,   4, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 2:
            stddev_kernel<T,   2, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;

        case 1:
            stddev_kernel<T,   1, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output, mean);
            break;
    }
}

template <typename T>
T stddev_kernel_run(size_t n, const T* input, size_t incx, T mean) {
    T result = 0;

    const size_t cpu_threshold = 1024;

    if (n <= cpu_threshold && incx == 1) {
        T* host_data = new T[n];

        cuda_check(cudaMemcpy(host_data, input, n * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < n; i++) {
            result += (host_data[i] - mean) * (host_data[i] - mean);
        }

        delete[] host_data;

        return sqrt(result / T(n));
    }

    const size_t maxThreads    = 256;
    const size_t maxBlocks     = 64;

    // Compute the launch configuration of the kernel
    size_t numThreads = n < maxThreads * 2 ? nextPow2((n + 1) / 2) : maxThreads;
    size_t numBlocks  = std::min((n + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

    // Allocate memory on the device

    T* y_gpu_1;
    T* y_gpu_2;
    cuda_check(cudaMalloc((void**)&y_gpu_1, numBlocks * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu_2, numBlocks * sizeof(T)));

    cudaMemset(y_gpu_1, 0, numBlocks * sizeof(T));
    cudaMemset(y_gpu_2, 0, numBlocks * sizeof(T));

    // Run the first reduction on GPU

    invoke_stddev_kernel<T, false>(n, input, incx, y_gpu_2, numThreads, numBlocks, mean);

    size_t s = numBlocks;

    // Run the following reductions on GPU

    while(s > cpu_threshold){
        // Compute again the configuration of the reduction kernel
        numThreads = n < maxThreads * 2 ? nextPow2((n + 1) / 2) : maxThreads;
        numBlocks  = std::min((n + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

        cuda_check(cudaMemcpy(y_gpu_1, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToDevice));

        invoke_stddev_kernel<T, true>(s, y_gpu_1, 1, y_gpu_2, numThreads, numBlocks, mean);

        s = (s + numThreads * 2 - 1) / (numThreads * 2);
    }

    // Finish the reduction on CPU

    if(s > 1){
        // Reduce several elements

        T* host_data = new T[s];

        cuda_check(cudaMemcpy(host_data, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < s; i++) {
            result += host_data[i];
        }

        delete[] host_data;
    } else {
        // Copy the final result directly to the result
        cuda_check(cudaMemcpy(&result, y_gpu_2, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    }

    cuda_check(cudaFree(y_gpu_1));
    cuda_check(cudaFree(y_gpu_2));

    return std::sqrt(result / n);
}

float egblas_sstddev_mean(float* x, size_t n, size_t s, float mean) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_sstddev");
    egblas_unused(s);

    return stddev_kernel_run(n, x, s, mean);
}

double egblas_dstddev_mean(double* x, size_t n, size_t s, double mean) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dstddev");
    egblas_unused(s);

    return stddev_kernel_run(n, x, s, mean);
}

float egblas_sstddev(float* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_sstddev");
    egblas_unused(s);

    return egblas_sstddev_mean(x, n, s, egblas_smean(x, n, s));
}

double egblas_dstddev(double* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dstddev");
    egblas_unused(s);

    return egblas_dstddev_mean(x, n, s, egblas_dmean(x, n, s));
}
