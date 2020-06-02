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
#include "egblas/max_reduce.hpp"
#include "egblas/cuda_check.hpp"

#include "max_reduce.hpp"

template <class T, size_t blockSize>
__global__ void max_kernel(size_t n, const T* input, size_t incx, T* output) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + tid;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of reduction,
    // reading from global memory and writing to shared memory

    T myMax = 0;

    if (i < n) {
        myMax = input[i * incx];

        while (i < n) {
            myMax = input[i * incx];

            if (i + blockSize < n) {
                myMax = max(input[(i + blockSize) * incx], myMax);
            }

            i += gridSize;
        }
    }

    shared_data[tid] = myMax;

    __syncthreads();

    max_reduce_impl<T, blockSize>(output, shared_data);
}

template <class T, size_t blockSize>
__global__ void max_kernel1(size_t n, const T* input, T* output) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + tid;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of reduction,
    // reading from global memory and writing to shared memory

    T myMax = 0;

    if (i < n) {
        myMax = input[i];

        while (i < n) {
            myMax = max(input[i], myMax);

            if (i + blockSize < n) {
                myMax = max(input[i + blockSize], myMax);
            }

            i += gridSize;
        }
    }

    shared_data[tid] = myMax;

    __syncthreads();

    max_reduce_impl<T, blockSize>(output, shared_data);
}

template <typename T>
void invoke_max_kernel(size_t n, const T* input, size_t incx, T* output, size_t numThreads, size_t numBlocks) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            if (incx == 1) {
                max_kernel1<T, 512><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 512><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 256:
            if (incx == 1) {
                max_kernel1<T, 256><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 256><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 128:
            if (incx == 1) {
                max_kernel1<T, 128><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 128><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 64:
            if (incx == 1) {
                max_kernel1<T, 64><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 64><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 32:
            if (incx == 1) {
                max_kernel1<T, 32><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 32><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 16:
            if (incx == 1) {
                max_kernel1<T, 16><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 16><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 8:
            if (incx == 1) {
                max_kernel1<T, 8><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 8><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 4:
            if (incx == 1) {
                max_kernel1<T, 4><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 4><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 2:
            if (incx == 1) {
                max_kernel1<T, 2><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 2><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;

        case 1:
            if (incx == 1) {
                max_kernel1<T, 1><<<numBlocks, numThreads, sharedSize>>>(n, input, output);
            } else {
                max_kernel<T, 1><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            }
            break;
    }
}

template <typename T>
T max_kernel_run(size_t n, const T* input, size_t incx) {
    T result = 0;

    const size_t cpu_threshold = 1024;

    if (n <= cpu_threshold && incx == 1) {
        if (n > 1) {
            T* host_data = new T[n];

            cuda_check(cudaMemcpy(host_data, input, n * sizeof(T), cudaMemcpyDeviceToHost));

            result = host_data[0];

            for (size_t i = 1; i < n; i++) {
                result = host_data[i] > result ? host_data[i] : result;
            }

            delete[] host_data;
        } else {
            cuda_check(cudaMemcpy(&result, input, 1 * sizeof(T), cudaMemcpyDeviceToHost));
        }

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

    invoke_max_kernel<T>(n, input, incx, tmp_gpu, numThreads, numBlocks);

    size_t s = numBlocks;

    // Run the following reductions on GPU

    while(s > cpu_threshold){
        // Compute again the configuration of the reduction kernel
        numThreads = s < maxThreads * 2 ? nextPow2((s + 1) / 2) : maxThreads;
        numBlocks  = std::min((s + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

        invoke_max_kernel<T>(s, tmp_gpu, 1, tmp_gpu, numThreads, numBlocks);

        s = (s + numThreads * 2 - 1) / (numThreads * 2);
    }

    if(s > 1){
        T* host_data = new T[s];

        cuda_check(cudaMemcpy(host_data, tmp_gpu, s * sizeof(T), cudaMemcpyDeviceToHost));

        result = host_data[0];

        for (size_t i = 1; i < s; i++) {
            result = host_data[i] > result ? host_data[i] : result;
        }

        delete[] host_data;
    } else {
        cuda_check(cudaMemcpy(&result, tmp_gpu, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    }

    cuda_check(cudaFree(tmp_gpu));

    return result;
}

float egblas_smax(float* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_smax");
    egblas_unused(s);

    return max_kernel_run(n, x, s);
}

double egblas_dmax(double* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_smax");
    egblas_unused(s);

    return max_kernel_run(n, x, s);
}
