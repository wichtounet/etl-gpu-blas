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
__global__ void sum_kernel(size_t n, const T* input, size_t incx, T* output) {
    extern __shared__ __align__(sizeof(T)) volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of durection,
    // reading from global memory and writing to shared memory

    T mySum = 0;

    while (i < n) {
        mySum += input[i * incx];

        if (i + blockSize < n) {
            mySum += input[(i + blockSize) * incx];
        }

        i += gridSize;
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(output, shared_data, mySum);
}

template <typename T>
void invoke_sum_kernel(size_t n, const T* input, size_t incx, T* output, size_t numThreads, size_t numBlocks) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            sum_kernel<T, 512><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 256:
            sum_kernel<T, 256><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 128:
            sum_kernel<T, 128><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 64:
            sum_kernel<T,  64><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 32:
            sum_kernel<T,  32><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 16:
            sum_kernel<T,  16><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 8:
            sum_kernel<T,   8><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 4:
            sum_kernel<T,   4><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 2:
            sum_kernel<T,   2><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;

        case 1:
            sum_kernel<T,   1><<<numBlocks, numThreads, sharedSize>>>(n, input, incx, output);
            break;
    }
}

template <typename T>
T sum_kernel_run(size_t n, const T* input, size_t incx) {
    T result = 0;

    const size_t cpu_threshold = 1024;

    if (n <= cpu_threshold && incx == 1) {
        if (n > 1) {
            T* host_data = new T[n];

            cuda_check(cudaMemcpy(host_data, input, n * sizeof(T), cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < n; i++) {
                result += host_data[i];
            }

            delete[] host_data;
        } else {
            cuda_check(cudaMemcpy(&result, input, 1 * sizeof(T), cudaMemcpyDeviceToHost));
        }

        return result;
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

    invoke_sum_kernel<T>(n, input, incx, y_gpu_2, numThreads, numBlocks);

    size_t s = numBlocks;

    // Run the following reductions on GPU

    while(s > cpu_threshold){
        // Compute again the configuration of the reduction kernel
        numThreads = n < maxThreads * 2 ? nextPow2((n + 1) / 2) : maxThreads;
        numBlocks  = std::min((n + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

        cuda_check(cudaMemcpy(y_gpu_1, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToDevice));

        invoke_sum_kernel<T>(s, y_gpu_1, 1, y_gpu_2, numThreads, numBlocks);

        s = (s + numThreads * 2 - 1) / (numThreads * 2);
    }

    if(s > 1){
        T* host_data = new T[s];

        cuda_check(cudaMemcpy(host_data, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < s; i++) {
            result += host_data[i];
        }

        delete[] host_data;
    } else {
        cuda_check(cudaMemcpy(&result, y_gpu_2, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    }

    cuda_check(cudaFree(y_gpu_1));
    cuda_check(cudaFree(y_gpu_2));

    return result;
}

float egblas_ssum(float* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_ssum");
    egblas_unused(s);

    return sum_kernel_run(n, x, s);
}

double egblas_dsum(double* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dsum");
    egblas_unused(s);

    return sum_kernel_run(n, x, s);
}

// Complex sums are done with thrust

struct single_complex_add {
    __device__ cuComplex operator()(cuComplex x, cuComplex y) {
        return cuCaddf(x, y);
    }
};

struct double_complex_add {
    __device__ cuDoubleComplex operator()(cuDoubleComplex x, cuDoubleComplex y) {
        return cuCadd(x, y);
    }
};

cuComplex egblas_csum(cuComplex* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_csum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n, make_cuComplex(0, 0), single_complex_add());
}

cuDoubleComplex egblas_zsum(cuDoubleComplex* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_zsum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n, make_cuDoubleComplex(0, 0), double_complex_add());
}
