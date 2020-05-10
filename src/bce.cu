//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/bce.hpp"
#include "egblas/cuda_check.hpp"
#include "egblas/utils.hpp"

#include "sum_reduce.hpp"

template<typename T>
__device__ T bce_loss(T output, T label){
    return (logf(output) * label) + ((T(1) - label) * logf(T(1) - output));
}

template<typename T>
__device__ T bce_error(T output, T label){
    return fabsf(label - output);
}

template <class T, size_t blockSize, bool Reduce>
__global__ void bce_loss_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* r_output) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of reduction,
    // reading from global memory and writing to shared memory

    T mySum = 0;

    if (Reduce) {
        // In case of reductions, this a simple sum (labels are ignored)

        while (i < n) {
            mySum += output[i * incx];

            if (i + blockSize < n) {
                mySum += output[(i + blockSize) * incx];
            }

            i += gridSize;
        }
    } else {
        // In the basic case, perform reduction and BCE loss

        while (i < n) {
            mySum += bce_loss(output[i * incx], labels[i * incy]);

            if (i + blockSize < n) {
                mySum += bce_loss(output[(i + blockSize) * incx], labels[(i + blockSize) * incx]);
            }

            i += gridSize;
        }
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(r_output, shared_data, mySum);
}

template <class T, size_t blockSize, bool Reduce>
__global__ void bce_error_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* r_output) {
    extern __shared__ volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of reduction,
    // reading from global memory and writing to shared memory

    T mySum = 0;

    if (Reduce) {
        // In case of reductions, this a simple sum (labels are ignored)

        while (i < n) {
            mySum += output[i * incx];

            if (i + blockSize < n) {
                mySum += output[(i + blockSize) * incx];
            }

            i += gridSize;
        }
    } else {
        // In the basic case, perform reduction and BCE error

        while (i < n) {
            mySum += bce_error(output[i * incx], labels[i * incy]);

            if (i + blockSize < n) {
                mySum += bce_error(output[(i + blockSize) * incx], labels[(i + blockSize) * incx]);
            }

            i += gridSize;
        }
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(r_output, shared_data, mySum);
}

template <typename T, bool Reduce>
void invoke_bce_loss_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* r_output, size_t numThreads, size_t numBlocks) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            bce_loss_kernel<T, 512, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 256:
            bce_loss_kernel<T, 256, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 128:
            bce_loss_kernel<T, 128, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 64:
            bce_loss_kernel<T,  64, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 32:
            bce_loss_kernel<T,  32, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 16:
            bce_loss_kernel<T,  16, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 8:
            bce_loss_kernel<T,   8, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 4:
            bce_loss_kernel<T,   4, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 2:
            bce_loss_kernel<T,   2, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 1:
            bce_loss_kernel<T,   1, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;
    }
}

template <typename T, bool Reduce>
void invoke_bce_error_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* r_output, size_t numThreads, size_t numBlocks) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            bce_error_kernel<T, 512, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 256:
            bce_error_kernel<T, 256, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 128:
            bce_error_kernel<T, 128, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 64:
            bce_error_kernel<T,  64, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 32:
            bce_error_kernel<T,  32, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 16:
            bce_error_kernel<T,  16, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 8:
            bce_error_kernel<T,   8, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 4:
            bce_error_kernel<T,   4, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 2:
            bce_error_kernel<T,   2, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 1:
            bce_error_kernel<T,   1, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;
    }
}

template <bool Loss, typename T>
T bce_kernel_run(size_t n, const T* output, size_t incx, const T* labels, size_t incy) {
    T result = 0;

    const size_t cpu_threshold = Loss ? 128 : 1024;

    if (Loss && n < cpu_threshold && incx == 1 && incy == 1) {
        T* host_output = new T[n];
        T* host_labels = new T[n];

        cuda_check(cudaMemcpy(host_output, output, n * sizeof(T), cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(host_labels, labels, n * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < n; i++) {
            result += (logf(host_output[i]) * host_labels[i]) + ((T(1) - host_labels[i]) * logf(T(1) - host_output[i]));
        }

        delete[] host_output;
        delete[] host_labels;
    }

    if (!Loss && n < cpu_threshold && incx == 1 && incy == 1) {
        T* host_output = new T[n];
        T* host_labels = new T[n];

        cuda_check(cudaMemcpy(host_output, output, n * sizeof(T), cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(host_labels, labels, n * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < n; i++) {
            result += fabsf(host_labels[i] - host_output[i]);
        }

        delete[] host_output;
        delete[] host_labels;

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

    if (Loss) {
        invoke_bce_loss_kernel<T, false>(n, output, incx, labels, incy, y_gpu_2, numThreads, numBlocks);
    } else {
        invoke_bce_error_kernel<T, false>(n, output, incx, labels, incy, y_gpu_2, numThreads, numBlocks);
    }

    size_t s = numBlocks;

    // Run the following reductions on GPU

    while(s > cpu_threshold){
        // Compute again the configuration of the reduction kernel
        numThreads = n < maxThreads * 2 ? nextPow2((n + 1) / 2) : maxThreads;
        numBlocks  = std::min((n + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

        cuda_check(cudaMemcpy(y_gpu_1, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToDevice));

        if (Loss) {
            invoke_bce_loss_kernel<T, true>(s, y_gpu_1, 1, y_gpu_1, 1, y_gpu_2, numThreads, numBlocks);
        } else {
            invoke_bce_error_kernel<T, true>(s, y_gpu_1, 1, y_gpu_1, 1, y_gpu_2, numThreads, numBlocks);
        }

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

float egblas_bce_sloss(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy) {
    return alpha * bce_kernel_run<true>(n, output, incx, labels, incy);
}

double egblas_bce_dloss(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy) {
    return alpha * bce_kernel_run<true>(n, output, incx, labels, incy);
}

float egblas_bce_serror(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy) {
    return alpha * bce_kernel_run<false>(n, output, incx, labels, incy);
}

double egblas_bce_derror(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy) {
    return alpha * bce_kernel_run<false>(n, output, incx, labels, incy);
}
