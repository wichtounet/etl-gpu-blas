//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/cce.hpp"
#include "egblas/cuda_check.hpp"
#include "egblas/utils.hpp"

#include "sum_reduce.hpp"

template<typename T>
__device__ T cce_loss(T output, T label){
    return logf(output) * label;
}

/*template <typename T>*/
/*__global__ void cce_loss_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* y) {*/
    /*auto index  = threadIdx.x + blockIdx.x * blockDim.x;*/
    /*auto stride = blockDim.x * gridDim.x;*/

    /*for (; index < n; index += stride) {*/
        /*y[incy * index] = cce_loss(output[incx * index], labels[incx * index]);*/
    /*}*/
/*}*/

template <class T, size_t blockSize, bool Reduce>
__global__ void cce_loss_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* r_output) {
    extern __shared__ __align__(sizeof(T)) volatile unsigned char shared_data_raw[];

    volatile T* shared_data = reinterpret_cast<volatile T*>(shared_data_raw);

    size_t tid      = threadIdx.x;
    size_t i        = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;

    // Perform first level of durection,
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
        // In the basic case, perform reduction and loss

        while (i < n) {
            mySum += cce_loss(output[i * incx], labels[i * incy]);

            if (i + blockSize < n) {
                mySum += cce_loss(output[(i + blockSize) * incx], labels[(i + blockSize) * incx]);
            }

            i += gridSize;
        }
    }

    shared_data[tid] = mySum;

    __syncthreads();

    sum_reduce_impl<T, blockSize>(r_output, shared_data, mySum);
}

template <typename T, bool Reduce>
void invoke_cce_loss_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* r_output, size_t numThreads, size_t numBlocks) {
    int sharedSize = (numThreads <= 32) ? 64 * sizeof(T) : numThreads * sizeof(T);

    switch (numThreads) {
        case 512:
            cce_loss_kernel<T, 512, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 256:
            cce_loss_kernel<T, 256, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 128:
            cce_loss_kernel<T, 128, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 64:
            cce_loss_kernel<T,  64, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 32:
            cce_loss_kernel<T,  32, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 16:
            cce_loss_kernel<T,  16, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 8:
            cce_loss_kernel<T,   8, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 4:
            cce_loss_kernel<T,   4, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 2:
            cce_loss_kernel<T,   2, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;

        case 1:
            cce_loss_kernel<T,   1, Reduce><<<numBlocks, numThreads, sharedSize>>>(n, output, incx, labels, incy, r_output);
            break;
    }
}

template <typename T>
T cce_loss_kernel_run(size_t n, const T* output, size_t incx, const T* labels, size_t incy) {
    T result = 0;

    const size_t cpu_threshold = 1024;

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

    invoke_cce_loss_kernel<T, false>(n, output, incx, labels, incy, y_gpu_2, numThreads, numBlocks);

    size_t s = numBlocks;

    // Run the following reductions on GPU

    while(s > cpu_threshold){
        // Compute again the configuration of the reduction kernel
        numThreads = n < maxThreads * 2 ? nextPow2((n + 1) / 2) : maxThreads;
        numBlocks  = std::min((n + numThreads * 2 - 1) / (numThreads * 2), maxBlocks);

        cuda_check(cudaMemcpy(y_gpu_1, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToDevice));

        invoke_cce_loss_kernel<T, false>(s, y_gpu_1, 1, y_gpu_1, 1, y_gpu_2, numThreads, numBlocks);

        s = (s + numThreads * 2 - 1) / (numThreads * 2);
    }

    cudaDeviceSynchronize();

    bool need_read_back = true;

    if(s > 1){
        T* host_data = new T[s];

        cuda_check(cudaMemcpy(host_data, y_gpu_2, s * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < s; i++) {
            result += host_data[i];
        }

        need_read_back = false;

        delete[] host_data;
    }

    if(need_read_back){
        cuda_check(cudaMemcpy(&result, y_gpu_2, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    }

    cuda_check(cudaFree(y_gpu_1));
    cuda_check(cudaFree(y_gpu_2));

    return result;
}

template <typename T>
__global__ void cce_error_kernel(size_t n, size_t m, const T* output, const T* labels, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto i = index;

        int max_l = 0;
        int max_o = 0;

        // Compute the max for argmax

        for (size_t j = 1; j < m; ++j) {
            if (labels[i * m + j] > labels[i * m + max_l]) {
                max_l = j;
            }

            if (output[i * m + j] > output[i * m + max_o]) {
                max_o = j;
            }
        }

        // Compute the final value

        y[i] = fmin(abs(max_l - max_o), T(1.0));
    }
}

template <typename T>
void cce_error_kernel_run(size_t n, size_t m, const T* output, const T* labels, T* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cce_error_kernel<T>, 0, 0);

    int gridSize = (n + blockSize - 1) / blockSize;

    cce_error_kernel<T><<<gridSize, blockSize>>>(n, m, output, labels, y);

    cudaDeviceSynchronize();
}

float egblas_cce_sloss(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy) {
    return alpha * cce_loss_kernel_run(n, output, incx, labels, incy);
}

double egblas_cce_dloss(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy) {
    return alpha * cce_loss_kernel_run(n, output, incx, labels, incy);
}

float egblas_cce_serror(size_t n, size_t m, float alpha, const float* output, const float* labels) {
    float* temp;
    cuda_check(cudaMalloc((void**)&temp, n * sizeof(float)));

    cce_error_kernel_run(n, m, output, labels, temp);

    float loss = thrust::reduce(thrust::device, temp, temp + n);

    cuda_check(cudaFree(temp));

    return alpha * loss * 1.0f;
}

double egblas_cce_derror(size_t n, size_t m, double alpha, const double* output, const double* labels) {
    double* temp;
    cuda_check(cudaMalloc((void**)&temp, n * sizeof(double)));

    cce_error_kernel_run(n, m, output, labels, temp);

    double loss = thrust::reduce(thrust::device, temp, temp + n);

    cuda_check(cudaFree(temp));

    return alpha * loss;
}
