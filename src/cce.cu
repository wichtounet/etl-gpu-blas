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

template <typename T>
__global__ void cce_loss_kernel(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* y) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = logf(output[incx * index]) * labels[incx * index];
    }
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
void cce_loss_kernel_run(size_t n, const T* output, size_t incx, const T* labels, size_t incy, T* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cce_loss_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    cce_loss_kernel<T><<<gridSize, blockSize>>>(n, output, incx, labels, incy, y);

    cudaDeviceSynchronize();
}

template <typename T>
void cce_error_kernel_run(size_t n, size_t m, const T* output, const T* labels, T* y) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cce_loss_kernel<T>, 0, 0);

    int gridSize = (n + blockSize - 1) / blockSize;

    cce_error_kernel<T><<<gridSize, blockSize>>>(n, m, output, labels, y);

    cudaDeviceSynchronize();
}

float egblas_cce_sloss(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy) {
    float* temp;
    cuda_check(cudaMalloc((void**)&temp, n * sizeof(float)));

    cce_loss_kernel_run(n, output, incx, labels, incy, temp);

    float loss = thrust::reduce(thrust::device, temp, temp + n);

    cuda_check(cudaFree(temp));

    return alpha * loss;
}

double egblas_cce_dloss(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy) {
    double* temp;
    cuda_check(cudaMalloc((void**)&temp, n * sizeof(double)));

    cce_loss_kernel_run(n, output, incx, labels, incy, temp);

    double loss = thrust::reduce(thrust::device, temp, temp + n);

    cuda_check(cudaFree(temp));

    return alpha * loss;
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
