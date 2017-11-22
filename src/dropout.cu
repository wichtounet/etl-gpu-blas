//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <random>
#include <iostream>

#include "curand_kernel.h"

#include "egblas/dropout.hpp"

#include "egblas/cuda_check.hpp"

__global__ void setup_kernel(curandState* states, size_t seed) {
    int id = threadIdx.x + blockIdx.x * 64;

    curand_init(seed, id, 0, &states[id]);
}

template <typename T>
__global__ void dropout_kernel(curandState* states, size_t n, T p, T alpha, T* y, size_t incy) {
    auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    T r;

    // Copy state to local memory for efficiency
    auto local_state = states[base_index];

    for (auto index = base_index; index < n; index += stride) {
        r = curand_uniform(&local_state);

        if(r < p){
            y[incy * index] = T(0);
        } else {
            y[incy * index] = alpha * T(1);
        }
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

template <typename T>
__global__ void dropout_kernel1(curandState* states, size_t n, T p, T* y, size_t incy) {
    auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    T r;

    //Copy state to local memory for efficiency
    auto local_state = states[base_index];

    for (auto index = base_index; index < n; index += stride) {
        r = curand_uniform(&local_state);

        if(r < p){
            y[incy * index] = T(0);
        } else {
            y[incy * index] = T(1);
        }
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

template <typename T>
__global__ void dropout_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <typename T>
void dropout_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dropout_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    dropout_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_sdropout_seed(size_t n, float p, float alpha,  float* x, size_t incx, size_t seed) {
    if (alpha == 0.0f) {
        dropout_kernel0_run(n, x, incx);
        return;
    }

    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, 64 * 64 * sizeof(curandState)));

    // Initialize the seeds
    setup_kernel<<<64, 64>>>(states, seed);
    cudaDeviceSynchronize();

    // Compute the dropout mask
    if (alpha == 1.0f) {
        dropout_kernel1<float><<<64,64>>>(states, n, p, x, incx);
    } else {
        dropout_kernel<float><<<64,64>>>(states, n, p, alpha, x, incx);
    }

    // Free the states
    cuda_check(cudaFree(states));
}

void egblas_sdropout(size_t n, float p, float alpha,  float* x, size_t incx) {
    std::random_device rd;
    egblas_sdropout_seed(n, p, alpha, x, incx, rd());
}

void egblas_ddropout_seed(size_t n, double p, double alpha,  double* x, size_t incx, size_t seed) {
    if (alpha == 0.0f) {
        dropout_kernel0_run(n, x, incx);
        return;
    }

    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, 64 * 64 * sizeof(curandState)));

    // Initialize the seeds
    setup_kernel<<<64, 64>>>(states, seed);
    cudaDeviceSynchronize();

    // Compute the dropout mask
    if (alpha == 1.0f) {
        dropout_kernel1<double><<<64,64>>>(states, n, p, x, incx);
    } else {
        dropout_kernel<double><<<64,64>>>(states, n, p, alpha, x, incx);
    }

    // Free the states
    cuda_check(cudaFree(states));
}

void egblas_ddropout(size_t n, double p, double alpha,  double* x, size_t incx) {
    std::random_device rd;
    egblas_ddropout_seed(n, p, alpha, x, incx, rd());
}
