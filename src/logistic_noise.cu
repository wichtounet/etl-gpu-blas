//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <random>
#include <iostream>

#include "curand_kernel.h"

#include "egblas/logistic_noise.hpp"
#include "egblas/cuda_check.hpp"

// Kernel to setup the random states

__global__ void ln_setup_kernel(curandState* states, size_t seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(seed, id, 0, &states[id]);
}

// Kernels for logistic noise

template<typename T>
__device__ T logistic_sigmoid(T x){
    return T(1) / (T(1) + exp(-x));
}

template <typename T>
__global__ void logistic_noise_kernel(curandState* states, size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    const auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    // Copy state to local memory for efficiency
    auto local_state = states[base_index];

    for (auto index = base_index; index < n; index += stride) {
        y[incy * index] = alpha * (x[incx * index] + curand_normal(&local_state) * logistic_sigmoid(x[incx * index]));
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

template <typename T>
__global__ void logistic_noise_kernel1(curandState* states, size_t n, T alpha, const T* x, T* y) {
    const auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    // Copy state to local memory for efficiency
    auto local_state = states[base_index];

    for (auto index = base_index; index < n; index += stride) {
        y[index] = alpha * (x[index] + curand_normal(&local_state) * logistic_sigmoid(x[index]));
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

template <typename T>
__global__ void logistic_noise_kernel_alpha1(curandState* states, size_t n, const T* x, size_t incx, T* y, size_t incy) {
    const auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    // Copy state to local memory for efficiency
    auto local_state = states[base_index];

    for (auto index = base_index; index < n; index += stride) {
        y[incy * index] = x[incx * index] + curand_normal(&local_state) * logistic_sigmoid(x[incx * index]);
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

template <typename T>
__global__ void logistic_noise_kernel1_alpha1(curandState* states, size_t n, const T* x, T* y) {
    const auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    // Copy state to local memory for efficiency
    auto local_state = states[base_index];

    for (auto index = base_index; index < n; index += stride) {
        y[index] = x[index] + curand_normal(&local_state) * T(logistic_sigmoid(x[index]));
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

// Kernel for reset (when alpha = 0)

template <typename T>
__global__ void logistic_noise_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        y[incy * index] = T(0);
    }
}

template <typename T>
void logistic_noise_kernel0_run(size_t n, T* y, size_t incy) {
    static int blockSize;
    static int minGridSize;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, logistic_noise_kernel0<T>, 0, 0);
    }

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;
    logistic_noise_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

// Preparation

void* egblas_logistic_noise_prepare(){
    std::random_device rd;
    return egblas_logistic_noise_prepare_seed(rd());
}

void* egblas_logistic_noise_prepare_seed(size_t seed){
    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, 64 * 64 * sizeof(curandState)));

    // Initialize the seeds
    ln_setup_kernel<<<64, 64>>>(states, seed);

    return states;
}

void egblas_logistic_noise_release(void* states){
    // Free the states
    cuda_check(cudaFree(states));
}

// Regular logistic_noise

void egblas_slogistic_noise_seed(size_t n, float alpha, const float* x, size_t incx, float * y, size_t incy, size_t seed) {
    if (alpha == 0.0f) {
        logistic_noise_kernel0_run(n, y, incy);
        return;
    }

    size_t gridSize  = 64;
    size_t blockSize = 64;

    if (n <= 100) {
        gridSize  = 1;
        blockSize = 64;
    } else if(n <= 1000){
        gridSize  = 8;
        blockSize = 64;
    } else if(n <= 10000){
        gridSize  = 16;
        blockSize = 64;
    } else if(n <= 100000){
        gridSize  = 32;
        blockSize = 64;
    }

    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, gridSize * blockSize * sizeof(curandState)));

    // Initialize the seeds
    ln_setup_kernel<<<gridSize, blockSize>>>(states, seed);

    // Compute the logistic_noise
    if (incx == 1 && incy == 1) {
        if (alpha == 1.0) {
            logistic_noise_kernel1_alpha1<float><<<gridSize, blockSize>>>(states, n, x, y);
        } else {
            logistic_noise_kernel1<float><<<gridSize, blockSize>>>(states, n, alpha, x, y);
        }
    } else {
        if (alpha == 1.0) {
            logistic_noise_kernel_alpha1<float><<<gridSize, blockSize>>>(states, n, x, incx, y, incy);
        } else {
            logistic_noise_kernel<float><<<gridSize, blockSize>>>(states, n, alpha, x, incx, y, incy);
        }
    }

    // Free the states
    cuda_check(cudaFree(states));
}

void egblas_slogistic_noise(size_t n, float alpha,  const float* x, size_t incx, float* y, size_t incy) {
    std::random_device rd;
    egblas_slogistic_noise_seed(n, alpha, x, incx, y, incy, rd());
}

void egblas_dlogistic_noise_seed(size_t n, double alpha,  const double* x, size_t incx, double* y, size_t incy, size_t seed) {
    if (alpha == 0.0) {
        logistic_noise_kernel0_run(n, y, incy);
        return;
    }

    size_t gridSize  = 64;
    size_t blockSize = 64;

    if (n <= 100) {
        gridSize  = 1;
        blockSize = 64;
    } else if(n <= 1000){
        gridSize  = 8;
        blockSize = 64;
    } else if(n <= 10000){
        gridSize  = 16;
        blockSize = 64;
    } else if(n <= 100000){
        gridSize  = 32;
        blockSize = 64;
    }

    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, gridSize * blockSize * sizeof(curandState)));

    // Initialize the seeds
    ln_setup_kernel<<<gridSize, blockSize>>>(states, seed);

    // Compute the logistic_noise
    if (incx == 1 && incy == 1) {
        if (alpha == 1.0) {
            logistic_noise_kernel1_alpha1<double><<<gridSize, blockSize>>>(states, n, x, y);
        } else {
            logistic_noise_kernel1<double><<<gridSize, blockSize>>>(states, n, alpha, x, y);
        }
    } else {
        if (alpha == 1.0) {
            logistic_noise_kernel_alpha1<double><<<gridSize, blockSize>>>(states, n, x, incx, y, incy);
        } else {
            logistic_noise_kernel<double><<<gridSize, blockSize>>>(states, n, alpha, x, incx, y, incy);
        }
    }

    // Free the states
    cuda_check(cudaFree(states));
}

void egblas_dlogistic_noise(size_t n, double alpha,  const double* x, size_t incx, double* y, size_t incy) {
    std::random_device rd;
    egblas_dlogistic_noise_seed(n, alpha, x, incx, y, incy, rd());
}

// Function with stats

void egblas_slogistic_noise_states(size_t n, float alpha,  const float* x, size_t incx, float* y, size_t incy, void* states) {
    if (alpha == 0.0f) {
        logistic_noise_kernel0_run(n, y, incy);
        return;
    }

    size_t gridSize  = 64;
    size_t blockSize = 64;

    // Compute the logistic_noise
    curandState* cstates = reinterpret_cast<curandState*>(states);

    if (incx == 1 && incy == 1) {
        if (alpha == 1.0f) {
            logistic_noise_kernel1_alpha1<float><<<gridSize, blockSize>>>(cstates, n, x, y);
        } else {
            logistic_noise_kernel1<float><<<gridSize, blockSize>>>(cstates, n, alpha, x, y);
        }
    } else {
        if (alpha == 1.0f) {
            logistic_noise_kernel_alpha1<float><<<gridSize, blockSize>>>(cstates, n, x, incx, y, incy);
        } else {
            logistic_noise_kernel<float><<<gridSize, blockSize>>>(cstates, n, alpha, x, incx, y, incy);
        }
    }
}

void egblas_dlogistic_noise_states(size_t n, double alpha,  const double* x, size_t incx, double* y, size_t incy, void* states) {
    if (alpha == 0.0) {
        logistic_noise_kernel0_run(n, y, incy);
        return;
    }

    size_t gridSize  = 64;
    size_t blockSize = 64;

    // Compute the logistic_noise
    curandState* cstates = reinterpret_cast<curandState*>(states);

    if (incx == 1 && incy == 1) {
        if (alpha == 1.0) {
            logistic_noise_kernel1_alpha1<double><<<gridSize, blockSize>>>(cstates, n, x, y);
        } else {
            logistic_noise_kernel1<double><<<gridSize, blockSize>>>(cstates, n, alpha, x, y);
        }
    } else {
        if (alpha == 1.0) {
            logistic_noise_kernel_alpha1<double><<<gridSize, blockSize>>>(cstates, n, x, incx, y, incy);
        } else {
            logistic_noise_kernel<double><<<gridSize, blockSize>>>(cstates, n, alpha, x, incx, y, incy);
        }
    }
}
