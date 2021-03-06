//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <random>

#include "curand_kernel.h"

#include "egblas/shuffle.hpp"
#include "egblas/cuda_check.hpp"

namespace {

// Kernel to setup the random states

__global__ void setup_kernel(curandState* states, size_t seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(seed, id, 0, &states[id]);
}

__global__ void setup_permutation_kernel(size_t n, size_t* permutation) {
    auto index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        permutation[index] = index;
    }
}

template <typename T>
__global__ void apply_permutation_kernel(size_t n, size_t* permutation, T* x_tmp, T* x, size_t incx) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        size_t new_i = permutation[i];

        for (size_t index = 0; index < incx; ++index) {
            x[i * incx + index] = x_tmp[new_i * incx + index];
        }
    }
}

template <size_t Threads, size_t Distance, typename T>
__global__ void shuffle_one_kernel_states(curandState* states, size_t n, T* x) {
    const auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t split_size = n / Threads;
    const size_t half_split_size = split_size / 2;

    //Copy state to local memory for efficiency
    auto local_state = states[base_index];

    if (Distance == 1) {
        const size_t i_start = base_index * split_size;
        const size_t i_end   = i_start + half_split_size;

        const size_t new_i_start = i_end;

        for (size_t i = i_start; i < i_end; ++i) {
            unsigned int new_i = new_i_start + ceilf(curand_uniform(&local_state) * half_split_size) - 1;

            if (i < n && new_i < n) {
                T tmp    = x[i];
                x[i]     = x[new_i];
                x[new_i] = tmp;
            }
        }
    } else if (Distance == 2) {
        const size_t i_start = base_index * half_split_size;
        const size_t i_end   = i_start + half_split_size;

        const size_t new_i_start = i_start + Threads * half_split_size;

        for (size_t i = i_start; i < i_end; ++i) {
            unsigned int new_i = new_i_start + ceilf(curand_uniform(&local_state) * half_split_size) - 1;

            if (i < n && new_i < n) {
                T tmp    = x[i];
                x[i]     = x[new_i];
                x[new_i] = tmp;
            }
        }
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

template <size_t Threads, size_t Distance, typename T>
__global__ void par_shuffle_one_kernel_states(curandState* states, size_t n, T* x, T* y) {
    const auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t split_size = n / Threads;
    const size_t half_split_size = split_size / 2;

    //Copy state to local memory for efficiency
    auto local_state = states[base_index];

    if (Distance == 1) {
        const size_t i_start = base_index * split_size;
        const size_t i_end   = i_start + half_split_size;

        const size_t new_i_start = i_end;

        for (size_t i = i_start; i < i_end; ++i) {
            unsigned int new_i = new_i_start + ceilf(curand_uniform(&local_state) * half_split_size) - 1;

            if (i < n && new_i < n) {
                T tmp    = x[i];
                x[i]     = x[new_i];
                x[new_i] = tmp;

                tmp      = y[i];
                y[i]     = y[new_i];
                y[new_i] = tmp;
            }
        }
    } else if (Distance == 2) {
        const size_t i_start = base_index * half_split_size;
        const size_t i_end   = i_start + half_split_size;

        const size_t new_i_start = i_start + Threads * half_split_size;

        for (size_t i = i_start; i < i_end; ++i) {
            unsigned int new_i = new_i_start + ceilf(curand_uniform(&local_state) * half_split_size) - 1;

            if (i < n && new_i < n) {
                T tmp    = x[i];
                x[i]     = x[new_i];
                x[new_i] = tmp;

                tmp      = y[i];
                y[i]     = y[new_i];
                y[new_i] = tmp;
            }
        }
    }

    // Copy state back to global memory
    states[base_index] = local_state;
}

// Swap one element from the array
template<typename T>
__global__ void shuffle_kernel(T* x, size_t i, size_t new_i, size_t inc) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < inc; index += stride) {
        T tmp                    = x[i * inc + index];
        x[i * inc + index]     = x[new_i * inc + index];
        x[new_i * inc + index] = tmp;
    }
}

// Swap one element from the each array
template<typename T>
__global__ void par_shuffle_kernel(T* x, T* y, size_t i, size_t new_i, size_t incx, size_t incy) {
    auto base_index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (auto index = base_index; index < incx; index += stride) {
        T tmp                   = x[i * incx + index];
        x[i * incx + index]     = x[new_i * incx + index];
        x[new_i * incx + index] = tmp;
    }

    for (auto index = base_index; index < incy; index += stride) {
        T tmp                   = y[i * incy + index];
        y[i * incy + index]     = y[new_i * incy + index];
        y[new_i * incy + index] = tmp;
    }
}

template<typename T>
void shuffle_kernel_run(T* x, size_t i, size_t new_i, size_t inc) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, shuffle_kernel<T>, 0, 0);
    }

    int gridSize = (inc + blockSize - 1) / blockSize;
    shuffle_kernel<T><<<gridSize, blockSize>>>(x, i, new_i, inc);
}

template<typename T>
void par_shuffle_kernel_run(T* x, T* y, size_t i, size_t new_i, size_t incx, size_t incy) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, par_shuffle_kernel<T>, 0, 0);
    }

    size_t n = incx;

    if(incy > n){
        n = incy;
    }

    int gridSize = (n + blockSize - 1) / blockSize;
    par_shuffle_kernel<T><<<gridSize, blockSize>>>(x, y, i, new_i, incx, incy);
}

} // end of anonymous namespace

template <typename T>
void egblas_shuffle_one(size_t n, T * x, size_t seed) {
    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, 64 * 64 * sizeof(curandState)));

    if (n < 1000) {
        setup_kernel<<<8, 1>>>(states, seed);

        shuffle_one_kernel_states<8, 1, T><<<8, 1>>>(states, n, x);
        shuffle_one_kernel_states<8, 2, T><<<8, 1>>>(states, n, x);
        shuffle_one_kernel_states<8, 1, T><<<8, 1>>>(states, n, x);
        shuffle_one_kernel_states<8, 2, T><<<8, 1>>>(states, n, x);
    } else if (n < 50000) {
        setup_kernel<<<64, 1>>>(states, seed);

        shuffle_one_kernel_states<64, 1, T><<<64, 1>>>(states, n, x);
        shuffle_one_kernel_states<64, 2, T><<<64, 1>>>(states, n, x);
        shuffle_one_kernel_states<64, 1, T><<<64, 1>>>(states, n, x);
        shuffle_one_kernel_states<64, 2, T><<<64, 1>>>(states, n, x);
    } else {
        setup_kernel<<<64, 64>>>(states, seed);

        shuffle_one_kernel_states<64 * 64, 1, T><<<64, 64>>>(states, n, x);
        shuffle_one_kernel_states<64 * 64, 2, T><<<64, 64>>>(states, n, x);
        shuffle_one_kernel_states<64 * 64, 1, T><<<64, 64>>>(states, n, x);
        shuffle_one_kernel_states<64 * 64, 2, T><<<64, 64>>>(states, n, x);
    }

    cuda_check(cudaFree(states));
}

void egblas_shuffle_seed(size_t n, void* x, size_t incx, size_t seed){
    std::default_random_engine g(seed);

    using distribution_t = typename std::uniform_int_distribution<size_t>;
    using param_t        = typename distribution_t::param_type;

    distribution_t dist;

    // Optimized version for arrays of 8B, 4B, 1B
    if (incx == 8) {
        uint64_t* x_flat = reinterpret_cast<uint64_t*>(x);

        egblas_shuffle_one(n, x_flat, seed);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif

        return;
    } else if (incx == 4) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);

        egblas_shuffle_one(n, x_flat, seed);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif

        return;
    } else if (incx == 1) {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);

        egblas_shuffle_one(n, x_flat, seed);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif

        return;
    }

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // TODO Instead of fixing the size to 64, 32 and 8
    // This could be simply delegate the task of doing it efficienly in the 
    // apply_permutation_kernel

    if (n < 512) {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            shuffle_kernel_run(x_flat, size_t(i), new_i, incx);
        }
    } else if (incx % 8 == 0) {
        uint64_t* x_flat = reinterpret_cast<uint64_t*>(x);

        uint64_t* x_tmp;
        cuda_check(cudaMalloc((void**)&x_tmp, n * incx));
        cuda_check(cudaMemcpy(x_tmp, x_flat, n * incx, cudaMemcpyDeviceToDevice));

        size_t* permutation;
        cuda_check(cudaMalloc((void**)&permutation, n * sizeof(size_t)));

        setup_permutation_kernel<<<gridSize, blockSize>>>(n, permutation);

        egblas_shuffle_one(n, permutation, seed);

        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, x_tmp, x_flat, incx / 8);

        cuda_check(cudaFree(x_tmp));
        cuda_check(cudaFree(permutation));
    } else if (incx % 4 == 0) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);

        uint32_t* x_tmp;
        cuda_check(cudaMalloc((void**)&x_tmp, n * incx));
        cuda_check(cudaMemcpy(x_tmp, x_flat, n * incx, cudaMemcpyDeviceToDevice));

        size_t* permutation;
        cuda_check(cudaMalloc((void**)&permutation, n * sizeof(size_t)));

        setup_permutation_kernel<<<gridSize, blockSize>>>(n, permutation);

        egblas_shuffle_one(n, permutation, seed);

        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, x_tmp, x_flat, incx / 4);

        cuda_check(cudaFree(x_tmp));
        cuda_check(cudaFree(permutation));
    } else {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);

        uint8_t* x_tmp;
        cuda_check(cudaMalloc((void**)&x_tmp, n * incx));
        cuda_check(cudaMemcpy(x_tmp, x_flat, n * incx, cudaMemcpyDeviceToDevice));

        size_t* permutation;
        cuda_check(cudaMalloc((void**)&permutation, n * sizeof(size_t)));

        setup_permutation_kernel<<<gridSize, blockSize>>>(n, permutation);

        egblas_shuffle_one(n, permutation, seed);

        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, x_tmp, x_flat, incx);

        cuda_check(cudaFree(x_tmp));
        cuda_check(cudaFree(permutation));
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_shuffle(size_t n, void* x, size_t incx){
    std::random_device rd;
    egblas_shuffle_seed(n, x, incx, rd());
}

template <typename T>
void egblas_par_shuffle_one(size_t n, T* x, T * y, size_t seed) {
    // Allocate room for the states
    curandState* states;
    cuda_check(cudaMalloc((void**)&states, 64 * 64 * sizeof(curandState)));

    if (n < 1000) {
        setup_kernel<<<8, 1>>>(states, seed);

        par_shuffle_one_kernel_states<8, 1, T><<<8, 1>>>(states, n, x, y);
        par_shuffle_one_kernel_states<8, 2, T><<<8, 1>>>(states, n, x, y);
        par_shuffle_one_kernel_states<8, 1, T><<<8, 1>>>(states, n, x, y);
        par_shuffle_one_kernel_states<8, 2, T><<<8, 1>>>(states, n, x, y);
    } else if (n < 50000) {
        setup_kernel<<<64, 1>>>(states, seed);

        par_shuffle_one_kernel_states<64, 1, T><<<64, 1>>>(states, n, x, y);
        par_shuffle_one_kernel_states<64, 2, T><<<64, 1>>>(states, n, x, y);
        par_shuffle_one_kernel_states<64, 1, T><<<64, 1>>>(states, n, x, y);
        par_shuffle_one_kernel_states<64, 2, T><<<64, 1>>>(states, n, x, y);
    } else {
        setup_kernel<<<64, 64>>>(states, seed);

        par_shuffle_one_kernel_states<64 * 64, 1, T><<<64, 64>>>(states, n, x, y);
        par_shuffle_one_kernel_states<64 * 64, 2, T><<<64, 64>>>(states, n, x, y);
        par_shuffle_one_kernel_states<64 * 64, 1, T><<<64, 64>>>(states, n, x, y);
        par_shuffle_one_kernel_states<64 * 64, 2, T><<<64, 64>>>(states, n, x, y);
    }

    cuda_check(cudaFree(states));
}

void egblas_par_shuffle_seed(size_t n, void* x, size_t incx, void* y, size_t incy, size_t seed){
    std::default_random_engine g(seed);

    using distribution_t = typename std::uniform_int_distribution<size_t>;
    using param_t        = typename distribution_t::param_type;

    distribution_t dist;

    // Optimized version for arrays of 8B, 4B, 1B
    if (incx == 8 && incy == 8) {
        uint64_t* x_flat = reinterpret_cast<uint64_t*>(x);
        uint64_t* y_flat = reinterpret_cast<uint64_t*>(y);

        egblas_par_shuffle_one(n, x_flat, y_flat, seed);

#ifdef EGBLAS_SYNCHRONIZE
        cudaDeviceSynchronize();
#endif
    } else if (incx == 4 && incy == 4) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);
        uint32_t* y_flat = reinterpret_cast<uint32_t*>(y);

        egblas_par_shuffle_one(n, x_flat, y_flat, seed);

#ifdef EGBLAS_SYNCHRONIZE
        cudaDeviceSynchronize();
#endif
        return;
    } else if (incx == 1 && incy == 1) {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);
        uint8_t* y_flat = reinterpret_cast<uint8_t*>(y);

        egblas_par_shuffle_one(n, x_flat, y_flat, seed);

#ifdef EGBLAS_SYNCHRONIZE
        cudaDeviceSynchronize();
#endif
        return;
    }

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // TODO Instead of fixing the size to 64, 32 and 8
    // This could be simply delegate the task of doing it efficienly in the 
    // apply_permutation_kernel

    if (n < 512) {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);
        uint8_t* y_flat = reinterpret_cast<uint8_t*>(y);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            par_shuffle_kernel_run(x_flat, y_flat, size_t(i), new_i, incx, incy);
        }
    } else if (incx % 8 == 0 && incy % 8 == 0) {
        uint64_t* x_flat = reinterpret_cast<uint64_t*>(x);
        uint64_t* y_flat = reinterpret_cast<uint64_t*>(y);

        uint64_t* x_tmp;
        uint64_t* y_tmp;

        cuda_check(cudaMalloc((void**)&x_tmp, n * incx));
        cuda_check(cudaMalloc((void**)&y_tmp, n * incy));

        cuda_check(cudaMemcpy(x_tmp, x_flat, n * incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(y_tmp, y_flat, n * incy, cudaMemcpyDeviceToDevice));

        size_t* permutation;
        cuda_check(cudaMalloc((void**)&permutation, n * sizeof(size_t)));

        setup_permutation_kernel<<<gridSize, blockSize>>>(n, permutation);

        egblas_shuffle_one(n, permutation, seed);

        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, x_tmp, x_flat, incx / 8);
        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, y_tmp, y_flat, incy / 8);

        cuda_check(cudaFree(x_tmp));
        cuda_check(cudaFree(y_tmp));
        cuda_check(cudaFree(permutation));
    } else if (incx % 4 == 0 && incy % 4 == 0) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);
        uint32_t* y_flat = reinterpret_cast<uint32_t*>(y);

        uint32_t* x_tmp;
        uint32_t* y_tmp;

        cuda_check(cudaMalloc((void**)&x_tmp, n * incx));
        cuda_check(cudaMalloc((void**)&y_tmp, n * incy));

        cuda_check(cudaMemcpy(x_tmp, x_flat, n * incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(y_tmp, y_flat, n * incy, cudaMemcpyDeviceToDevice));

        size_t* permutation;
        cuda_check(cudaMalloc((void**)&permutation, n * sizeof(size_t)));

        setup_permutation_kernel<<<gridSize, blockSize>>>(n, permutation);

        egblas_shuffle_one(n, permutation, seed);

        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, x_tmp, x_flat, incx / 4);
        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, y_tmp, y_flat, incy / 4);

        cuda_check(cudaFree(x_tmp));
        cuda_check(cudaFree(y_tmp));
        cuda_check(cudaFree(permutation));
    } else {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);
        uint8_t* y_flat = reinterpret_cast<uint8_t*>(y);

        uint8_t* x_tmp;
        uint8_t* y_tmp;

        cuda_check(cudaMalloc((void**)&x_tmp, n * incx));
        cuda_check(cudaMalloc((void**)&y_tmp, n * incy));

        cuda_check(cudaMemcpy(x_tmp, x_flat, n * incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(y_tmp, y_flat, n * incy, cudaMemcpyDeviceToDevice));

        size_t* permutation;
        cuda_check(cudaMalloc((void**)&permutation, n * sizeof(size_t)));

        setup_permutation_kernel<<<gridSize, blockSize>>>(n, permutation);

        egblas_shuffle_one(n, permutation, seed);

        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, x_tmp, x_flat, incx);
        apply_permutation_kernel<<<gridSize, blockSize>>>(n, permutation, y_tmp, y_flat, incy);

        cuda_check(cudaFree(x_tmp));
        cuda_check(cudaFree(y_tmp));
        cuda_check(cudaFree(permutation));
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_par_shuffle(size_t n, void* x, size_t incx, void* y, size_t incy){
    std::random_device rd;
    egblas_par_shuffle_seed(n, x, incx, y, incy, rd());
}
