//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <random>

#include "egblas/shuffle.hpp"
#include "egblas/cuda_check.hpp"

// Swap one 4 bytes element from the array
__global__ void shuffle_one_4_kernel(uint32_t* x, size_t i, size_t new_i) {
    uint32_t tmp;

    tmp = x[i];
    x[i] = x[new_i];
    x[new_i] = tmp;
}

// Swap one 4 bytes element from each array
__global__ void par_shuffle_one_4_kernel(uint32_t* x, uint32_t* y, size_t i, size_t new_i) {
    uint32_t tmp;

    tmp = x[i];
    x[i] = x[new_i];
    x[new_i] = tmp;

    tmp = y[i];
    y[i] = y[new_i];
    y[new_i] = tmp;
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

void egblas_shuffle_seed(size_t n, void* x, size_t incx, size_t seed){
    std::default_random_engine g(seed);

    using distribution_t = typename std::uniform_int_distribution<size_t>;
    using param_t        = typename distribution_t::param_type;

    distribution_t dist;

    // Optimized version for arrays of 4B
    if (incx == 4) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            shuffle_one_4_kernel<<<1, 1, 1>>>(x_flat, i, new_i);
        }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif

        return;
    }

    if (incx % 8 == 0) {
        uint64_t* x_flat = reinterpret_cast<uint64_t*>(x);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            shuffle_kernel_run(x_flat, size_t(i), new_i, incx / 8);
        }
    } else if (incx % 4 == 0) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            shuffle_kernel_run(x_flat, size_t(i), new_i, incx / 4);
        }
    } else {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            shuffle_kernel_run(x_flat, size_t(i), new_i, incx);
        }
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_shuffle(size_t n, void* x, size_t incx){
    std::random_device rd;
    egblas_shuffle_seed(n, x, incx, rd());
}

void egblas_par_shuffle_seed(size_t n, void* x, size_t incx, void* y, size_t incy, size_t seed){
    std::default_random_engine g(seed);

    using distribution_t = typename std::uniform_int_distribution<size_t>;
    using param_t        = typename distribution_t::param_type;

    distribution_t dist;

    // Optimized version for arrays of 4B
    if(incx == 4 && incy == 4){
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);
        uint32_t* y_flat = reinterpret_cast<uint32_t*>(y);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            par_shuffle_one_4_kernel<<<1,1,1>>>(x_flat, y_flat, i, new_i);
        }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif

        return;
    }

    if (incx % 8 == 0 && incy % 8 == 0) {
        uint64_t* x_flat = reinterpret_cast<uint64_t*>(x);
        uint64_t* y_flat = reinterpret_cast<uint64_t*>(y);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            par_shuffle_kernel_run(x_flat, y_flat, size_t(i), new_i, incx / 8, incy / 8);
        }
    } else if (incx % 4 == 0 && incy % 4 == 0) {
        uint32_t* x_flat = reinterpret_cast<uint32_t*>(x);
        uint32_t* y_flat = reinterpret_cast<uint32_t*>(y);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            par_shuffle_kernel_run(x_flat, y_flat, size_t(i), new_i, incx / 4, incy / 4);
        }
    } else {
        uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);
        uint8_t* y_flat = reinterpret_cast<uint8_t*>(y);

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            par_shuffle_kernel_run(x_flat, y_flat, size_t(i), new_i, incx, incy);
        }
    }

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_par_shuffle(size_t n, void* x, size_t incx, void* y, size_t incy){
    std::random_device rd;
    egblas_par_shuffle_seed(n, x, incx, y, incy, rd());
}
