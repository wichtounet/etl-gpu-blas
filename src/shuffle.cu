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

void egblas_shuffle_seed(size_t n, void* x, size_t incx, size_t seed){
    // Allocate memory for temporary for swapping
    void* tmp;
    cuda_check(cudaMalloc((void**)&tmp, incx));

    std::default_random_engine g(seed);

    using distribution_t = typename std::uniform_int_distribution<size_t>;
    using param_t        = typename distribution_t::param_type;

    distribution_t dist;

    uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);

    for (auto i = n - 1; i > 0; --i) {
        auto new_i = dist(g, param_t(0, i));

        // Swap x[i] and x[new_i]
        cuda_check(cudaMemcpy(tmp, x_flat + i * incx, incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(x_flat + i * incx, x_flat + new_i * incx, incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(x_flat + i * incx, tmp, incx, cudaMemcpyDeviceToDevice));
    }

    // Release the tmp memory
    cuda_check(cudaFree(tmp));
}

void egblas_shuffle(size_t n, void* x, size_t incx){
    std::random_device rd;
    egblas_shuffle_seed(n, x, incx, rd());
}

void egblas_par_shuffle_seed(size_t n, void* x, size_t incx, void* y, size_t incy, size_t seed){
    // Allocate memory for temporary for swapping
    void* tx;
    void* ty;
    cuda_check(cudaMalloc((void**)&tx, incx));
    cuda_check(cudaMalloc((void**)&ty, incy));

    std::default_random_engine g(seed);

    using distribution_t = typename std::uniform_int_distribution<size_t>;
    using param_t        = typename distribution_t::param_type;

    distribution_t dist;

    uint8_t* x_flat = reinterpret_cast<uint8_t*>(x);
    uint8_t* y_flat = reinterpret_cast<uint8_t*>(y);

    for (auto i = n - 1; i > 0; --i) {
        auto new_i = dist(g, param_t(0, i));

        // Swap x[i] and x[new_i]
        cuda_check(cudaMemcpy(tx, x_flat + i * incx, incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(x_flat + i * incx, x_flat + new_i * incx, incx, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(x_flat + i * incx, tx, incx, cudaMemcpyDeviceToDevice));

        // Swap y[i] and y[new_i]
        cuda_check(cudaMemcpy(ty, y_flat + i * incy, incy, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(y_flat + i * incy, y_flat + new_i * incy, incy, cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(y_flat + i * incy, ty, incy, cudaMemcpyDeviceToDevice));
    }

    // Release the tmp memory
    cuda_check(cudaFree(tx));
    cuda_check(cudaFree(ty));
}

void egblas_par_shuffle(size_t n, void* x, size_t incx, void* y, size_t incy){
    std::random_device rd;
    egblas_par_shuffle_seed(n, x, incx, y, incy, rd());
}
