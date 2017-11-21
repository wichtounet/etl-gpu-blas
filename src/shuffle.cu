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
