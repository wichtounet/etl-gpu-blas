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

template <typename T>
__global__ void binarize_kernel(size_t n, T* x, size_t incx, T threshold) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        x[incx * index] = (x[incx * index] > threshold) ? T(1) : T(0);
    }
}

template <typename T>
void binarize_kernel_run(size_t n, T* x, size_t incx, T threshold) {
    static int blockSize   = 0;
    static int minGridSize = 0;

    if (!blockSize) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, binarize_kernel<T>, 0, 0);
    }

    int gridSize = ((n / incx) + blockSize - 1) / blockSize;
    binarize_kernel<T><<<gridSize, blockSize>>>(n, x, incx, threshold);

    cudaDeviceSynchronize();
}

void egblas_sbinarize(size_t n, float* x, size_t incx, float threshold){
    binarize_kernel_run(n, x, incx, threshold);
}

void egblas_dbinarize(size_t n, double* x, size_t incx, double threshold){
    binarize_kernel_run(n, x, incx, threshold);
}
