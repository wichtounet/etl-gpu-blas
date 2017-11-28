//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/bias_add.hpp"

template <typename T>
__global__ void bias_add_2d_kernel(size_t m, size_t n, const T* x, size_t incx, const T* b, size_t incb, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    for (; index < m * n; index += stride) {
        y[incy * index] = x[incx * index] + b[(index % n) * incb];
    }
}

template <typename T>
void bias_add_2d_kernel_run(size_t m, size_t n, const T* x, size_t incx, const T* b, size_t incb, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bias_add_2d_kernel<T>, 0, 0);

    int gridSize = (((m * n) / incy) + blockSize - 1) / blockSize;

    bias_add_2d_kernel<T><<<gridSize, blockSize>>>(m, n, x, incx, b, incb, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sbias_add_2d(size_t m, size_t n, const float* x, size_t incx, const float* b, size_t incb, float* y, size_t incy){
    bias_add_2d_kernel_run(m, n, x, incx, b, incb, y, incy);
}

void egblas_dbias_add_2d(size_t m, size_t n, const double* x, size_t incx, const double* b, size_t incb, double* y, size_t incy){
    bias_add_2d_kernel_run(m, n, x, incx, b, incb, y, incy);
}
