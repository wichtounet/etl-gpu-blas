//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/apxdbpy.hpp"

#include "complex.hpp"

template <typename T>
__global__ void apxdbpy_kernel(size_t n, const T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = (alpha + x[incx * index]) / (beta + y[incy * index]);
    }
}

template <typename T>
void apxdbpy_kernel_run(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, apxdbpy_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    apxdbpy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, beta, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_sapxdbpy(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy) {
    apxdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}

void egblas_dapxdbpy(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy) {
    apxdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}

void egblas_capxdbpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy) {
    apxdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}

void egblas_zapxdbpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy) {
    apxdbpy_kernel_run(n, alpha, x, incx, beta, y, incy);
}
