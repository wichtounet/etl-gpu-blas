//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/apxdbpy_3.hpp"

#include "complex.hpp"

template <typename T>
__global__ void apxdbpy_3_kernel(size_t n, const T alpha, const T* x, size_t incx, T beta, const T* y, size_t incy, T* yy, size_t incyy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        yy[incy * index] = (alpha + x[incx * index]) / (beta + y[incy * index]);
    }
}

template <typename T>
void apxdbpy_3_kernel_run(size_t n, T alpha, const T* x, size_t incx, T beta, const T* y, size_t incy, T* yy, size_t incyy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, apxdbpy_3_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    apxdbpy_3_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, beta, y, incy, yy, incyy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

#ifdef EGBLAS_HAS_HAPXDBPY_3

void egblas_hapxdbpy_3(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16 beta, const fp16* y, size_t incy, fp16* yy, size_t incyy) {
    apxdbpy_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
}

#endif

#ifdef EGBLAS_HAS_BAPXDBPY_3

void egblas_bapxdbpy_3(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16 beta, const bf16* y, size_t incy, bf16* yy, size_t incyy) {
    apxdbpy_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
}

#endif

void egblas_sapxdbpy_3(size_t n, float alpha, const float* x, size_t incx, float beta, const float* y, size_t incy, float* yy, size_t incyy) {
    apxdbpy_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
}

void egblas_dapxdbpy_3(size_t n, double alpha, const double* x, size_t incx, double beta, const double* y, size_t incy, double* yy, size_t incyy) {
    apxdbpy_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
}

void egblas_capxdbpy_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, const cuComplex* y, size_t incy, cuComplex* yy, size_t incyy) {
    apxdbpy_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
}

void egblas_zapxdbpy_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, const cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy) {
    apxdbpy_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
}
