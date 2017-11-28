//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/not_equal.hpp"

#include "complex.hpp"

__device__ bool operator!=(cuComplex lhs, cuComplex rhs){
    return lhs.x != rhs.x || lhs.y != rhs.y;
}

__device__ bool operator!=(cuDoubleComplex lhs, cuDoubleComplex rhs){
    return lhs.x != rhs.x || lhs.y != rhs.y;
}

template <typename T>
__global__ void not_equal_kernel(size_t n, const T* a, size_t inca, const T* b, size_t incb, bool* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = a[inca * index] != b[incb * index];
    }
}

template <typename T>
void not_equal_kernel_run(size_t n, const T* a, size_t inca, const T* b, size_t incb, bool* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, not_equal_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    not_equal_kernel<T><<<gridSize, blockSize>>>(n, a, inca, b, incb, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_snot_equal(size_t n, const float* a, size_t inca, const float* b, size_t incb, bool* y, size_t incy) {
    not_equal_kernel_run(n, a, inca, b, incb, y, incy);
}

void egblas_dnot_equal(size_t n, const double* a, size_t inca, const double* b, size_t incb, bool* y, size_t incy) {
    not_equal_kernel_run(n, a, inca, b, incb, y, incy);
}

void egblas_cnot_equal(size_t n, const cuComplex* a, size_t inca, const cuComplex* b, size_t incb, bool* y, size_t incy) {
    not_equal_kernel_run(n, a, inca, b, incb, y, incy);
}

void egblas_znot_equal(size_t n, const cuDoubleComplex* a, size_t inca, const cuDoubleComplex* b, size_t incb, bool* y, size_t incy) {
    not_equal_kernel_run(n, a, inca, b, incb, y, incy);
}
