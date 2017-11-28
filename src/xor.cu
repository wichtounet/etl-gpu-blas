//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/xor.hpp"

__global__ void xor_kernel(size_t n, const bool* a, size_t inca, const bool* b, size_t incb, bool* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = a[inca * index] != b[incb * index];
    }
}

void xor_kernel_run(size_t n, const bool* a, size_t inca, const bool* b, size_t incb, bool* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, xor_kernel, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    xor_kernel<<<gridSize, blockSize>>>(n, a, inca, b, incb, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_bxor(size_t n, const bool* a, size_t inca, const bool* b, size_t incb, bool* y, size_t incy) {
    xor_kernel_run(n, a, inca, b, incb, y, incy);
}
