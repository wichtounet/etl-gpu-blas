//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/assert.hpp"
#include "egblas/sum.hpp"
#include "egblas/cuda_check.hpp"

template <typename T>
__global__ void sum_kernel(size_t n, const T* g_idata, size_t incx, T* g_odata) {
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];

    T* sdata = reinterpret_cast<T*>(sdata_raw);

    // each thread loads one element from global to shared mem
    auto tid = threadIdx.x;
    auto i   = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = g_idata[i * incx];
    } else {
        sdata[tid] = T(0);
    }

    __syncthreads();

    // do reduction in shared mem
    for (size_t s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <typename T>
T sum_kernel_run(size_t n, const T* input, size_t incx) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sum_kernel<T>, 0, 0);

    T* y_cpu = new T[blockSize];

    T* y_gpu;
    cuda_check(cudaMalloc((void**)&y_gpu, blockSize * sizeof(T)));

    int gridSize = ((n / incx) + blockSize - 1) / blockSize;

    sum_kernel<T><<<gridSize, blockSize, blockSize * sizeof(T)>>>(n, input, incx, y_gpu);

    cudaDeviceSynchronize();

    cuda_check(cudaMemcpy(y_cpu, y_gpu, blockSize * sizeof(T), cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(y_gpu));

    T sum = y_cpu[0];

    delete[] y_cpu;

    return sum;
}

float egblas_ssum(float* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_ssum");
    egblas_unused(s);

    return sum_kernel_run(n, x, s);
}

double egblas_dsum(double* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dsum");
    egblas_unused(s);

    return sum_kernel_run(n, x, s);
}

// Complex sums are done with thrust

struct single_complex_add {
    __device__ cuComplex operator()(cuComplex x, cuComplex y) {
        return cuCaddf(x, y);
    }
};

struct double_complex_add {
    __device__ cuDoubleComplex operator()(cuDoubleComplex x, cuDoubleComplex y) {
        return cuCadd(x, y);
    }
};

cuComplex egblas_csum(cuComplex* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_csum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n, make_cuComplex(0, 0), single_complex_add());
}

cuDoubleComplex egblas_zsum(cuDoubleComplex* x, size_t n, size_t s) {
    egblas_assert(s == 1, "Stride is not yet supported for egblas_zsum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n, make_cuDoubleComplex(0, 0), double_complex_add());
}
