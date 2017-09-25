//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/tanh.hpp"

template <typename T>
__global__ void tanh_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * tanh(x[incx * index]);
    }
}

template <>
__global__ void tanh_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        auto res_sinh = make_cuComplex(sinh(c.x) * cos(c.y), cosh(c.x) * sin(c.y));
        auto res_cosh = make_cuComplex(cosh(c.x) * cos(c.y), sinh(c.x) * sin(c.y));
        auto res = cuCdivf(res_sinh, res_cosh);

        y[incx * index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void tanh_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        auto res_sinh = make_cuDoubleComplex(sinh(c.x) * cos(c.y), cosh(c.x) * sin(c.y));
        auto res_cosh = make_cuDoubleComplex(cosh(c.x) * cos(c.y), sinh(c.x) * sin(c.y));
        auto res = cuCdiv(res_sinh, res_cosh);

        y[incx * index] = cuCmul(alpha, res);
    }
}

template <typename T>
__global__ void tanh_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = tanh(x[incx * index]);
    }
}

template <>
__global__ void tanh_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        auto res_sinh = make_cuComplex(sinh(c.x) * cos(c.y), cosh(c.x) * sin(c.y));
        auto res_cosh = make_cuComplex(cosh(c.x) * cos(c.y), sinh(c.x) * sin(c.y));
        auto res = cuCdivf(res_sinh, res_cosh);

        y[incx * index] = res;
    }
}

template <>
__global__ void tanh_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        auto res_sinh = make_cuDoubleComplex(sinh(c.x) * cos(c.y), cosh(c.x) * sin(c.y));
        auto res_cosh = make_cuDoubleComplex(cosh(c.x) * cos(c.y), sinh(c.x) * sin(c.y));
        auto res = cuCdiv(res_sinh, res_cosh);

        y[incx * index] = res;
    }
}

template <typename T>
__global__ void tanh_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void tanh_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void tanh_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void tanh_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tanh_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    tanh_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void tanh_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tanh_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    tanh_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void tanh_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tanh_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    tanh_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_stanh(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        tanh_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        tanh_kernel0_run(n, y, incy);
    } else {
        tanh_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dtanh(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        tanh_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        tanh_kernel0_run(n, y, incy);
    } else {
        tanh_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_ctanh(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        tanh_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        tanh_kernel0_run(n, y, incy);
    } else {
        tanh_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_ztanh(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        tanh_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        tanh_kernel0_run(n, y, incy);
    } else {
        tanh_kernel_run(n, alpha, x, incx, y, incy);
    }
}
