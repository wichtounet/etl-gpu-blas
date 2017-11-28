//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/tan.hpp"

template <typename T>
__global__ void tan_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * tan(x[incx * index]);
    }
}

template <>
__global__ void tan_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        auto res_sin = make_cuComplex(sin(c.x) * cosh(c.y), cos(c.x) * sinh(c.y));
        auto res_cos = make_cuComplex(cos(c.x) * cosh(c.y), -sin(c.x) * sinh(c.y));
        auto res = cuCdivf(res_sin, res_cos);

        y[incx * index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void tan_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        auto res_sin = make_cuDoubleComplex(sin(c.x) * cosh(c.y), cos(c.x) * sinh(c.y));
        auto res_cos = make_cuDoubleComplex(cos(c.x) * cosh(c.y), -sin(c.x) * sinh(c.y));
        auto res = cuCdiv(res_sin, res_cos);

        y[incx * index] = cuCmul(alpha, res);
    }
}

template <typename T>
__global__ void tan_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = tan(x[incx * index]);
    }
}

template <>
__global__ void tan_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuComplex c = x[incx * index];

        auto res_sin = make_cuComplex(sin(c.x) * cosh(c.y), cos(c.x) * sinh(c.y));
        auto res_cos = make_cuComplex(cos(c.x) * cosh(c.y), -sin(c.x) * sinh(c.y));
        auto res = cuCdivf(res_sin, res_cos);

        y[incx * index] = res;
    }
}

template <>
__global__ void tan_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        cuDoubleComplex c = x[incx * index];

        auto res_sin = make_cuDoubleComplex(sin(c.x) * cosh(c.y), cos(c.x) * sinh(c.y));
        auto res_cos = make_cuDoubleComplex(cos(c.x) * cosh(c.y), -sin(c.x) * sinh(c.y));
        auto res = cuCdiv(res_sin, res_cos);

        y[incx * index] = res;
    }
}

template <typename T>
__global__ void tan_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void tan_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void tan_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void tan_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tan_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    tan_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void tan_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tan_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    tan_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void tan_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tan_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    tan_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_stan(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        tan_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        tan_kernel0_run(n, y, incy);
    } else {
        tan_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dtan(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        tan_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        tan_kernel0_run(n, y, incy);
    } else {
        tan_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_ctan(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        tan_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        tan_kernel0_run(n, y, incy);
    } else {
        tan_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_ztan(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        tan_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        tan_kernel0_run(n, y, incy);
    } else {
        tan_kernel_run(n, alpha, x, incx, y, incy);
    }
}
