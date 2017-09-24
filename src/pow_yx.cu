//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/pow.hpp"

template <typename T>
__global__ void pow_yx_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * pow(y[incy * index], x[incx * index]);
    }
}

template <>
__global__ void pow_yx_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        float c_abs = hypot(y_.x, y_.y);
        float c_arg = atan2f(y_.y, y_.x);

        auto logx = make_cuComplex(logf(c_abs), c_arg);
        auto ylogx = cuCmulf(x_, logx);

        float e = expf(ylogx.x);
        auto res = make_cuComplex(e * cosf(ylogx.y), e * sinf(ylogx.y));

        y[incy * index] = cuCmulf(alpha, res);
    }
}

template <>
__global__ void pow_yx_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        double c_abs = hypot(y_.x, y_.y);
        double c_arg = atan2(y_.y, y_.x);

        auto logx = make_cuDoubleComplex(log(c_abs), c_arg);
        auto ylogx = cuCmul(x_, logx);

        double e = expf(ylogx.x);
        auto res = make_cuDoubleComplex(e * cos(ylogx.y), e * sin(ylogx.y));

        y[incy * index] = cuCmul(alpha, res);
    }
}

template <typename T>
__global__ void pow_yx_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = pow(y[incy * index], x[incx * index]);
    }
}

template <>
__global__ void pow_yx_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        float c_abs = hypot(y_.x, y_.y);
        float c_arg = atan2f(y_.y, y_.x);

        auto logx = make_cuComplex(logf(c_abs), c_arg);

        auto ylogx = cuCmulf(x_, logx);

        float e = expf(ylogx.x);
        auto res = make_cuComplex(e * cosf(ylogx.y), e * sinf(ylogx.y));

        y[incy * index] = res;
    }
}

template <>
__global__ void pow_yx_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        double c_abs = hypot(y_.x, y_.y);
        double c_arg = atan2(y_.y, y_.x);

        auto logx = make_cuDoubleComplex(log(c_abs), c_arg);
        auto ylogx = cuCmul(x_, logx);

        double e = exp(ylogx.x);
        auto res = make_cuDoubleComplex(e * cos(ylogx.y), e * sin(ylogx.y));

        y[incy * index] = res;
    }
}

template <typename T>
__global__ void pow_yx_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void pow_yx_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void pow_yx_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void pow_yx_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pow_yx_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    pow_yx_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void pow_yx_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pow_yx_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    pow_yx_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void pow_yx_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pow_yx_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    pow_yx_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_spow_yx(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        pow_yx_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        pow_yx_kernel0_run(n, y, incy);
    } else {
        pow_yx_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dpow_yx(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        pow_yx_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        pow_yx_kernel0_run(n, y, incy);
    } else {
        pow_yx_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_cpow_yx(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        pow_yx_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        pow_yx_kernel0_run(n, y, incy);
    } else {
        pow_yx_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zpow_yx(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        pow_yx_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        pow_yx_kernel0_run(n, y, incy);
    } else {
        pow_yx_kernel_run(n, alpha, x, incx, y, incy);
    }
}
