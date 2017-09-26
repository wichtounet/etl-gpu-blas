//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/min.hpp"

template <typename T>
__global__ void min_kernel(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = alpha * min(x[incx * index], y[incy * index]);
    }
}

template <>
__global__ void min_kernel(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        if (x_.x > y_.x) {
            y[incy * index] = cuCmulf(alpha, y_);
        } else if (y_.x > x_.x) {
            y[incy * index] = cuCmulf(alpha, x_);
        } else {
            if (x_.y > y_.y) {
                y[incy * index] = cuCmulf(alpha, y_);
            } else {
                y[incy * index] = cuCmulf(alpha, x_);
            }
        }
    }
}

template <>
__global__ void min_kernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        if (x_.x > y_.x) {
            y[incy * index] = cuCmul(alpha, y_);
        } else if (y_.x > x_.x) {
            y[incy * index] = cuCmul(alpha, x_);
        } else {
            if (x_.y > y_.y) {
                y[incy * index] = cuCmul(alpha, y_);
            } else {
                y[incy * index] = cuCmul(alpha, x_);
            }
        }
    }
}

template <typename T>
__global__ void min_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = min(x[incx * index], y[incy * index]);
    }
}

template <>
__global__ void min_kernel1(size_t n, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        if (x_.x > y_.x) {
            y[incy * index] = y_;
        } else if (y_.x > x_.x) {
            y[incy * index] = x_;
        } else {
            if (x_.y > y_.y) {
                y[incy * index] = y_;
            } else {
                y[incy * index] = x_;
            }
        }
    }
}

template <>
__global__ void min_kernel1(size_t n, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        auto x_ =  x[incx * index];
        auto y_ =  y[incy * index];

        if (x_.x > y_.x) {
            y[incy * index] = y_;
        } else if (y_.x > x_.x) {
            y[incy * index] = x_;
        } else {
            if (x_.y > y_.y) {
                y[incy * index] = y_;
            } else {
                y[incy * index] = x_;
            }
        }
    }
}

template <typename T>
__global__ void min_kernel0(size_t n, T* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = T(0);
    }
}

template <>
__global__ void min_kernel0(size_t n, cuComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuComplex(0, 0);
    }
}

template <>
__global__ void min_kernel0(size_t n, cuDoubleComplex* y, size_t incy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        y[incy * index] = make_cuDoubleComplex(0, 0);
    }
}

template <typename T>
void min_kernel_run(size_t n, T alpha, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, min_kernel<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    min_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void min_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, min_kernel1<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    min_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy);

    cudaDeviceSynchronize();
}

template <typename T>
void min_kernel0_run(size_t n, T* y, size_t incy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, min_kernel0<T>, 0, 0);

    int gridSize = ((n / incy) + blockSize - 1) / blockSize;

    min_kernel0<T><<<gridSize, blockSize>>>(n, y, incy);

    cudaDeviceSynchronize();
}

void egblas_smin(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy) {
    if (alpha == 1.0f) {
        min_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0f) {
        min_kernel0_run(n, y, incy);
    } else {
        min_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_dmin(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy) {
    if (alpha == 1.0) {
        min_kernel1_run(n, x, incx, y, incy);
    } else if (alpha == 0.0) {
        min_kernel0_run(n, y, incy);
    } else {
        min_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_cmin(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f) {
        min_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f) {
        min_kernel0_run(n, y, incy);
    } else {
        min_kernel_run(n, alpha, x, incx, y, incy);
    }
}

void egblas_zmin(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy) {
    if (alpha.x == 1.0 && alpha.y == 0.0) {
        min_kernel1_run(n, x, incx, y, incy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0) {
        min_kernel0_run(n, y, incy);
    } else {
        min_kernel_run(n, alpha, x, incx, y, incy);
    }
}
