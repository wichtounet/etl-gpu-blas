//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuda_fp16.h>
#include <cuComplex.h>


void egblas_haxpy(size_t n, __half2 alpha, const __half2* x, size_t incx, __half2* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_saxpy(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_daxpy(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_caxpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zaxpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in 32 bits integer
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_iaxpy(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in 64 bits integer
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_laxpy(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t* y, size_t incy);

#define EGBLAS_HAS_SAXPY true
#define EGBLAS_HAS_DAXPY true
#define EGBLAS_HAS_CAXPY true
#define EGBLAS_HAS_ZAXPY true
#define EGBLAS_HAS_IAXPY true
#define EGBLAS_HAS_LAXPY true
