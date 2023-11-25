//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

#include "half.hpp"

#ifndef DISABLE_FP16

/*!
 * \brief Compute y = alpha * x + y (element wise), in half-precision FP16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_haxpy(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16* y, size_t incy);

#define EGBLAS_HAS_HAXPY true
#else
#define EGBLAS_HAS_HAXPY true
#endif

#ifndef DISABLE_BF16

/*!
 * \brief Compute y = alpha * x + y (element wise), in half-precision BF16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_baxpy(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16* y, size_t incy);

#define EGBLAS_HAS_BAXPY true
#else
#define EGBLAS_HAS_BAXPY false
#endif

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
 * \brief Compute y = alpha * x + y (element wise), in 8 bits integer
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_oaxpy(size_t n, int8_t alpha, const int8_t* x, size_t incx, int8_t* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + y (element wise), in 16 bits integer
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_waxpy(size_t n, int16_t alpha, const int16_t* x, size_t incx, int16_t* y, size_t incy);

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
