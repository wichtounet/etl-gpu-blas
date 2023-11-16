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
 * \brief Compute yy = alpha * x + y (element wise), in half-precision FP16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_haxpy_3(size_t n, float alpha, const float* x, size_t incx, const float* y, size_t incy, float* yy, size_t incyy);

#define EGBLAS_HAS_HAXPY_3 true
#else
#define EGBLAS_HAS_HAXPY_3 true
#endif

#ifndef DISABLE_BF16

/*!
 * \brief Compute yy = alpha * x + y (element wise), in half-precision BF16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_baxpy_3(size_t n, float alpha, const float* x, size_t incx, const float* y, size_t incy, float* yy, size_t incyy);

#define EGBLAS_HAS_BAXPY_3 true
#else
#define EGBLAS_HAS_BAXPY_3 false
#endif

/*!
 * \brief Compute yy = alpha * x + y (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_saxpy_3(size_t n, float alpha, const float* x, size_t incx, const float* y, size_t incy, float* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + y (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_daxpy_3(size_t n, double alpha, const double* x, size_t incx, const double* y, size_t incy, double* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + y (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_caxpy_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, const cuComplex* y, size_t incy, cuComplex* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + y (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_zaxpy_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, const cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + y (element wise), in 32bits integers precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_iaxpy_3(size_t n, int32_t alpha, const int32_t* x, size_t incx, const int32_t* y, size_t incy, int32_t* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + y (element wise), in 64bits integers precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_laxpy_3(size_t n, int64_t alpha, const int64_t* x, size_t incx, const int64_t* y, size_t incy, int64_t* yy, size_t incyy);

#define EGBLAS_HAS_SAXPY_3 true
#define EGBLAS_HAS_DAXPY_3 true
#define EGBLAS_HAS_CAXPY_3 true
#define EGBLAS_HAS_ZAXPY_3 true
#define EGBLAS_HAS_IAXPY_3 true
#define EGBLAS_HAS_LAXPY_3 true
