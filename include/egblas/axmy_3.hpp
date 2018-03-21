//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute yy = alpha * x * y (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_saxmy_3(size_t n, float alpha, const float* x, size_t incx, const float* y, size_t incy, float* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x * y (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_daxmy_3(size_t n, double alpha, const double* x, size_t incx, const double* y, size_t incy, double* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x * y (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_caxmy_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, const cuComplex* y, size_t incy, cuComplex* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x * y (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_zaxmy_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, const cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x * y (element wise), in 32 bits integers
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_iaxmy_3(size_t n, int32_t alpha, const int32_t* x, size_t incx, const int32_t* y, size_t incy, int32_t* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x * y (element wise), in 64bit integers
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incyy The stride of yy
 */
void egblas_laxmy_3(size_t n, int64_t alpha, const int64_t* x, size_t incx, const int64_t* y, size_t incy, int64_t* yy, size_t incyy);

#define EGBLAS_HAS_SAXMY_3 true
#define EGBLAS_HAS_DAXMY_3 true
#define EGBLAS_HAS_CAXMY_3 true
#define EGBLAS_HAS_ZAXMY_3 true
#define EGBLAS_HAS_IAXMY_3 true
#define EGBLAS_HAS_LAXMY_3 true
