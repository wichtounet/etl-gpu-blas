//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = alpha * x * y (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_saxmy (size_t n, float alpha, const float *x, size_t incx, float *y, size_t incy);

/*!
 * \brief Compute y = alpha * x * y (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_daxmy (size_t n, double alpha, const double *x, size_t incx, double *y, size_t incy);

/*!
 * \brief Compute y = alpha * x * y (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_caxmy (size_t n, cuComplex alpha, const cuComplex *x, size_t incx, cuComplex *y, size_t incy);

/*!
 * \brief Compute y = alpha * x * y (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zaxmy (size_t n, cuDoubleComplex alpha, const cuDoubleComplex *x, size_t incx, cuDoubleComplex *y, size_t incy);

#define EGBLAS_HAS_SAXMY true
#define EGBLAS_HAS_DAXMY true
#define EGBLAS_HAS_CAXMY true
#define EGBLAS_HAS_ZAXMY true
