//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute yy = alpha * x + beta * y (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incy The stride of yy
 */
void egblas_saxpby_3(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy, float* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + beta * y (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incy The stride of yy
 */
void egblas_daxpby_3(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy, double* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + beta * y (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incy The stride of yy
 */
void egblas_caxpby_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy, cuComplex* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + beta * y (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param yy The vector yy (GPU memory)
 * \param incy The stride of yy
 */
void egblas_zaxpby_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy);

#define EGBLAS_HAS_SAXPBY_3 true
#define EGBLAS_HAS_DAXPBY_3 true
#define EGBLAS_HAS_CAXPBY_3 true
#define EGBLAS_HAS_ZAXPBY_3 true
