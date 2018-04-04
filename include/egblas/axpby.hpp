//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = alpha * x + beta * y (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_saxpby(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + beta * y (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_daxpby(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + beta * y (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_caxpby(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + beta * y (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zaxpby(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + beta * y (element wise), in 32bits integer
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_iaxpby(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t beta, int32_t* y, size_t incy);

/*!
 * \brief Compute y = alpha * x + beta * y (element wise), in 64bits integer
 * \param n The size of the two vectors
 * \param alpha The x multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The y multiplicator
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_laxpby(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t beta, int64_t* y, size_t incy);

#define EGBLAS_HAS_SAXPBY true
#define EGBLAS_HAS_DAXPBY true
#define EGBLAS_HAS_CAXPBY true
#define EGBLAS_HAS_ZAXPBY true
#define EGBLAS_HAS_IAXPBY true
#define EGBLAS_HAS_LAXPBY true
