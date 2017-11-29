//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = (alpha + x) / (beta * y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_sapxdby(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy);

/*!
 * \brief Compute y = (alpha + x) / (beta * y) (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dapxdby(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy);

/*!
 * \brief Compute y = (alpha + x) / (beta * y) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_capxdby(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = (alpha + x) / (beta * y) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zapxdby(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy);

#define EGBLAS_HAS_SAPXDBY true
#define EGBLAS_HAS_DAPXDBY true
#define EGBLAS_HAS_CAPXDBY true
#define EGBLAS_HAS_ZAPXDBY true
