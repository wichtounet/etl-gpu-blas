//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Add the 1D bias to to the batched 1D input
 * \param m The first dimension of x and y
 * \param n The second dimension of x and y
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param b The vector x (GPU memory)
 * \param incb The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_sbias_add_2d(size_t m, size_t n, const float* x, size_t incx, const float* b, size_t incb, float* y, size_t incy);

/*!
 * \brief Add the 1D bias to to the batched 1D input
 * \param m The first dimension of x and y
 * \param n The second dimension of x and y
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param b The vector x (GPU memory)
 * \param incb The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dbias_add_2d(size_t m, size_t n, const double* x, size_t incx, const double* b, size_t incb, double* y, size_t incy);

#define EGBLAS_HAS_SBIAS_ADD_2D true
#define EGBLAS_HAS_DBIAS_ADD_2D true
