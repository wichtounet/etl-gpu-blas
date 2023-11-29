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
 * \brief Compute y = (alpha + x) / (beta + y) (element wise), in half-precision fp16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_hapxdbpy(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16 beta, fp16* y, size_t incy);

#define EGBLAS_HAS_HAPXDBPY true
#else
#define EGBLAS_HAS_HAPXDBPY false
#endif

#ifndef DISABLE_BF16

/*!
 * \brief Compute y = (alpha + x) / (beta + y) (element wise), in half-precision bf16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_bapxdbpy(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16 beta, bf16* y, size_t incy);

#define EGBLAS_HAS_BAPXDBPY true
#else
#define EGBLAS_HAS_BAPXDBPY false
#endif

/*!
 * \brief Compute y = (alpha + x) / (beta + y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_sapxdbpy(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy);

/*!
 * \brief Compute y = (alpha + x) / (beta + y) (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dapxdbpy(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy);

/*!
 * \brief Compute y = (alpha + x) / (beta + y) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_capxdbpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = (alpha + x) / (beta + y) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zapxdbpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy);

#define EGBLAS_HAS_SAPXDBPY true
#define EGBLAS_HAS_DAPXDBPY true
#define EGBLAS_HAS_CAPXDBPY true
#define EGBLAS_HAS_ZAPXDBPY true
