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
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_haxdbpy(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16 beta, fp16* y, size_t incy);

#define EGBLAS_HAS_HAXDBPY true
#else
#define EGBLAS_HAS_HAXDBPY false
#endif

#ifndef DISABLE_BF16

/*!
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_baxdbpy(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16 beta, bf16* y, size_t incy);

#define EGBLAS_HAS_BAXDBPY true
#else
#define EGBLAS_HAS_BAXDBPY false
#endif

/*!
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_saxdbpy(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy);

/*!
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_daxdbpy(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy);

/*!
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_caxdbpy(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zaxdbpy(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy);

#define EGBLAS_HAS_SAXDBPY true
#define EGBLAS_HAS_DAXDBPY true
#define EGBLAS_HAS_CAXDBPY true
#define EGBLAS_HAS_ZAXDBPY true
