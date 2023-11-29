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
 * \brief Compute y = (alpha * x) / (beta + y) (element wise), in half-precision FP16
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param beta The addition
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_haxdbpy_3(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16 beta, const fp16* y, size_t incy, fp16* yy, size_t incyy);

#define EGBLAS_HAS_HAXDBPY_3 true
#else
#define EGBLAS_HAS_HAXDBPY_3 false
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
void egblas_baxdbpy_3(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16 beta, const bf16* y, size_t incy, bf16* yy, size_t incyy);

#define EGBLAS_HAS_BAXDBPY_3 true
#else
#define EGBLAS_HAS_BAXDBPY_3 false
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
void egblas_saxdbpy_3(size_t n, float alpha, const float* x, size_t incx, float beta, const float* y, size_t incy, float* yy, size_t incyy);

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
void egblas_daxdbpy_3(size_t n, double alpha, const double* x, size_t incx, double beta, const double* y, size_t incy, double* yy, size_t incyy);

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
void egblas_caxdbpy_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, const cuComplex* y, size_t incy, cuComplex* yy, size_t incyy);

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
void egblas_zaxdbpy_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, const cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy);

#define EGBLAS_HAS_SAXDBPY_3 true
#define EGBLAS_HAS_DAXDBPY_3 true
#define EGBLAS_HAS_CAXDBPY_3 true
#define EGBLAS_HAS_ZAXDBPY_3 true
