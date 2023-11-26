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
void egblas_haxpby_3(size_t n, fp16 alpha, const fp16* x, size_t incx, fp16 beta, fp16* y, size_t incy, fp16* yy, size_t incyy);

#define EGBLAS_HAS_HAXPBY_3 true
#else
#define EGBLAS_HAS_HAXPBY_3 true
#endif

#ifndef DISABLE_BF16

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
void egblas_baxpby_3(size_t n, bf16 alpha, const bf16* x, size_t incx, bf16 beta, bf16* y, size_t incy, bf16* yy, size_t incyy);

#define EGBLAS_HAS_BAXPBY_3 true
#else
#define EGBLAS_HAS_BAXPBY_3 false
#endif

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

/*!
 * \brief Compute yy = alpha * x + beta * y (element wise), in 32bits integer
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
void egblas_iaxpby_3(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t beta, int32_t* y, size_t incy, int32_t* yy, size_t incyy);

/*!
 * \brief Compute yy = alpha * x + beta * y (element wise), in 64bits integer
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
void egblas_laxpby_3(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t beta, int64_t* y, size_t incy, int64_t* yy, size_t incyy);

#define EGBLAS_HAS_SAXPBY_3 true
#define EGBLAS_HAS_DAXPBY_3 true
#define EGBLAS_HAS_CAXPBY_3 true
#define EGBLAS_HAS_ZAXPBY_3 true
#define EGBLAS_HAS_IAXPBY_3 true
#define EGBLAS_HAS_LAXPBY_3 true
