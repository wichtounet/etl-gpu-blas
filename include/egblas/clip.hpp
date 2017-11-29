//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_sclip(size_t n, float alpha, const float* x, size_t incx, const float* z, size_t incz, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dclip(size_t n, double alpha, const double* x, size_t incx, const double* z, size_t incz, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_cclip(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, const cuComplex* z, size_t incz, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zclip(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, const cuDoubleComplex* z, size_t incz, cuDoubleComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_sclip_value(size_t n, float alpha, float x, float z, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dclip_value(size_t n, double alpha, double x, double z, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_cclip_value(size_t n, cuComplex alpha, cuComplex x, cuComplex z, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * clip(y, x, z) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param z The vector z (GPU memory)
 * \param incz The stride of z
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zclip_value(size_t n, cuDoubleComplex alpha, cuDoubleComplex x, cuDoubleComplex z, cuDoubleComplex* y, size_t incy);

#define EGBLAS_HAS_SCLIP true
#define EGBLAS_HAS_DCLIP true
#define EGBLAS_HAS_CCLIP true
#define EGBLAS_HAS_ZCLIP true

#define EGBLAS_HAS_SCLIP_VALUE true
#define EGBLAS_HAS_DCLIP_VALUE true
#define EGBLAS_HAS_CCLIP_VALUE true
#define EGBLAS_HAS_ZCLIP_VALUE true
