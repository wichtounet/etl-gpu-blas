//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = alpha * max(x,y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_smax(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(x,y) (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dmax(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(x,y) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_cmax(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(x,y) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zmax(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(x,y) (element wise), in single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector b (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_smax(size_t n, float alpha, const float* a, size_t inca, const float* b, size_t incb, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(a,y) (element wise), in double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector b (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dmax(size_t n, double alpha, const double* a, size_t inca, const double* b, size_t incb, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(a,y) (element wise), in complex single-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector b (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_cmax(size_t n, cuComplex alpha, const cuComplex* a, size_t inca, const cuComplex* b, size_t incb, cuComplex* y, size_t incy);

/*!
 * \brief Compute y = alpha * max(a,y) (element wise), in complex double-precision
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector b (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zmax(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* a, size_t inca, const cuDoubleComplex* b, size_t incb, cuDoubleComplex* y, size_t incy);

#define EGBLAS_HAS_SMAX true
#define EGBLAS_HAS_DMAX true
#define EGBLAS_HAS_CMAX true
#define EGBLAS_HAS_ZMAX true

#define EGBLAS_HAS_SMAX3 true
#define EGBLAS_HAS_DMAX3 true
#define EGBLAS_HAS_CMAX3 true
#define EGBLAS_HAS_ZMAX3 true
