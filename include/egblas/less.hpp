//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = a < b (element wise), in single-precision
 * \param n The size of the three vectors
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector z (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_sless(size_t n, const float* x, size_t incx, const float* z, size_t incz, bool* y, size_t incy);

/*!
 * \brief Compute y = a < b (element wise), in double-precision
 * \param n The size of the three vectors
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector z (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dless(size_t n, const double* x, size_t incx, const double* z, size_t incz, bool* y, size_t incy);

/*!
 * \brief Compute y = a < b (element wise), in complex single-precision
 * \param n The size of the three vectors
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector z (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_cless(size_t n, const cuComplex* x, size_t incx, const cuComplex* z, size_t incz, bool* y, size_t incy);

/*!
 * \brief Compute y = a < b (element wise), in complex double-precision
 * \param n The size of the three vectors
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector z (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_zless(size_t n, const cuDoubleComplex* a, size_t inca, const cuDoubleComplex* b, size_t incb, bool* y, size_t incy);

#define EGBLAS_HAS_SLESS true
#define EGBLAS_HAS_DLESS true
#define EGBLAS_HAS_CLESS true
#define EGBLAS_HAS_ZLESS true
