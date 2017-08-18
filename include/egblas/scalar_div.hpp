//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Divide the scalar beta by each element of the single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to div
 */
void egblas_scalar_sdiv(float beta, float* x, size_t n, size_t s);

/*!
 * \brief Divide the scalar beta by each element of the double-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to div
 */
void egblas_scalar_ddiv(double beta, double* x, size_t n, size_t s);

/*!
 * \brief Divide the scalar beta by each element of the complex single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to div
 */
void egblas_scalar_cdiv(cuComplex beta, cuComplex* x, size_t n, size_t s);

/*!
 * \brief Divide the scalar beta by each element of the complex double-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to div
 */
void egblas_scalar_zdiv(cuDoubleComplex beta, cuDoubleComplex* x, size_t n, size_t s);

#define EGBLAS_HAS_SCALAR_SDIV true
#define EGBLAS_HAS_SCALAR_DDIV true
#define EGBLAS_HAS_SCALAR_CDIV true
#define EGBLAS_HAS_SCALAR_ZDIV true
