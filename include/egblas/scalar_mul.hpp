//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Muls the scalar beta to each element of the single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
void egblas_scalar_smul(float* x, size_t n, size_t s, float beta);

/*!
 * \brief Muls the scalar beta to each element of the double-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
void egblas_scalar_dmul(double* x, size_t n, size_t s, double alpha);

/*!
 * \brief Muls the scalar beta to each element of the single-precision complex vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
void egblas_scalar_cmul(cuComplex* x, size_t n, size_t s, cuComplex alpha);

/*!
 * \brief Muls the scalar beta to each element of the double-precision complex vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
void egblas_scalar_zmul(cuDoubleComplex* x, size_t n, size_t s, cuDoubleComplex alpha);

#define EGBLAS_HAS_SCALAR_SMUL true
#define EGBLAS_HAS_SCALAR_DMUL true
#define EGBLAS_HAS_SCALAR_CMUL true
#define EGBLAS_HAS_SCALAR_ZMUL true
