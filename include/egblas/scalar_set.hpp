//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Sets the scalar beta to each element of the single-precision vector x
 * \param x The vector to set the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to set
 */
void egblas_scalar_sset(float* x, size_t n, size_t s, float beta);

/*!
 * \brief Sets the scalar beta to each element of the double-precision vector x
 * \param x The vector to set the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to set
 */
void egblas_scalar_dset(double* x, size_t n, size_t s, double alpha);

#define EGBLAS_HAS_SCALAR_SSET true
#define EGBLAS_HAS_SCALAR_DSET true
