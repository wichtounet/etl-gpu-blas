//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Adds the scalar beta to each element of the vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
void egblas_scalar_sadd(float* x, size_t n, size_t s, float beta);

/*!
 * \brief Adds the scalar beta to each element of the vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
void egblas_scalar_dadd(double* x, size_t n, size_t s, double alpha);

#define EGBLAS_HAS_SCALAR_SADD true
#define EGBLAS_HAS_SCALAR_DADD true
