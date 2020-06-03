//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute the single-precision minimum of the given vector
 * \param x The vector to compute the sum from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the minimum of the elements of the vector
 */
float egblas_smin(const float* x, size_t n, size_t s);

/*!
 * \brief Compute the double-precision minimum of the given vector
 * \param x The vector to compute the sum from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the minimum of the elements of the vector
 */
double egblas_dmin(const double* x, size_t n, size_t s);

#define EGBLAS_HAS_SMIN_REDUCE true
#define EGBLAS_HAS_DMIN_REDUCE true
