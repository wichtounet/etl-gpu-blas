//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute the single-precision maximum of the given vector
 * \param x The vector to compute the sum from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the maximum of the elements of the vector
 */
float egblas_smax(float* x, size_t n, size_t s);

/*!
 * \brief Compute the double-precision maximum of the given vector
 * \param x The vector to compute the sum from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the maximum of the elements of the vector
 */
double egblas_dmax(double* x, size_t n, size_t s);

#define EGBLAS_HAS_SMAX_REDUCE true
#define EGBLAS_HAS_DMAX_REDUCE true
