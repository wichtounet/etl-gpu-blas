//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Compute the single-precision mean of the given vector
 * \param x The vector to compute the mean from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the mean of the elements of the vector
 */
float egblas_smean(float* x, size_t n, size_t s);

/*!
 * \brief Compute the double-precision mean of the given vector
 * \param x The vector to compute the mean from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the mean of the elements of the vector
 */
double egblas_dmean(double* x, size_t n, size_t s);

#define EGBLAS_HAS_SMEAN true
#define EGBLAS_HAS_DMEAN true
