//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute the single-precision standard deviation of the given vector
 * \param x The vector to compute the standard deviation from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the standard deviation of the elements of the vector
 */
float egblas_sstddev_mean(float* x, size_t n, size_t s, double mean);

/*!
 * \brief Compute the double-precision standard deviation of the given vector
 * \param x The vector to compute the standard deviation from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the standard deviation of the elements of the vector
 */
double egblas_dstddev_mean(double* x, size_t n, size_t s, double mean);

/*!
 * \brief Compute the single-precision standard deviation of the given vector
 * \param x The vector to compute the standard deviation from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the standard deviation of the elements of the vector
 */
float egblas_sstddev(float* x, size_t n, size_t s);

/*!
 * \brief Compute the double-precision standard deviation of the given vector
 * \param x The vector to compute the standard deviation from (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \return the standard deviation of the elements of the vector
 */
double egblas_dstddev(double* x, size_t n, size_t s);

#define EGBLAS_HAS_SSTDDEV true
#define EGBLAS_HAS_DSTDDEV true
