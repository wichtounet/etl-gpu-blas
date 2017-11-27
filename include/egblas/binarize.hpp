//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Binarize a matrix or a vector
 * \param n The size of the vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param threshold The threshold for binarization
 */
void egblas_sbinarize(size_t n, float* x, size_t incx, float threshold);

/*!
 * \brief Binarize a matrix or a vector
 * \param n The size of the vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param threshold The threshold for binarization
 */
void egblas_dbinarize(size_t n, double* x, size_t incx, double threshold);

#define EGBLAS_HAS_SBINARIZE true
#define EGBLAS_HAS_DBINARIZE true
