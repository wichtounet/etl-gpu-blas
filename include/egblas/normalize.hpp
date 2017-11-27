//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Normalize a matrix or a vector
 * \param n The size of the vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_snormalize_flat(size_t n, float* x, size_t incx);

/*!
 * \brief Normalize a matrix or a vector
 * \param n The size of the vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_dnormalize_flat(size_t n, double* x, size_t incx);

/*!
 * \brief Normalize a matrix or a vector
 * \param n The size of the vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_snormalize_sub(size_t n, float* x, size_t sub_n, size_t incx);

/*!
 * \brief Normalize a matrix or a vector
 * \param n The size of the vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_dnormalize_sub(size_t n, double* x, size_t sub_n, size_t incx);

#define EGBLAS_HAS_SNORMALIZE_FLAT true
#define EGBLAS_HAS_DNORMALIZE_FLAT true
#define EGBLAS_HAS_SNORMALIZE_SUB true
#define EGBLAS_HAS_DNORMALIZE_SUB true
