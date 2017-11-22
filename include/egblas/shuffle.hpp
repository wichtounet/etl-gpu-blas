//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Shuffle GPU memory
 * \param n The size of the three vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_shuffle(size_t n, void* x, size_t incx);

/*!
 * \brief Shuffle GPU memory
 * \param n The size of the three vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_shuffle_seed(size_t n, void* x, size_t incx, size_t seed);

/*!
 * \brief Shuffle GPU memory
 * \param n The size of the three vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_par_shuffle(size_t n, void* x, size_t incx, void* y, size_t incy);

/*!
 * \brief Shuffle GPU memory
 * \param n The size of the three vectors
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_par_shuffle_seed(size_t n, void* x, size_t incx, void* y, size_t incy, size_t seed);

#define EGBLAS_HAS_SHUFFLE true
#define EGBLAS_HAS_SHUFFLE_SEED true
#define EGBLAS_HAS_PAR_SHUFFLE true
#define EGBLAS_HAS_PAR_SHUFFLE_SEED true
