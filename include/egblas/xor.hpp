//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = a != b (element wise)
 * \param n The size of the three vectors
 * \param a The vector a (GPU memory)
 * \param inca The stride of a
 * \param b The vector z (GPU memory)
 * \param incb The stride of b
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_bxor(size_t n, const bool* x, size_t incx, const bool* z, size_t incz, bool* y, size_t incy);

#define EGBLAS_HAS_BXOR true
