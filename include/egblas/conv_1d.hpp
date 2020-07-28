//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute y = alpha * conv_1d_valid(x, y), in single-precision
 * \param N The input size
 * \param K The kernel size
 * \param alpha The scaling factor
 * \param x The input vector
 * \param incx The step of the input vector
 * \param k The kernel vector
 * \param inck The step of the kernel vector
 * \param y The output vector
 * \param incy The step of the output vector
 */
void egblas_sconv1_valid(size_t N, size_t K, float alpha, const float* x, size_t incx, const float * k, size_t inck, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * conv_1d_valid(x, y), in double-precision
 * \param N The input size
 * \param K The kernel size
 * \param alpha The scaling factor
 * \param x The input vector
 * \param incx The step of the input vector
 * \param k The kernel vector
 * \param inck The step of the kernel vector
 * \param y The output vector
 * \param incy The step of the output vector
 */
void egblas_dconv1_valid(size_t N, size_t K, double alpha, const double* x, size_t incx, const double * k, size_t inck, double* y, size_t incy);

/*!
 * \brief Compute y = alpha * conv_1d_same(x, y), in single-precision
 * \param N The input size
 * \param K The kernel size
 * \param alpha The scaling factor
 * \param x The input vector
 * \param incx The step of the input vector
 * \param k The kernel vector
 * \param inck The step of the kernel vector
 * \param y The output vector
 * \param incy The step of the output vector
 */
void egblas_sconv1_same(size_t N, size_t K, float alpha, const float* x, size_t incx, const float * k, size_t inck, float* y, size_t incy);

/*!
 * \brief Compute y = alpha * conv_1d_same(x, y), in double-precision
 * \param N The input size
 * \param K The kernel size
 * \param alpha The scaling factor
 * \param x The input vector
 * \param incx The step of the input vector
 * \param k The kernel vector
 * \param inck The step of the kernel vector
 * \param y The output vector
 * \param incy The step of the output vector
 */
void egblas_dconv1_same(size_t N, size_t K, double alpha, const double* x, size_t incx, const double * k, size_t inck, double* y, size_t incy);

#define EGBLAS_HAS_SCONV1_VALID true
#define EGBLAS_HAS_DCONV1_VALID true
#define EGBLAS_HAS_SCONV1_SAME true
#define EGBLAS_HAS_DCONV1_SAME true
