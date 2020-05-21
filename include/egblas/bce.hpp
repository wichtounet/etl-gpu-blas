//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <utility>

/*!
 * \brief Compute Binary-Cross Entropy Loss
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
float egblas_bce_sloss(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Binary-Cross Entropy Loss
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
double egblas_bce_dloss(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy);

/*!
 * \brief Compute Binary-Cross Entropy Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
float egblas_bce_serror(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Binary-Cross Entropy Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
double egblas_bce_derror(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy);

/*!
 * \brief Compute Binary-Cross Entropy Loss and Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
std::pair<float, float> egblas_sbce(size_t n, float alpha, float beta, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Binary-Cross Entropy Loss and Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
std::pair<double, double> egblas_dbce(size_t n, double alpha, double beta, const double* output, size_t incx, const double* labels, size_t incy);

#define EGBLAS_HAS_BCE_SLOSS true
#define EGBLAS_HAS_BCE_DLOSS true

#define EGBLAS_HAS_BCE_SERROR true
#define EGBLAS_HAS_BCE_DERROR true

#define EGBLAS_HAS_SBCE true
#define EGBLAS_HAS_DBCE true
