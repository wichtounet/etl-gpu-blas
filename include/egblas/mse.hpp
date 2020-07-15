//=======================================================================
// Copyright (c) 2017-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <utility>

/*!
 * \brief Compute Mean Squared Error Loss
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
float egblas_mse_sloss(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Mean Squared Error Loss
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
double egblas_mse_dloss(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy);

/*!
 * \brief Compute Mean Squared Error Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
float egblas_mse_serror(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Mean Squared Error Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
double egblas_mse_derror(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy);

/*!
 * \brief Compute Mean Squared Error Loss and Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
std::pair<float, float> egblas_smse(size_t n, float alpha, float beta, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Mean Squared Error Loss and Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
std::pair<double, double> egblas_dmse(size_t n, double alpha, double beta, const double* output, size_t incx, const double* labels, size_t incy);

#define EGBLAS_HAS_MSE_SLOSS true
#define EGBLAS_HAS_MSE_DLOSS true

#define EGBLAS_HAS_MSE_SERROR true
#define EGBLAS_HAS_MSE_DERROR true

#define EGBLAS_HAS_SMSE true
#define EGBLAS_HAS_DMSE true
