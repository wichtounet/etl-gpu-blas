//=======================================================================
// Copyright (c) 2017-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <utility>

/*!
 * \brief Compute Categorical-Cross Entropy Loss
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
float egblas_cce_sloss(size_t n, float alpha, const float* output, size_t incx, const float* labels, size_t incy);

/*!
 * \brief Compute Categorical-Cross Entropy Loss
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param incx The stride of output
 * \param labels The vector labels (GPU memory)
 * \param incy The stride of labels
 */
double egblas_cce_dloss(size_t n, double alpha, const double* output, size_t incx, const double* labels, size_t incy);

/*!
 * \brief Compute Categorical-Cross Entropy Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param labels The vector labels (GPU memory)
 */
float egblas_cce_serror(size_t n, size_t m, float alpha, const float* output, const float* labels);

/*!
 * \brief Compute Categorical-Cross Entropy Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param labels The vector labels (GPU memory)
 */
double egblas_cce_derror(size_t n, size_t m, double alpha, const double* output, const double* labels);

/*!
 * \brief Compute Categorical-Cross Entropy Loss and Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param labels The vector labels (GPU memory)
 */
std::pair<float, float> egblas_scce(size_t n, size_t m, float alpha, float beta, const float* output, const float* labels);

/*!
 * \brief Compute Categorical-Cross Entropy Loss and Error
 * \param n The size of the two vectors
 * \param alpha The multiplicator
 * \param output The vector output (GPU memory)
 * \param labels The vector labels (GPU memory)
 */
std::pair<double, double> egblas_dcce(size_t n, size_t m, double alpha, double beta, const double* output, const double* labels);

#define EGBLAS_HAS_CCE_SLOSS true
#define EGBLAS_HAS_CCE_DLOSS true

#define EGBLAS_HAS_CCE_SERROR true
#define EGBLAS_HAS_CCE_DERROR true

#define EGBLAS_HAS_SCCE true
#define EGBLAS_HAS_DCCE true
