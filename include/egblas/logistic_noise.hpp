//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Compute the single-precision logistic noise of x
 *
 * y = alpha * (x + N(0, logistic_sigmoid(x)))
 *
 * This function is not really efficient since it will allocate and release
 * curand states.
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_slogistic_noise(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy);

/*!
 * \brief Compute the double-precision logistic noise of x
 *
 * y = alpha * (x + N(0, logistic_sigmoid(x)))
 *
 * This function is not really efficient since it will allocate and release
 * curand states.
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
void egblas_dlogistic_noise(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy);

/*!
 * \brief Compute the single-precision logistic noise of x with the given seed
 *
 * y = alpha * (x + N(0, logistic_sigmoid(x)))
 *
 * This function is not really efficient since it will allocate and release
 * curand states.
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param seed The seed to start with
 */
void egblas_slogistic_noise_seed(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy, size_t seed);

/*!
 * \brief Compute the double-precision logistic noise of x with the given seed
 *
 * y = alpha * (x + N(0, logistic_sigmoid(x)))
 *
 * This function is not really efficient since it will allocate and release
 * curand states.
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param seed The seed to start with
 */
void egblas_dlogistic_noise_seed(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy, size_t seed);

/*!
 * \brief Compute the single-precision logistic noise of x
 *
 * y = alpha * (x + N(0, logistic_sigmoid(x)))
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param states The curand states (from prepare)
 */
void egblas_slogistic_noise_states(size_t n, float alpha, const float* x, size_t incx, float* y, size_t incy, void* states);

/*!
 * \brief Compute the double-precision logistic noise of x
 *
 * y = alpha * (x + N(0, logistic_sigmoid(x)))
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 * \param states The curand states (from prepare)
 */
void egblas_dlogistic_noise_states(size_t n, double alpha, const double* x, size_t incx, double* y, size_t incy, void* states);

/*!
 * \brief Prepare the curand states for the logistic noise operation
 *
 * \return the newly prepared states
 */
void* egblas_logistic_noise_prepare();

/*!
 * \brief Prepare the curand states for the logistic noise operation
 *
 * \param seed The Seed to initialize the states with
 *
 * \return the newly prepared states
 */
void* egblas_logistic_noise_prepare_seed(size_t seed);

/*!
 * \brief Release the curand states for the logistic noise operation
 *
 * \param state The states to release
 */
void egblas_logistic_noise_release(void* state);

#define EGBLAS_HAS_LOGISTIC_NOISE_PREPARE true
#define EGBLAS_HAS_LOGISTIC_NOISE_PREPARE_SEED true
#define EGBLAS_HAS_LOGISTIC_NOISE_RELEASE true

#define EGBLAS_HAS_SLOGISTIC_NOISE true
#define EGBLAS_HAS_DLOGISTIC_NOISE true

#define EGBLAS_HAS_SLOGISTIC_NOISE_SEED true
#define EGBLAS_HAS_DLOGISTIC_NOISE_SEED true

#define EGBLAS_HAS_SLOGISTIC_NOISE_STATES true
#define EGBLAS_HAS_DLOGISTIC_NOISE_STATES true
