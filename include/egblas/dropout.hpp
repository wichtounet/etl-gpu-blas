//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

/*!
 * \brief Compute a single precision dropout mask for the given probability
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_sdropout(size_t n, float p, float alpha, float* x, size_t incx);

/*!
 * \brief Compute a double precision, dropout mask for the given probability
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_ddropout(size_t n, double p, double alpha, double* x, size_t incx);

/*!
 * \brief Compute a single precision dropout mask for the given probability
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_sdropout_seed(size_t n, float p, float alpha, float* x, size_t incx, size_t seed);

/*!
 * \brief Compute a double precision, dropout mask for the given probability
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_ddropout_seed(size_t n, double p, double alpha, double* x, size_t incx, size_t seed);

/*!
 * \brief Compute a single precision dropout mask for the given probability
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_sdropout_states(size_t n, float p, float alpha, float* x, size_t incx, void* states);

/*!
 * \brief Compute a double precision, dropout mask for the given probability
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_ddropout_states(size_t n, double p, double alpha, double* x, size_t incx, void* states);

/*!
 * \brief Prepare random states for dropout
 * \return random states for dropout
 */
void* egblas_dropout_prepare();

/*!
 * \brief Prepare random states for dropout with the given seed
 * \param seed The seed
 * \return random states for dropout
 */
void* egblas_dropout_prepare_seed(size_t seed);

/*!
 * \brief Release random states for dropout
 * \param states The random states
 */
void egblas_dropout_release(void* state);

/*!
 * \brief Compute a single precision inverted dropout mask for the given probability
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_sinv_dropout(size_t n, float p, float alpha, float* x, size_t incx);

/*!
 * \brief Compute a double precision inverted dropout mask for the given probability
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 */
void egblas_dinv_dropout(size_t n, double p, double alpha, double* x, size_t incx);

/*!
 * \brief Compute a single precision inverted dropout mask for the given probability
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_sinv_dropout_seed(size_t n, float p, float alpha, float* x, size_t incx, size_t seed);

/*!
 * \brief Compute a double precision inverted dropout mask for the given probability
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_dinv_dropout_seed(size_t n, double p, double alpha, double* x, size_t incx, size_t seed);

/*!
 * \brief Compute a single precision inverted dropout mask for the given probability
 *
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_sinv_dropout_states(size_t n, float p, float alpha, float* x, size_t incx, void* states);

/*!
 * \brief Compute a double precision inverted dropout mask for the given probability
 * \param n The size of the vector
 * \param alpha The multiplicator
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param seed The seed to start the random generation at
 */
void egblas_dinv_dropout_states(size_t n, double p, double alpha, double* x, size_t incx, void* states);

#define EGBLAS_HAS_DROPOUT_PREPARE true
#define EGBLAS_HAS_DROPOUT_PREPARE_SEED true
#define EGBLAS_HAS_DROPOUT_RELEASE true

#define EGBLAS_HAS_SDROPOUT true
#define EGBLAS_HAS_DDROPOUT true

#define EGBLAS_HAS_SDROPOUT_SEED true
#define EGBLAS_HAS_DDROPOUT_SEED true

#define EGBLAS_HAS_SDROPOUT_STATES true
#define EGBLAS_HAS_DDROPOUT_STATES true

#define EGBLAS_HAS_SINV_DROPOUT true
#define EGBLAS_HAS_DINV_DROPOUT true

#define EGBLAS_HAS_SINV_DROPOUT_SEED true
#define EGBLAS_HAS_DINV_DROPOUT_SEED true

#define EGBLAS_HAS_SINV_DROPOUT_STATES true
#define EGBLAS_HAS_DINV_DROPOUT_STATES true
