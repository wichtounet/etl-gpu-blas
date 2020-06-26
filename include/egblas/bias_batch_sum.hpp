//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

void egblas_sbias_batch_sum(size_t b, size_t n, float* x, size_t incx, float* y, size_t incy);
void egblas_dbias_batch_sum(size_t b, size_t n, double* x, size_t incx, double* y, size_t incy);

void egblas_sbias_batch_mean(size_t b, size_t n, float* x, size_t incx, float* y, size_t incy);
void egblas_dbias_batch_mean(size_t b, size_t n, double* x, size_t incx, double* y, size_t incy);

#define EGBLAS_HAS_SBIAS_BATCH_SUM true
#define EGBLAS_HAS_DBIAS_BATCH_SUM true

#define EGBLAS_HAS_SBIAS_BATCH_MEAN true
#define EGBLAS_HAS_DBIAS_BATCH_MEAN true

void egblas_sbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, float* x, float* y);
void egblas_dbias_batch_sum4(size_t b, size_t n, size_t w, size_t h, double* x, double* y);

void egblas_sbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, float* x, float* y);
void egblas_dbias_batch_mean4(size_t b, size_t n, size_t w, size_t h, double* x, double* y);

#define EGBLAS_HAS_SBIAS_BATCH_SUM4 true
#define EGBLAS_HAS_DBIAS_BATCH_SUM4 true

#define EGBLAS_HAS_SBIAS_BATCH_MEAN4 true
#define EGBLAS_HAS_DBIAS_BATCH_MEAN4 true
