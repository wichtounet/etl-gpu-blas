//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

void egblas_sbatch_k_scale2(size_t b, size_t k, const float* x, const float* gamma, float* y);
void egblas_dbatch_k_scale2(size_t b, size_t k, const double* x, const double* gamma, double* y);

#define EGBLAS_HAS_SBATCH_K_SCALE2 true
#define EGBLAS_HAS_DBATCH_K_SCALE2 true

void egblas_sbatch_k_scale4(size_t b, size_t k, size_t m, size_t n, const float* x, const float* gamma, float* y);
void egblas_dbatch_k_scale4(size_t b, size_t k, size_t m, size_t n, const double* x, const double* gamma, double* y);

#define EGBLAS_HAS_SBATCH_K_SCALE4 true
#define EGBLAS_HAS_DBATCH_K_SCALE4 true

void egblas_sbatch_k_scale_plus2(size_t b, size_t k, const float* x, const float* gamma, const float* beta, float* y);
void egblas_dbatch_k_scale_plus2(size_t b, size_t k, const double* x, const double* gamma, const double* beta, double* y);

#define EGBLAS_HAS_SBATCH_K_SCALE_PLUS2 true
#define EGBLAS_HAS_DBATCH_K_SCALE_PLUS2 true

void egblas_sbatch_k_scale_plus4(size_t b, size_t k, size_t m, size_t n, const float* x, const float* gamma, const float * beta, float* y);
void egblas_dbatch_k_scale_plus4(size_t b, size_t k, size_t m, size_t n, const double* x, const double* gamma, const double* beta, double* y);

#define EGBLAS_HAS_SBATCH_K_SCALE_PLUS4 true
#define EGBLAS_HAS_DBATCH_K_SCALE_PLUS4 true

void egblas_sbatch_k_minus_scale2(size_t b, size_t k, const float* x, const float* gamma, const float* beta, float* y);
void egblas_dbatch_k_minus_scale2(size_t b, size_t k, const double* x, const double* gamma, const double* beta, double* y);

#define EGBLAS_HAS_SBATCH_K_MINUS_SCALE2 true
#define EGBLAS_HAS_DBATCH_K_MINUS_SCALE2 true

void egblas_sbatch_k_minus_scale4(size_t b, size_t k, size_t m, size_t n, const float* x, const float* gamma, const float * beta, float* y);
void egblas_dbatch_k_minus_scale4(size_t b, size_t k, size_t m, size_t n, const double* x, const double* gamma, const double* beta, double* y);

#define EGBLAS_HAS_SBATCH_K_MINUS_SCALE4 true
#define EGBLAS_HAS_DBATCH_K_MINUS_SCALE4 true
