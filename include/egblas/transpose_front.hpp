//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

void egblas_stranspose_front(size_t m, size_t n, size_t k, float* x, float* y);
void egblas_dtranspose_front(size_t m, size_t n, size_t k, double* x, double* y);

#define EGBLAS_HAS_STRANSPOSE_FRONT true
#define EGBLAS_HAS_DTRANSPOSE_FRONT true
