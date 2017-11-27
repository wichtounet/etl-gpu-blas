//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/sum.hpp"
#include "egblas/mean.hpp"

float egblas_smean(float* x, size_t n, size_t s) {
    return egblas_ssum(x, n, s) / n;
}

double egblas_dmean(double* x, size_t n, size_t s) {
    return egblas_dsum(x, n, s) / n;
}
