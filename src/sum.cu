//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/reduce.h>

#include "egblas/assert.hpp"

float egblas_ssum(float* x, size_t n, size_t s){
    egblas_assert(s == 1, "Stride is not yet supported for egblas_ssum");
    egblas_unused(s);

    return thrust::reduce(x, x + n);
}

float egblas_dsum(double* x, size_t n, size_t s){
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dsum");
    egblas_unused(s);

    return thrust::reduce(x, x + n);
}
