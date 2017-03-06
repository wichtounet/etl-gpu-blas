//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/reduce.h>

#include "cpp_utils/assert.hpp"

float egblas_ssum(float* x, size_t n, size_t s){
    cpp_assert(s == 1, "Stride is not yet supported for egblas_ssum");
    cpp_unused(s);

    return thrust::reduce(x, x + n);
}

float egblas_dsum(double* x, size_t n, size_t s){
    cpp_assert(s == 1, "Stride is not yet supported for egblas_dsum");
    cpp_unused(s);

    return thrust::reduce(x, x + n);
}
