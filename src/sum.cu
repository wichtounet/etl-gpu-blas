//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/sum.hpp"
#include "egblas/assert.hpp"

float egblas_ssum(float* x, size_t n, size_t s){
    egblas_assert(s == 1, "Stride is not yet supported for egblas_ssum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n);
}

double egblas_dsum(double* x, size_t n, size_t s){
    egblas_assert(s == 1, "Stride is not yet supported for egblas_dsum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n);
}

struct single_complex_add {
    __device__ cuComplex operator()(cuComplex x, cuComplex y){
        return cuCaddf(x, y);
    }
};

struct double_complex_add {
    __device__ cuDoubleComplex operator()(cuDoubleComplex x, cuDoubleComplex y){
        return cuCadd(x, y);
    }
};

cuComplex egblas_csum(cuComplex* x, size_t n, size_t s){
    egblas_assert(s == 1, "Stride is not yet supported for egblas_csum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n, make_cuComplex(0, 0), single_complex_add());
}

cuDoubleComplex egblas_zsum(cuDoubleComplex* x, size_t n, size_t s){
    egblas_assert(s == 1, "Stride is not yet supported for egblas_zsum");
    egblas_unused(s);

    return thrust::reduce(thrust::device, x, x + n, make_cuDoubleComplex(0, 0), double_complex_add());
}
