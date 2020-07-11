//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// General Utility functions
#include "egblas/apxdbpy.hpp"   //  y = (alpha + x) / (beta + y)
#include "egblas/apxdbpy_3.hpp" // yy = (alpha + x) / (beta + y)
#include "egblas/apxdby.hpp"    //  y = (alpha + x) / (beta * y)
#include "egblas/apxdby_3.hpp"  // yy = (alpha + x) / (beta * y)
#include "egblas/axdbpy.hpp"    //  y = (alpha * x) / (beta + y)
#include "egblas/axdbpy_3.hpp"  // yy = (alpha * x) / (beta + y)
#include "egblas/axdy.hpp"      //  y = y / (alpha * x)
#include "egblas/axdy_3.hpp"    // yy = y / (alpha * x)
#include "egblas/axmy.hpp"      //  y = alpha * x * y
#include "egblas/axmy_3.hpp"    // yy = alpha * x * y
#include "egblas/axpby.hpp"     //  y = alpha * x + beta * y
#include "egblas/axpby_3.hpp"   // yy = alpha * x + beta * y
#include "egblas/axpy.hpp"      //  y = alpha * x + y
#include "egblas/axpy_3.hpp"    // yy = alpha * x + y

// Binary functions
#include "egblas/and.hpp"
#include "egblas/equal.hpp"
#include "egblas/greater.hpp"
#include "egblas/greater_equal.hpp"
#include "egblas/less.hpp"
#include "egblas/less_equal.hpp"
#include "egblas/not_equal.hpp"
#include "egblas/or.hpp"
#include "egblas/pow.hpp"
#include "egblas/pow_yx.hpp"
#include "egblas/xor.hpp"

// Scalar manipulations
#include "egblas/scalar_add.hpp"
#include "egblas/scalar_div.hpp"
#include "egblas/scalar_mul.hpp"
#include "egblas/scalar_set.hpp"

// Unary element-wise functions
#include "egblas/abs.hpp"
#include "egblas/cbrt.hpp"
#include "egblas/ceil.hpp"
#include "egblas/clip.hpp"
#include "egblas/conj.hpp"
#include "egblas/cos.hpp"
#include "egblas/cosh.hpp"
#include "egblas/exp.hpp"
#include "egblas/floor.hpp"
#include "egblas/imag.hpp"
#include "egblas/invcbrt.hpp"
#include "egblas/invsqrt.hpp"
#include "egblas/log.hpp"
#include "egblas/log10.hpp"
#include "egblas/log2.hpp"
#include "egblas/max.hpp"
#include "egblas/min.hpp"
#include "egblas/minus.hpp"
#include "egblas/real.hpp"
#include "egblas/sign.hpp"
#include "egblas/sin.hpp"
#include "egblas/sinh.hpp"
#include "egblas/softplus.hpp"
#include "egblas/sqrt.hpp"
#include "egblas/tan.hpp"
#include "egblas/tanh.hpp"
#include "egblas/sigmoid.hpp"
#include "egblas/one_if_max.hpp"
#include "egblas/one_if_max_sub.hpp"

// Reductions
#include "egblas/mean.hpp"
#include "egblas/stddev.hpp"
#include "egblas/sum.hpp"
#include "egblas/asum.hpp"
#include "egblas/bias_batch_sum.hpp"
#include "egblas/max_reduce.hpp"
#include "egblas/min_reduce.hpp"

// Special batch operations
#include "egblas/batch_k_scale.hpp"

// Machine Learning Special functions
#include "egblas/bias_add.hpp"
#include "egblas/cce.hpp"
#include "egblas/bce.hpp"
#include "egblas/relu_der_out.hpp"

// Random
#include "egblas/dropout.hpp"
#include "egblas/shuffle.hpp"
#include "egblas/logistic_noise.hpp"
#include "egblas/bernoulli.hpp"

// Transformations
#include "egblas/transpose_front.hpp"

// Modifications of vector/matrix
#include "egblas/binarize.hpp"
#include "egblas/normalize.hpp"
