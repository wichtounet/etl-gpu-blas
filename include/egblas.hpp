//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// Utility functions
#include "egblas/axdy.hpp"
#include "egblas/axpy.hpp"
#include "egblas/axmy.hpp"

// Binary functions
#include "egblas/pow.hpp"
#include "egblas/pow_yx.hpp"
#include "egblas/equal.hpp"
#include "egblas/not_equal.hpp"
#include "egblas/less.hpp"
#include "egblas/less_equal.hpp"
#include "egblas/greater.hpp"
#include "egblas/greater_equal.hpp"
#include "egblas/or.hpp"
#include "egblas/and.hpp"
#include "egblas/xor.hpp"

// Scalar manipulations
#include "egblas/scalar_set.hpp"
#include "egblas/scalar_add.hpp"
#include "egblas/scalar_mul.hpp"
#include "egblas/scalar_div.hpp"

// Unary element-wise functions
#include "egblas/sqrt.hpp"
#include "egblas/invsqrt.hpp"
#include "egblas/cbrt.hpp"
#include "egblas/invcbrt.hpp"
#include "egblas/log.hpp"
#include "egblas/log10.hpp"
#include "egblas/log2.hpp"
#include "egblas/exp.hpp"
#include "egblas/cos.hpp"
#include "egblas/sin.hpp"
#include "egblas/tan.hpp"
#include "egblas/cosh.hpp"
#include "egblas/sinh.hpp"
#include "egblas/tanh.hpp"
#include "egblas/minus.hpp"
#include "egblas/max.hpp"
#include "egblas/min.hpp"
#include "egblas/conj.hpp"
#include "egblas/imag.hpp"
#include "egblas/real.hpp"
#include "egblas/sign.hpp"
#include "egblas/softplus.hpp"
#include "egblas/abs.hpp"
#include "egblas/floor.hpp"
#include "egblas/ceil.hpp"
#include "egblas/clip.hpp"

// Reductions
#include "egblas/sum.hpp"

// Machine Learning Special functions
#include "egblas/cce.hpp"
#include "egblas/relu_der_out.hpp"

// Random
#include "egblas/shuffle.hpp"
#include "egblas/dropout.hpp"
