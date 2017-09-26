//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// Binary functions
#include "egblas/axdy.hpp"
#include "egblas/axmy.hpp"
#include "egblas/pow.hpp"
#include "egblas/pow_yx.hpp"

// Scalar manipulations
#include "egblas/scalar_set.hpp"
#include "egblas/scalar_add.hpp"
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

// Reductions
#include "egblas/sum.hpp"

// Machine Learning Special functions
#include "egblas/cce.hpp"
#include "egblas/relu_der_out.hpp"
