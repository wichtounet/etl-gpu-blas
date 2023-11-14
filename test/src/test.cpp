//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch_complex_approx.hpp"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

float large_eps = std::numeric_limits<float>::epsilon() * 20000;
float half_eps = std::numeric_limits<float>::epsilon() * 100000;
