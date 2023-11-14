//=======================================================================
// Copyright (c) 2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*
 * Utilties for half-precision
 */

#ifndef DISABLE_FP16

#include <cuda_fp16.h>

using fp16 = __half2;

#endif

#ifndef DISABLE_BF16

#include <cuda_bf16.h>

using bf16 = __nv_bfloat162;

#endif
