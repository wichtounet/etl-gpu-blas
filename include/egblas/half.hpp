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

template <typename T>
T fromFloat(float f);

#ifndef DISABLE_FP16

#include <cuda_fp16.h>

using fp16 = __half2;

template<>
fp16 fromFloat<fp16>(float f){
    return __float2half2_rn(f);
}

#endif

#ifndef DISABLE_BF16

#include <cuda_bf16.h>

using bf16 = __nv_bfloat162;

template<>
bf16 fromFloat<bf16>(float f){
    return __float2bfloat162_rn(f);
}

#endif
