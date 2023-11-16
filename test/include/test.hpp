//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <complex>
#include <iostream>
#include <type_traits>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"
//#include "half.hpp"

#include "catch_complex_approx.hpp"
#include "doctest/doctest.h"

#undef TEST_CASE
#define TEST_CASE(name, tags) DOCTEST_TEST_CASE(tags " " name)

using doctest::Approx;

#define cuda_check(call)                                                                                \
    {                                                                                                   \
        auto status = call;                                                                             \
        if (status != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                           \
            FAIL("Cuda ERROR"); \
        }                                                                                               \
    }

#if !defined(DISABLE_FP16) && !defined(DISABLE_BF16)
#define TEST_CASE_HALF(name) TEST_CASE_TEMPLATE(name, T, fp16, bf16)
#define TEST_HALF
#elif !defined(DISABLE_FP16)
#define TEST_CASE_HALF(name) TEST_CASE_TEMPLATE(name, T, fp16)
#define TEST_HALF
#elif !defined(DISABLE_BF16)
#define TEST_CASE_HALF(name) TEST_CASE_TEMPLATE(name, T, bf16)
#define TEST_HALF
#endif

extern float large_eps;
extern float half_eps;

/*!
 * \brief Utility for testing on dual (CPU/GPU) arrays
 */
template<typename T>
struct dual_array {
    T* _cpu;
    T* _gpu;
    const size_t N;

    explicit dual_array(size_t N) : N(N) {
        _cpu = new T[N];

        cuda_check(cudaMalloc((void**)&_gpu, N * sizeof(T)));
    }

    ~dual_array(){
        cuda_check(cudaFree(_gpu));

        delete[] _cpu;
    }

    void cpu_to_gpu(){
        cuda_check(cudaMemcpy(_gpu, _cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    }

    void gpu_to_cpu(){
        cuda_check(cudaMemcpy(_cpu, _gpu, N * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* cpu(){
        return _cpu;
    }

    T* gpu(){
        return _gpu;
    }

    template<typename TT = T, typename std::enable_if<std::is_same<TT, std::complex<float>>::value, int>::type = 42>
    cuComplex* complex_gpu(){
        return reinterpret_cast<cuComplex*>(_gpu);
    }

    template<typename TT = T, typename std::enable_if<std::is_same<TT, std::complex<double>>::value, int>::type = 42>
    cuDoubleComplex* complex_gpu(){
        return reinterpret_cast<cuDoubleComplex*>(_gpu);
    }
};
