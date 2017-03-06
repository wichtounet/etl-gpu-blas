//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"
#include "test.hpp"

#include "catch.hpp"

TEST_CASE( "sum/float/0", "[float][sum]"){
    const size_t N = 123;

    float* cpu_vec = new float[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = i;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void **)&gpu_vec, N*sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N*sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_ssum(gpu_vec, N, 1) == Approx(float(7503)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE( "sum/float/1", "[float][sum]"){
    const size_t N = 389;

    float* cpu_vec = new float[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void **)&gpu_vec, N*sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N*sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_ssum(gpu_vec, N, 1) == Approx(float(116972)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE( "sum/double/0", "[double][sum]"){
    const size_t N = 123;

    double* cpu_vec = new double[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void **)&gpu_vec, N*sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N*sizeof(double), cudaMemcpyHostToDevice));

    REQUIRE(egblas_dsum(gpu_vec, N, 1) == Approx(double(7503)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE( "sum/double/1", "[double][sum]"){
    const size_t N = 389;

    double* cpu_vec = new double[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void **)&gpu_vec, N*sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N*sizeof(double), cudaMemcpyHostToDevice));

    REQUIRE(egblas_dsum(gpu_vec, N, 1) == Approx(double(116972)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}
