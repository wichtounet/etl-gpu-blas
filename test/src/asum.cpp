//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <complex>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"
#include "test.hpp"

#include "catch.hpp"

TEST_CASE("asum/s/0", "[float][asum]") {
    const size_t N = 257;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = -float(i + 1);
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_sasum(gpu_vec, N, 1) == Approx(float((N * (N + 1)) / 2)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("asum/s/1", "[float][asum]") {
    const size_t N = 389;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = -(3.1f * i) / 2.0;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_sasum(gpu_vec, N, 1) == Approx(float(116972)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("asum/s/2", "[float][asum]") {
    const size_t N = 1024 * 128 + 5;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = -float(i + 1);
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_sasum(gpu_vec, N, 1) == Approx(float((N * (N + 1)) / 2)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("asum/s/3", "[float][asum]") {
    const size_t N = 1024 * 1024 * 6 + 126;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = -float(i + 1);
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_sasum(gpu_vec, N, 1) == Approx(float((N * (N + 1)) / 2)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("asum/d/0", "[double][asum]") {
    const size_t N = 123;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    REQUIRE(egblas_dasum(gpu_vec, N, 1) == Approx(double(7503)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("asum/d/1", "[double][asum]") {
    const size_t N = 389;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    REQUIRE(egblas_dasum(gpu_vec, N, 1) == Approx(double(116972)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}
