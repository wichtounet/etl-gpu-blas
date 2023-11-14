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

TEST_CASE("scalar_set/s/0", "[float][scalar_set]") {
    const size_t N = 123;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_scalar_sset(gpu_vec, N, 1, 1.5f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i] == Approx(1.5f));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_set/d/0", "[double][scalar_set]") {
    const size_t N = 254;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_scalar_dset(gpu_vec, N, 1, 2.2f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i] == Approx(2.2));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_set/d/1", "[double][scalar_set]") {
    const size_t N = 302;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_scalar_dadd(gpu_vec, N, 3, 2.2f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(cpu_vec[i] == Approx(i + 2.2));
        } else {
            REQUIRE(cpu_vec[i] == Approx(i));
        }
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}
