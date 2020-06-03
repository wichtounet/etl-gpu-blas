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

TEST_CASE("one_if_max_sub/s/0", "[float][one_if_max_sub]") {
    const size_t B = 3;
    const size_t N = 2;

    float* x_cpu = new float[B * N];
    float* y_cpu = new float[B * N];

    x_cpu[0] = -1.0f;
    x_cpu[1] = 1.0f;

    x_cpu[2] = 4.0f;
    x_cpu[3] = 6.0f;

    x_cpu[4] = 13.0f;
    x_cpu[5] = 6.0f;

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sone_if_max_sub(B, N, 1.1, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * N * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 0.0f);
    REQUIRE(y_cpu[1] == 1.1f);

    REQUIRE(y_cpu[2] == 0.0f);
    REQUIRE(y_cpu[3] == 1.1f);

    REQUIRE(y_cpu[4] == 1.1f);
    REQUIRE(y_cpu[5] == 0.0f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("one_if_max_sub/d/0", "[double][one_if_max_sub]") {
    const size_t B = 3;
    const size_t N = 3;

    double* x_cpu = new double[B * N];
    double* y_cpu = new double[B * N];

    x_cpu[0] = -1.0f;
    x_cpu[1] = 1.0f;
    x_cpu[2] = 4.0f;

    x_cpu[3] = 6.0f;
    x_cpu[4] = 13.0f;
    x_cpu[5] = 6.0f;

    x_cpu[6] = 29.0f;
    x_cpu[7] = 13.0f;
    x_cpu[8] = 6.0f;

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, B * N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_done_if_max_sub(B, N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * N * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 0.0);
    REQUIRE(y_cpu[1] == 0.0);
    REQUIRE(y_cpu[2] == 1.0);

    REQUIRE(y_cpu[3] == 0.0);
    REQUIRE(y_cpu[4] == 1.0);
    REQUIRE(y_cpu[5] == 0.0);

    REQUIRE(y_cpu[6] == 1.0);
    REQUIRE(y_cpu[7] == 0.0);
    REQUIRE(y_cpu[8] == 0.0);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
