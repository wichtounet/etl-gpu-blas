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

TEST_CASE("cce/loss/s/0", "[float][cce]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.1 * (i + 1);
        y_cpu[i] = (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    float loss = egblas_cce_sloss(N, 1.1, x_gpu, 1, y_gpu, 1);

    REQUIRE(loss == Approx(22055.71875f));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cce/loss/d/0", "[double][cce]") {
    const size_t N = 145;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.01 * (i + 1);
        y_cpu[i] = (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    double loss = egblas_cce_sloss(N, 1.2, x_gpu, 1, y_gpu, 1);

    REQUIRE(loss == Approx(-1587.1035));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cce/loss/d/1", "[double][cce]") {
    const size_t N = 13 * 1024;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.01 * (i + 1);
        y_cpu[i] = (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    double loss = egblas_cce_sloss(N, 1.2, x_gpu, 1, y_gpu, 1);

    REQUIRE(loss == Approx(466941536.0));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cce/error/s/0", "[float][cce]") {
    const size_t N = 137;
    const size_t M = 8;

    float* x_cpu = new float[N * M];
    float* y_cpu = new float[N * M];

    for (size_t i = 0; i < N * M; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = ((i + 1) % 9);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * M * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * M * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * M * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * M * sizeof(float), cudaMemcpyHostToDevice));

    float loss = egblas_cce_serror(N, M, 1.0 / 137.0f, x_gpu, y_gpu);

    REQUIRE(loss == Approx(0.76642));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cce/error/d/0", "[double][cce]") {
    const size_t N = 128;
    const size_t M = 9;

    double* x_cpu = new double[N * M];
    double* y_cpu = new double[N * M];

    for (size_t i = 0; i < N * M; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = ((i + 1) % 11);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * M * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * M * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * M * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * M * sizeof(double), cudaMemcpyHostToDevice));

    double loss = egblas_cce_derror(N, M, 1.0 / 128.0f, x_gpu, y_gpu);

    REQUIRE(loss == Approx(0.71875));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cce/error/d/1", "[double][cce]") {
    const size_t N = 3 * 1024;
    const size_t M = 9;

    double* x_cpu = new double[N * M];
    double* y_cpu = new double[N * M];

    for (size_t i = 0; i < N * M; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = ((i + 1) % 11);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * M * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * M * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * M * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * M * sizeof(double), cudaMemcpyHostToDevice));

    double loss = egblas_cce_derror(N, M, 1.0 / 128.0f, x_gpu, y_gpu);

    REQUIRE(loss == Approx(17.453125));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
