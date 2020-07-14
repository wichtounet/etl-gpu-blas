//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
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

TEST_CASE("mse/loss/s/0", "[float][mse]") {
    const size_t N = 100;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.9 / (i + 1); // output
        y_cpu[i] = (i + 1); // labels
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    float loss = egblas_mse_sloss(N, 1.1f, x_gpu, 1, y_gpu, 1);

    REQUIRE(loss == Approx(3381.71 * 1.1f));

    auto both = egblas_smse(N, 1.1f, 1.1f, x_gpu, 1, y_gpu, 1);
    REQUIRE(both.first == Approx(loss));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("mse/loss/d/0", "[double][mse]") {
    const size_t N = 188;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.01 + (i+1) / (N + 13.0f);
        y_cpu[i] = (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    double loss = egblas_mse_dloss(N, 1.2, x_gpu, 1, y_gpu, 1);

    REQUIRE(loss == Approx(11755.7 * 1.2));

    auto both = egblas_dmse(N, 1.2, 1.2, x_gpu, 1, y_gpu, 1);
    REQUIRE(both.first == Approx(loss));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("mse/loss/d/1", "[double][mse]") {
    const size_t N = 1024;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.0001 + (i + 1) / (N + 16.0f);
        y_cpu[i] = (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    double loss = egblas_mse_dloss(N, 1.2, x_gpu, 1, y_gpu, 1);

    REQUIRE(loss == Approx(349365.0 * 1.2));

    auto both = egblas_dmse(N, 1.2, 1.2, x_gpu, 1, y_gpu, 1);
    REQUIRE(both.first == Approx(loss));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("mse/error/s/0", "[float][mse]") {
    const size_t N = 128 * 9;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = (i + 1) % 11;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    float error = egblas_mse_serror(N, 1.0f / 128.0f, x_gpu, 1, y_gpu, 1);

    REQUIRE(error == Approx(4.4649));

    auto both = egblas_smse(N, 1.2, 1.0f / 128.0f, x_gpu, 1, y_gpu, 1);
    REQUIRE(both.second == Approx(error));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("mse/error/d/0", "[double][mse]") {
    const size_t N = 145;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.01 * (i + 1);
        y_cpu[i] = (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    double error = egblas_mse_derror(N, 1.2, x_gpu, 1, y_gpu, 1);

    REQUIRE(error == Approx(86.724));

    auto both = egblas_dmse(N, 1.9, 1.2f, x_gpu, 1, y_gpu, 1);
    REQUIRE(both.second == Approx(error));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("mse/error/d/1", "[double][mse]") {
    const size_t N = 13 * 1024;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.01 * (i + 1);
        y_cpu[i] = (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    double error = egblas_mse_derror(N, 1.2, x_gpu, 1, y_gpu, 1);

    REQUIRE(error == Approx(7907.9225));

    auto both = egblas_dmse(N, 1.9, 1.2, x_gpu, 1, y_gpu, 1);
    REQUIRE(both.second == Approx(error));

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
