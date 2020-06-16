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

TEST_CASE("bias_batch_sum/s/0", "[float][bias_batch_sum]") {
    const size_t B = 2;
    const size_t N = 3;

    float* x_cpu = new float[B * N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < B * N; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 3.0f);
    REQUIRE(y_cpu[1] == 5.0f);
    REQUIRE(y_cpu[2] == 7.0f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_sum/s/1", "[float][bias_batch_sum]") {
    const size_t B = 128;
    const size_t N = 257;

    float* x_cpu = new float[B * N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < N; ++j) {
            x_cpu[i * N + j] = j;
        }
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < N; ++j) {
        REQUIRE(y_cpu[j] == float(B * j));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_sum/d/0", "[double][bias_batch_sum]") {
    const size_t B = 2;
    const size_t N = 3;

    double* x_cpu = new double[B * N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < B * N; ++i) {
        x_cpu[i] = i;
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_sum(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 3.0f);
    REQUIRE(y_cpu[1] == 5.0f);
    REQUIRE(y_cpu[2] == 7.0f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_sum/d/1", "[double][bias_batch_sum]") {
    const size_t B = 128;
    const size_t N = 257;

    double* x_cpu = new double[B * N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < N; ++j) {
            x_cpu[i * N + j] = j;
        }
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_sum(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < N; ++j) {
        REQUIRE(y_cpu[j] == double(B * j));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_mean/s/0", "[float][bias_batch_mean]") {
    const size_t B = 2;
    const size_t N = 3;

    float* x_cpu = new float[B * N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < B * N; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_mean(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 3.0f / B);
    REQUIRE(y_cpu[1] == 5.0f / B);
    REQUIRE(y_cpu[2] == 7.0f / B);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_mean/s/1", "[float][bias_batch_mean]") {
    const size_t B = 128;
    const size_t N = 257;

    float* x_cpu = new float[B * N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < N; ++j) {
            x_cpu[i * N + j] = j;
        }
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_mean(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < N; ++j) {
        REQUIRE(y_cpu[j] == float(B * j / B));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_mean/d/0", "[double][bias_batch_mean]") {
    const size_t B = 2;
    const size_t N = 3;

    double* x_cpu = new double[B * N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < B * N; ++i) {
        x_cpu[i] = i;
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_mean(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 3.0f / B);
    REQUIRE(y_cpu[1] == 5.0f / B);
    REQUIRE(y_cpu[2] == 7.0f / B);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_mean/d/1", "[double][bias_batch_mean]") {
    const size_t B = 128;
    const size_t N = 257;

    double* x_cpu = new double[B * N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < N; ++j) {
            x_cpu[i * N + j] = j;
        }
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_mean(B, N, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < N; ++j) {
        REQUIRE(y_cpu[j] == double(B * j / B));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
