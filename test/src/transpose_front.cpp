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

TEST_CASE("transpose_front/s/0", "[float][bias_batch_sum]") {
    const size_t M = 2;
    const size_t N = 3;
    const size_t K = 3;

    float* x_cpu = new float[M * N * K];
    float* y_cpu = new float[M * N * K];

    for (size_t i = 0; i < M * N * K; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, M * N * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * K * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * K * sizeof(float), cudaMemcpyHostToDevice));

    egblas_stranspose_front(M, N, K, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * K * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(x_cpu[0] == 0.0f);
    REQUIRE(x_cpu[1] == 1.0f);
    REQUIRE(x_cpu[2] == 2.0f);

    REQUIRE(y_cpu[3] == 9.0f);
    REQUIRE(y_cpu[4] == 10.0f);
    REQUIRE(y_cpu[5] == 11.0f);

    REQUIRE(y_cpu[6] == 3.0f);
    REQUIRE(y_cpu[7] == 4.0f);
    REQUIRE(y_cpu[8] == 5.0f);

    REQUIRE(y_cpu[9] == 12.0f);
    REQUIRE(y_cpu[10] == 13.0f);
    REQUIRE(y_cpu[11] == 14.0f);

    REQUIRE(y_cpu[12] == 6.0f);
    REQUIRE(y_cpu[13] == 7.0f);
    REQUIRE(y_cpu[14] == 8.0f);

    REQUIRE(y_cpu[15] == 15.0f);
    REQUIRE(y_cpu[16] == 16.0f);
    REQUIRE(y_cpu[17] == 17.0f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("transpose_front/s/1", "[float][bias_batch_sum]") {
    const size_t M = 129;
    const size_t N = 28;
    const size_t K = 24;

    float* x_cpu = new float[M * N * K];
    float* y_cpu = new float[M * N * K];
    float* y_ref = new float[M * N * K];

    for (size_t i = 0; i < M * N * K; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, M * N * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * K * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * K * sizeof(float), cudaMemcpyHostToDevice));

    egblas_stranspose_front(M, N, K, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * K * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                y_ref[n * (M * K) + m * K + k] = x_cpu[m * (N * K) + n * K + k];
            }
        }
    }

    for (size_t i = 0; i < M * N * K; ++i) {
        REQUIRE(y_cpu[i] == y_ref[i]);
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_ref;
}

TEST_CASE("transpose_front/s/2", "[float][bias_batch_sum]") {
    const size_t M = 1029;
    const size_t N = 19;
    const size_t K = 27;

    float* x_cpu = new float[M * N * K];
    float* y_cpu = new float[M * N * K];
    float* y_ref = new float[M * N * K];

    for (size_t i = 0; i < M * N * K; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, M * N * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * K * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * K * sizeof(float), cudaMemcpyHostToDevice));

    egblas_stranspose_front(M, N, K, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * K * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                y_ref[n * (M * K) + m * K + k] = x_cpu[m * (N * K) + n * K + k];
            }
        }
    }

    for (size_t i = 0; i < M * N * K; ++i) {
        REQUIRE(y_cpu[i] == y_ref[i]);
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_ref;
}
