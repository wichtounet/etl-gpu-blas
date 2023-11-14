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

TEST_CASE("batch_k_minus_scale2/s/0", "[float][batch_k_minus_scale2]") {
    const size_t B = 2;
    const size_t K = 3;

    float* x_cpu = new float[B * K];
    float* y_cpu = new float[B * K];
    float* gamma_cpu = new float[K];
    float* beta_cpu = new float[K];

    for (size_t i = 0; i < B * K; ++i) {
        x_cpu[i] = i + 1;
    }

    for (size_t i = 0; i < K; ++i) {
        gamma_cpu[i] = i + 1;
        beta_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;
    float* gamma_gpu;
    float* beta_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&gamma_gpu, K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&beta_gpu, K * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * K * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gamma_gpu, gamma_cpu, K * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(beta_gpu, beta_cpu, K * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbatch_k_minus_scale2(B, K, x_gpu, gamma_gpu, beta_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * K * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 1.0f);
    REQUIRE(y_cpu[1] == 2.0f);
    REQUIRE(y_cpu[2] == 3.0f);
    REQUIRE(y_cpu[3] == 4.0f);
    REQUIRE(y_cpu[4] == 8.0f);
    REQUIRE(y_cpu[5] == 12.0f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(gamma_gpu));
    cuda_check(cudaFree(beta_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] gamma_cpu;
    delete[] beta_cpu;
}

TEST_CASE("batch_k_minus_scale2/s/1", "[float][batch_k_minus_scale2]") {
    const size_t B = 3;
    const size_t K = 2;

    float* x_cpu = new float[B * K];
    float* y_cpu = new float[B * K];
    float* gamma_cpu = new float[K];
    float* beta_cpu = new float[K];

    for (size_t i = 0; i < B * K; ++i) {
        x_cpu[i] = i + 1;
    }

    for (size_t i = 0; i < K; ++i) {
        gamma_cpu[i] = i + 1;
        beta_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;
    float* gamma_gpu;
    float* beta_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&gamma_gpu, K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&beta_gpu, K * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * K * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gamma_gpu, gamma_cpu, K * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(beta_gpu, beta_cpu, K * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbatch_k_minus_scale2(B, K, x_gpu, gamma_gpu, beta_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * K * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 1.0f);
    REQUIRE(y_cpu[1] == 2.0f);
    REQUIRE(y_cpu[2] == 3.0f);
    REQUIRE(y_cpu[3] == 6.0f);
    REQUIRE(y_cpu[4] == 5.0f);
    REQUIRE(y_cpu[5] == 10.0f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(gamma_gpu));
    cuda_check(cudaFree(beta_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] gamma_cpu;
    delete[] beta_cpu;
}

TEST_CASE("batch_k_minus_scale2/d/0", "[double][batch_k_minus_scale2]") {
    const size_t B = 2;
    const size_t K = 3;

    double* x_cpu = new double[B * K];
    double* y_cpu = new double[B * K];
    double* gamma_cpu = new double[K];
    double* beta_cpu = new double[K];

    for (size_t i = 0; i < B * K; ++i) {
        x_cpu[i] = i + 1;
    }

    for (size_t i = 0; i < K; ++i) {
        gamma_cpu[i] = i + 1;
        beta_cpu[i] = i;
    }

    double* x_gpu;
    double* y_gpu;
    double* gamma_gpu;
    double* beta_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&gamma_gpu, K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&beta_gpu, K * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * K * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gamma_gpu, gamma_cpu, K * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(beta_gpu, beta_cpu, K * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbatch_k_minus_scale2(B, K, x_gpu, gamma_gpu, beta_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * K * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 1.0);
    REQUIRE(y_cpu[1] == 2.0);
    REQUIRE(y_cpu[2] == 3.0);
    REQUIRE(y_cpu[3] == 4.0);
    REQUIRE(y_cpu[4] == 8.0);
    REQUIRE(y_cpu[5] == 12.0);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(gamma_gpu));
    cuda_check(cudaFree(beta_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] gamma_cpu;
    delete[] beta_cpu;
}

TEST_CASE("batch_k_minus_scale2/d/1", "[double][batch_k_minus_scale2]") {
    const size_t B = 2;
    const size_t K = 4;

    double* x_cpu = new double[B * K];
    double* y_cpu = new double[B * K];
    double* gamma_cpu = new double[K];
    double* beta_cpu = new double[K];

    for (size_t i = 0; i < B * K; ++i) {
        x_cpu[i] = i + 1;
    }

    for (size_t i = 0; i < K; ++i) {
        gamma_cpu[i] = i + 1;
        beta_cpu[i] = i;
    }

    double* x_gpu;
    double* y_gpu;
    double* gamma_gpu;
    double* beta_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&gamma_gpu, K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&beta_gpu, K * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * K * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gamma_gpu, gamma_cpu, K * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(beta_gpu, beta_cpu, K * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbatch_k_minus_scale2(B, K, x_gpu, gamma_gpu, beta_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * K * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 1.0);
    REQUIRE(y_cpu[1] == 2.0);
    REQUIRE(y_cpu[2] == 3.0);
    REQUIRE(y_cpu[3] == 4.0);
    REQUIRE(y_cpu[4] == 5.0);
    REQUIRE(y_cpu[5] == 10.0);
    REQUIRE(y_cpu[6] == 15.0);
    REQUIRE(y_cpu[7] == 20.0);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(gamma_gpu));
    cuda_check(cudaFree(beta_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] gamma_cpu;
    delete[] beta_cpu;
}

TEST_CASE("batch_k_minus_scale4/s/0", "[float][batch_k_minus_scale2]") {
    const size_t B = 5;
    const size_t K = 3;
    const size_t M = 4;
    const size_t N = 4;

    float* x_cpu = new float[B * K * M * N];
    float* y_cpu = new float[B * K * M * N];
    float* gamma_cpu = new float[K];
    float* beta_cpu = new float[K];

    for (size_t i = 0; i < B * K * M * N; ++i) {
        x_cpu[i] = i + 1;
    }

    for (size_t i = 0; i < K; ++i) {
        gamma_cpu[i] = i + 1;
        beta_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;
    float* gamma_gpu;
    float* beta_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * K * M * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * K * M * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&gamma_gpu, K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&beta_gpu, K * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * K * M * N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gamma_gpu, gamma_cpu, K * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(beta_gpu, beta_cpu, K * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbatch_k_minus_scale4(B, K, M, N, x_gpu, gamma_gpu, beta_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * K * M * N * sizeof(float), cudaMemcpyDeviceToHost));

    size_t bkmn = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    REQUIRE(y_cpu[bkmn] == gamma_cpu[k] * (x_cpu[bkmn] - beta_cpu[k]));
                    ++bkmn;
                }
            }
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(gamma_gpu));
    cuda_check(cudaFree(beta_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] gamma_cpu;
    delete[] beta_cpu;
}

TEST_CASE("batch_k_minus_scale4/d/0", "[float][batch_k_minus_scale2]") {
    const size_t B = 2;
    const size_t K = 3;
    const size_t M = 4;
    const size_t N = 5;

    double* x_cpu = new double[B * K * M * N];
    double* y_cpu = new double[B * K * M * N];
    double* gamma_cpu = new double[K];
    double* beta_cpu = new double[K];

    for (size_t i = 0; i < B * K * M * N; ++i) {
        x_cpu[i] = i + 1;
    }

    for (size_t i = 0; i < K; ++i) {
        gamma_cpu[i] = i + 1;
        beta_cpu[i] = i;
    }

    double* x_gpu;
    double* y_gpu;
    double* gamma_gpu;
    double* beta_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * K * M * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, B * K * M * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&gamma_gpu, K * sizeof(double)));
    cuda_check(cudaMalloc((void**)&beta_gpu, K * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * K * M * N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gamma_gpu, gamma_cpu, K * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(beta_gpu, beta_cpu, K * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbatch_k_minus_scale4(B, K, M, N, x_gpu, gamma_gpu, beta_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, B * K * M * N * sizeof(double), cudaMemcpyDeviceToHost));

    size_t bkmn = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    REQUIRE(y_cpu[bkmn] == gamma_cpu[k] * (x_cpu[bkmn] - beta_cpu[k]));
                    ++bkmn;
                }
            }
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(gamma_gpu));
    cuda_check(cudaFree(beta_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] gamma_cpu;
    delete[] beta_cpu;
}
