//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("bias_add_2d/s/0", "[float][abs]") {
    const size_t M = 137;
    const size_t N = 19;

    float* x_cpu = new float[M * N];
    float* y_cpu = new float[M * N];
    float* b_cpu = new float[N];

    for (size_t i = 0; i < M * N; ++i) {
        x_cpu[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        b_cpu[i] = i;;
    }

    float* x_gpu;
    float* y_gpu;
    float* b_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, M * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_add_2d(M, N, x_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < M * N; ++i) {
        REQUIRE(y_cpu[i] == Approx(i + i % N));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_add_2d/d/0", "[double][abs]") {
    const size_t M = 137;
    const size_t N = 19;

    double* x_cpu = new double[M * N];
    double* y_cpu = new double[M * N];
    double* b_cpu = new double[N];

    for (size_t i = 0; i < M * N; ++i) {
        x_cpu[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        b_cpu[i] = i + 1;
    }

    double* x_gpu;
    double* y_gpu;
    double* b_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, M * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_add_2d(M, N, x_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < M * N; ++i) {
        REQUIRE(y_cpu[i] == Approx(x_cpu[i] + b_cpu[i % N]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_add_4d/s/0", "[float][abs]") {
    const size_t M = 13;
    const size_t N = 9;
    const size_t O = 11;
    const size_t P = 8;

    double* x_cpu = new double[M * N * O * P];
    double* y_cpu = new double[M * N * O * P];
    double* b_cpu = new double[N];

    for (size_t i = 0; i < M * N; ++i) {
        x_cpu[i] = 1.09 * i;
    }

    for (size_t i = 0; i < N; ++i) {
        b_cpu[i] = 2.04f * (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    double* b_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, M * N * O * P * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * O * P * sizeof(double)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * O * P * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_add_4d(M, N, O, P, x_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * O * P * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < M * N * O * P; ++i) {
        REQUIRE(y_cpu[i] == Approx(x_cpu[i] + b_cpu[(i / (O * P)) % N]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_add_4d/d/0", "[float][abs]") {
    const size_t M = 3;
    const size_t N = 9;
    const size_t O = 7;
    const size_t P = 5;

    double* x_cpu = new double[M * N * O * P];
    double* y_cpu = new double[M * N * O * P];
    double* b_cpu = new double[N];

    for (size_t i = 0; i < M * N; ++i) {
        x_cpu[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        b_cpu[i] = i + 1;
    }

    double* x_gpu;
    double* y_gpu;
    double* b_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, M * N * O * P * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, M * N * O * P * sizeof(double)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, M * N * O * P * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_add_4d(M, N, O, P, x_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * O * P * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < M * N * O * P; ++i) {
        REQUIRE(y_cpu[i] == Approx(x_cpu[i] + b_cpu[(i / (O * P)) % N]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
