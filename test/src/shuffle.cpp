//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("shuffle/0", "[shuffle]") {
    const size_t N = 137;

    float* a_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_shuffle(N, a_gpu, sizeof(float));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            REQUIRE(a_cpu[i] != a_cpu[j]);
        }
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("shuffle/1", "[shuffle]") {
    const size_t N = 129;

    double* a_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = i;
    }

    double* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_shuffle(N, a_gpu, sizeof(double));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            REQUIRE(a_cpu[i] != a_cpu[j]);
        }
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("shuffle/2", "[shuffle]") {
    const size_t N = 129;

    double* a_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = i;
    }

    double* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_shuffle_seed(N, a_gpu, sizeof(double), 123);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            REQUIRE(a_cpu[i] != a_cpu[j]);
        }
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}
