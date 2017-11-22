//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("dropout/s/0", "[shuffle]") {
    const size_t N = 137;

    float* a_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = 28 * i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sdropout(N, 0.2f, 1.0f, a_gpu, 1);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE((a_cpu[i] == 1.0f || a_cpu[i] == 0.0f));
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("dropout/s/1", "[shuffle]") {
    const size_t N = 157;

    float* a_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = 28 * i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sdropout(N, 0.5f, 0.7f, a_gpu, 1);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE((a_cpu[i] == 0.7f || a_cpu[i] == 0.0f));
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("dropout/d/0", "[shuffle]") {
    const size_t N = 137;

    double* a_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = 28 * i;
    }

    double* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_ddropout(N, 0.2, 1.0, a_gpu, 1);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE((a_cpu[i] == 1.0 || a_cpu[i] == 0.0));
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("dropout/d/1", "[shuffle]") {
    const size_t N = 157;

    double* a_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = 28 * i;
    }

    double* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_ddropout(N, 0.5, 0.7, a_gpu, 1);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE((a_cpu[i] == 0.7 || a_cpu[i] == 0.0));
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}
