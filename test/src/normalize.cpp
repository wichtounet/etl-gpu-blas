//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("normalize_flat/0", "[normalize]") {
    const size_t N = 137;

    float* a_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_snormalize_flat(N, a_gpu, 1);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    float m = 0;
    for (size_t i = 0; i < N; ++i) {
        m += a_cpu[i];
    }

    m = m / N;

    float s = 0;
    for (size_t i = 0; i < N; ++i) {
        s += (a_cpu[i] - m) * (a_cpu[i] - m);
    }

    s = std::sqrt(s / N);

    REQUIRE(m == Approx(0.0f).scale(1));
    REQUIRE(s == Approx(1.0f).scale(1));

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("normalize_sub/0", "[normalize]") {
    const size_t N1 = 3;
    const size_t N = 137;

    float* a_cpu = new float[N1 * N];

    for (size_t i = 0; i < N1 * N; ++i) {
        a_cpu[i] = i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N1 * N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N1 * N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_snormalize_sub(N1, a_gpu, N, 1);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N1 * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t ss = 0; ss < N1; ++ss) {
        float m = 0;
        for (size_t i = 0; i < N; ++i) {
            m += a_cpu[ss * N + i];
        }

        m = m / N;

        float s = 0;
        for (size_t i = 0; i < N; ++i) {
            s += (a_cpu[ss * N + i] - m) * (a_cpu[ss * N + i] - m);
        }

        s = std::sqrt(s / N);

        REQUIRE(m == Approx(0.0f).scale(1));
        REQUIRE(s == Approx(1.0f).scale(1));
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}
