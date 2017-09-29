//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("or/b/0", "[bool][or]") {
    const size_t N = 137;

    bool* a_cpu = new bool[N];
    bool* b_cpu = new bool[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = i % 5 == 0;
        b_cpu[i] = i % 2 == 0;
    }

    bool* a_gpu;
    bool* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(bool)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(bool)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(bool), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(bool), cudaMemcpyHostToDevice));

    egblas_bor(N, a_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] || b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}
