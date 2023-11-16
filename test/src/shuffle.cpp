//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include<algorithm>

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

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N, float(i)) == 1);
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

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N, double(i)) == 1);
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

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N, double(i)) == 1);
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("shuffle/3", "[shuffle]") {
    const size_t N = 129;
    const size_t S = 16;

    double* a_cpu = new double[N * S];

    for (size_t i = 0; i < N * S; ++i) {
        a_cpu[i] = i;
    }

    double* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S * sizeof(double), cudaMemcpyHostToDevice));

    egblas_shuffle_seed(N, a_gpu, S * sizeof(double), 123);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S * sizeof(double), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S; ++j){
            REQUIRE(a_cpu[(i * S) + j] == 1 + a_cpu[(i * S) + j - 1]);
        }
    }

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N * S; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N * S, double(i)) == 1);
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("shuffle/4", "[shuffle]") {
    const size_t N = 129;
    const size_t S = 23;

    double* a_cpu = new double[N * S];

    for (size_t i = 0; i < N * S; ++i) {
        a_cpu[i] = i;
    }

    double* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S * sizeof(double), cudaMemcpyHostToDevice));

    egblas_shuffle_seed(N, a_gpu, S * sizeof(double), 123);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S * sizeof(double), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S; ++j){
            REQUIRE(a_cpu[(i * S) + j] == 1 + a_cpu[(i * S) + j - 1]);
        }
    }

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N * S; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N * S, double(i)) == 1);
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("shuffle/5", "[shuffle]") {
    const size_t N = 129;
    const size_t S = 23;

    float* a_cpu = new float[N * S];

    for (size_t i = 0; i < N * S; ++i) {
        a_cpu[i] = i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S * sizeof(float), cudaMemcpyHostToDevice));

    egblas_shuffle_seed(N, a_gpu, S * sizeof(float), 123);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S * sizeof(float), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S; ++j){
            REQUIRE(a_cpu[(i * S) + j] == 1 + a_cpu[(i * S) + j - 1]);
        }
    }

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N * S; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N * S, double(i)) == 1);
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("shuffle/6", "[shuffle]") {
    const size_t N = 539;
    const size_t S = 29;

    float* a_cpu = new float[N * S];

    for (size_t i = 0; i < N * S; ++i) {
        a_cpu[i] = i;
    }

    float* a_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S * sizeof(float), cudaMemcpyHostToDevice));

    egblas_shuffle_seed(N, a_gpu, S * sizeof(float), 123);

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S * sizeof(float), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S; ++j){
            REQUIRE(a_cpu[(i * S) + j] == 1 + a_cpu[(i * S) + j - 1]);
        }
    }

    // Make sure we find each number in [0,N[
    for (size_t i = 0; i < N * S; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N * S, double(i)) == 1);
    }

    cuda_check(cudaFree(a_gpu));

    delete[] a_cpu;
}

TEST_CASE("par_shuffle/0", "[shuffle]") {
    const size_t N = 137;

    float* a_cpu = new float[N];
    float* b_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = float(i);
        b_cpu[i] = float(i) + 1.0f;
    }

    float* a_gpu;
    float* b_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_par_shuffle(N, a_gpu, sizeof(float), b_gpu, sizeof(float));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b_cpu, b_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(a_cpu[i] + 1.0f == b_cpu[i]);

        for (size_t j = i + 1; j < N; ++j) {
            REQUIRE(a_cpu[i] != a_cpu[j]);
            REQUIRE(b_cpu[i] != b_cpu[j]);
        }
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
}

TEST_CASE("par_shuffle/1", "[shuffle]") {
    const size_t N = 137;

    float* a_cpu = new float[N];
    double* b_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = float(i);
        b_cpu[i] = double(i) + 1.0;
    }

    float* a_gpu;
    double* b_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_par_shuffle(N, a_gpu, sizeof(float), b_gpu, sizeof(double));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b_cpu, b_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(double(a_cpu[i] + 1.0) == b_cpu[i]);

        for (size_t j = i + 1; j < N; ++j) {
            REQUIRE(a_cpu[i] != a_cpu[j]);
            REQUIRE(b_cpu[i] != b_cpu[j]);
        }
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
}

TEST_CASE("par_shuffle/2", "[shuffle]") {
    const size_t N = 137;

    float* a_cpu = new float[N];
    float* b_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = float(i);
        b_cpu[i] = float(i);
    }

    float* a_gpu;
    float* b_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_par_shuffle(N, a_gpu, sizeof(float), b_gpu, sizeof(float));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b_cpu, b_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(a_cpu[i] == b_cpu[i]);

        REQUIRE(std::count(a_cpu, a_cpu + N, float(i)) == 1);
        REQUIRE(std::count(b_cpu, b_cpu + N, float(i)) == 1);
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
}

TEST_CASE("par_shuffle/3", "[shuffle]") {
    const size_t N = 137;
    const size_t S = 17;

    float* a_cpu = new float[N * S];
    float* b_cpu = new float[N * S];

    for (size_t i = 0; i < N * S; ++i) {
        a_cpu[i] = float(i);
        b_cpu[i] = 1.0f + float(i);
    }

    float* a_gpu;
    float* b_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * S * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * S * sizeof(float), cudaMemcpyHostToDevice));

    egblas_par_shuffle(N, a_gpu, S * sizeof(float), b_gpu, S * sizeof(float));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b_cpu, b_gpu, N * S * sizeof(float), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S; ++j){
            REQUIRE(a_cpu[(i * S) + j] == 1 + a_cpu[(i * S) + j - 1]);
            REQUIRE(b_cpu[(i * S) + j] == 1 + b_cpu[(i * S) + j - 1]);
        }
    }

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(a_cpu[i] + 1.0f == b_cpu[i]);

        REQUIRE(std::count(a_cpu, a_cpu + N * S, float(i)) == 1);
        REQUIRE(std::count(b_cpu, b_cpu + N * S, float(i+1)) == 1);
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
}

TEST_CASE("par_shuffle/4", "[shuffle]") {
    const size_t N = 137;
    const size_t S1 = 19;
    const size_t S2 = 7;

    float* a_cpu = new float[N * S1];
    float* b_cpu = new float[N * S2];

    for (size_t i = 0; i < N * S1; ++i) {
        a_cpu[i] = float(i);
    }

    for (size_t i = 0; i < N * S2; ++i) {
        b_cpu[i] = 1.0f + float(i);
    }

    float* a_gpu;
    float* b_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S1 * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * S2 * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S1 * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * S2 * sizeof(float), cudaMemcpyHostToDevice));

    egblas_par_shuffle(N, a_gpu, S1 * sizeof(float), b_gpu, S2 * sizeof(float));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S1 * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b_cpu, b_gpu, N * S2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S1; ++j){
            REQUIRE(a_cpu[(i * S1) + j] == 1 + a_cpu[(i * S1) + j - 1]);
        }

        for(size_t j = 1; j < S2; ++j){
            REQUIRE(b_cpu[(i * S2) + j] == 1 + b_cpu[(i * S2) + j - 1]);
        }
    }

    for (size_t i = 0; i < N * S1; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N * S1, float(i)) == 1);
    }

    for (size_t i = 0; i < N * S2; ++i) {
        REQUIRE(std::count(b_cpu, b_cpu + N * S2, float(i+1)) == 1);
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
}

TEST_CASE("par_shuffle/5", "[shuffle]") {
    const size_t N = 666;
    const size_t S1 = 21;
    const size_t S2 = 9;

    float* a_cpu = new float[N * S1];
    float* b_cpu = new float[N * S2];

    for (size_t i = 0; i < N * S1; ++i) {
        a_cpu[i] = float(i);
    }

    for (size_t i = 0; i < N * S2; ++i) {
        b_cpu[i] = 1.0f + float(i);
    }

    float* a_gpu;
    float* b_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * S1 * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * S2 * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * S1 * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * S2 * sizeof(float), cudaMemcpyHostToDevice));

    egblas_par_shuffle(N, a_gpu, S1 * sizeof(float), b_gpu, S2 * sizeof(float));

    cuda_check(cudaMemcpy(a_cpu, a_gpu, N * S1 * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b_cpu, b_gpu, N * S2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Make sure the sub order is correct
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 1; j < S1; ++j){
            REQUIRE(a_cpu[(i * S1) + j] == 1 + a_cpu[(i * S1) + j - 1]);
        }

        for(size_t j = 1; j < S2; ++j){
            REQUIRE(b_cpu[(i * S2) + j] == 1 + b_cpu[(i * S2) + j - 1]);
        }
    }

    for (size_t i = 0; i < N * S1; ++i) {
        REQUIRE(std::count(a_cpu, a_cpu + N * S1, float(i)) == 1);
    }

    for (size_t i = 0; i < N * S2; ++i) {
        REQUIRE(std::count(b_cpu, b_cpu + N * S2, float(i+1)) == 1);
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
}
