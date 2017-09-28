//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("equal/s/0", "[float][equal]") {
    const size_t N = 137;

    float* a_cpu = new float[N];
    float* b_cpu = new float[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = -9.0 + i + 1;
        b_cpu[i] = 12.0 + (i + 1);
    }

    a_cpu[77] = 42.0;
    b_cpu[77] = 42.0;

    float* a_gpu;
    float* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sequal(N, a_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/s/1", "[float][equal]") {
    const size_t N = 333;

    float* a_cpu = new float[N];
    float* b_cpu = new float[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = -12.5f + i + 1;
        b_cpu[i] = 1.24f * (25.0f + (i + 1));
    }

    a_cpu[33] = -42.0;
    b_cpu[33] = -42.0;

    float* a_gpu;
    float* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sequal(N, a_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/d/0", "[double][equal]") {
    const size_t N = 137;

    double* a_cpu = new double[N];
    double* b_cpu = new double[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = -12.0 + i + 1;
        b_cpu[i] = 13.33 + i + 1 - 100.00;
    }

    a_cpu[33] = -42.42;
    b_cpu[33] = -42.42;

    double* a_gpu;
    double* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dequal(N, a_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/d/1", "[double][equal]") {
    const size_t N = 333;

    double* a_cpu = new double[N];
    double* b_cpu = new double[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = -12.0 + i + 1;
        b_cpu[i] = 13.33 + i + 1 - 100.00;
    }

    a_cpu[332] = -42.42;
    b_cpu[332] = -42.42;

    double* a_gpu;
    double* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dequal(N, a_gpu, 1, b_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/c/0", "[float][equal]") {
    const size_t N = 137;

    std::complex<float>* a_cpu = new std::complex<float>[N];
    std::complex<float>* b_cpu = new std::complex<float>[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = std::complex<float>(-2.0f + i / 1000.0f, 1.02f * i);
        b_cpu[i] = std::complex<float>(1.0f + i / 1000.0f, 1.03f * i);
    }

    a_cpu[67] = std::complex<float>(1.0f + 5.0 / 1000.0f, 1.02f * 5.0);
    b_cpu[67] = std::complex<float>(1.0f + 5.0 / 1000.0f, 1.03f * 5.0);

    std::complex<float>* a_gpu;
    std::complex<float>* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_cequal(N, reinterpret_cast<cuComplex*>(a_gpu), 1, reinterpret_cast<cuComplex*>(b_gpu), 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/c/1", "[float][equal]") {
    const size_t N = 338;

    std::complex<float>* a_cpu = new std::complex<float>[N];
    std::complex<float>* b_cpu = new std::complex<float>[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = std::complex<float>(-0.5 + i / 998.0f, -2.0f * i);
        b_cpu[i] = std::complex<float>(1.5 + i / 996.0f, -2.0f * i);
    }

    a_cpu[167] = std::complex<float>(2.0f + 5.0 / 1000.0f, 1.02f * 5.0);
    b_cpu[167] = std::complex<float>(2.0f + 5.0 / 1000.0f, 1.03f * 5.0);

    std::complex<float>* a_gpu;
    std::complex<float>* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_cequal(N, reinterpret_cast<cuComplex*>(a_gpu), 1, reinterpret_cast<cuComplex*>(b_gpu), 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/z/0", "[double][equal]") {
    const size_t N = 137;

    std::complex<double>* a_cpu = new std::complex<double>[N];
    std::complex<double>* b_cpu = new std::complex<double>[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = std::complex<double>((i-1) / 900.0, -1.1 * i);
        b_cpu[i] = std::complex<double>((i+1) / 999.0, -1.1 * i);
    }

    a_cpu[33] = std::complex<double>(2.0f + 5.0 / 1000.0f, 1.02f * 5.0);
    b_cpu[33] = std::complex<double>(2.0f + 5.0 / 1000.0f, 1.03f * 5.0);

    std::complex<double>* a_gpu;
    std::complex<double>* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zequal(N, reinterpret_cast<cuDoubleComplex*>(a_gpu), 1, reinterpret_cast<cuDoubleComplex*>(b_gpu), 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}

TEST_CASE("equal/z/1", "[double][equal]") {
    const size_t N = 338;

    std::complex<double>* a_cpu = new std::complex<double>[N];
    std::complex<double>* b_cpu = new std::complex<double>[N];
    bool* y_cpu = new bool[N];

    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = std::complex<double>(i / 996.0, 2.4 * i);
        b_cpu[i] = std::complex<double>(i / 996.0, 100.0);
    }

    a_cpu[66] = std::complex<double>(2.0f + 4.0 / 1000.0f, 1.02f * 5.0);
    b_cpu[66] = std::complex<double>(2.0f + 4.0 / 1000.0f, 1.03f * 5.0);

    std::complex<double>* a_gpu;
    std::complex<double>* b_gpu;
    bool* y_gpu;
    cuda_check(cudaMalloc((void**)&a_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(bool)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zequal(N, reinterpret_cast<cuDoubleComplex*>(a_gpu), 1, reinterpret_cast<cuDoubleComplex*>(b_gpu), 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(bool), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (a_cpu[i] == b_cpu[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(b_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] b_cpu;
    delete[] y_cpu;
}
