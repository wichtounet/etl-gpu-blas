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

TEST_CASE("sum/s/0", "[float][sum]") {
    const size_t N = 257;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i + 1;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_ssum(gpu_vec, N, 1) == Approx(float((N * (N + 1)) / 2)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/s/1", "[float][sum]") {
    const size_t N = 389;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_ssum(gpu_vec, N, 1) == Approx(float(116972)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/s/2", "[float][sum]") {
    const size_t N = 1024 * 128 + 5;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i + 1;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_ssum(gpu_vec, N, 1) == Approx(float((N * (N + 1)) / 2)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/s/3", "[float][sum]") {
    const size_t N = 1024 * 1024 * 6 + 126;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i + 1;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    REQUIRE(egblas_ssum(gpu_vec, N, 1) == Approx(float((N * (N + 1)) / 2)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/d/0", "[double][sum]") {
    const size_t N = 123;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    REQUIRE(egblas_dsum(gpu_vec, N, 1) == Approx(double(7503)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/d/1", "[double][sum]") {
    const size_t N = 389;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    REQUIRE(egblas_dsum(gpu_vec, N, 1) == Approx(double(116972)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/c/0", "[float][sum]") {
    const size_t N = 33;

    std::complex<float>* cpu_vec = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<float>(i, 2 * i);
    }

    std::complex<float>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    auto sum   = egblas_csum(reinterpret_cast<cuComplex*>(gpu_vec), N, 1);
    auto sum_c = *reinterpret_cast<std::complex<float>*>(&sum);
    REQUIRE(sum_c.real() == Approx(float(528)));
    REQUIRE(sum_c.imag() == Approx(float(2 * 528)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/c/1", "[float][sum]") {
    const size_t N = 111;

    std::complex<float>* cpu_vec = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<float>(-1.0 * i, 0.1 * i);
    }

    std::complex<float>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    auto sum   = egblas_csum(reinterpret_cast<cuComplex*>(gpu_vec), N, 1);
    auto sum_c = *reinterpret_cast<std::complex<float>*>(&sum);
    REQUIRE(sum_c.real() == Approx(float(-6105.0)));
    REQUIRE(sum_c.imag() == Approx(float(610.5)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/z/0", "[double][sum]") {
    const size_t N = 32;

    std::complex<double>* cpu_vec = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<double>(i, 2 * i);
    }

    std::complex<double>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    auto sum   = egblas_zsum(reinterpret_cast<cuDoubleComplex*>(gpu_vec), N, 1);
    auto sum_c = *reinterpret_cast<std::complex<double>*>(&sum);
    REQUIRE(sum_c.real() == Approx(double(496)));
    REQUIRE(sum_c.imag() == Approx(double(2 * 496)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("sum/z/1", "[double][sum]") {
    const size_t N = 123;

    std::complex<double>* cpu_vec = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<double>(-1.0 * i, 0.1 * i);
    }

    std::complex<double>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    auto sum   = egblas_zsum(reinterpret_cast<cuDoubleComplex*>(gpu_vec), N, 1);
    auto sum_c = *reinterpret_cast<std::complex<double>*>(&sum);
    REQUIRE(sum_c.real() == Approx(double(-7503.0)));
    REQUIRE(sum_c.imag() == Approx(double(750.3)));

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}
