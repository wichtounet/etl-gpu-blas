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

#include "catch.hpp"

TEST_CASE("scalar_mul/s/0", "[float][scalar_mul]") {
    const size_t N = 123;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_scalar_smul(gpu_vec, N, 1, 1.5f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i] == Approx(i * 1.5f));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/s/1", "[float][scalar_mul]") {
    const size_t N = 124;

    float* cpu_vec = new float[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    float* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(float)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_scalar_smul(gpu_vec, N, 2, 1.5f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            REQUIRE(cpu_vec[i] == Approx(i * 1.5f));
        } else {
            REQUIRE(cpu_vec[i] == Approx(i));
        }
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/d/0", "[double][scalar_mul]") {
    const size_t N = 254;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_scalar_dmul(gpu_vec, N, 1, 2.2f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i] == Approx(i * 2.2));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/d/1", "[double][scalar_mul]") {
    const size_t N = 302;

    double* cpu_vec = new double[N];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(double)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_scalar_dmul(gpu_vec, N, 3, 2.2f);

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(cpu_vec[i] == Approx(i * 2.2));
        } else {
            REQUIRE(cpu_vec[i] == Approx(i));
        }
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/c/0", "[float][scalar_mul]") {
    const size_t N = 123;

    std::complex<float>* cpu_vec = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<float>(i, i);
    }

    std::complex<float>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_scalar_cmul(reinterpret_cast<cuComplex*>(gpu_vec), N, 1, make_cuComplex(1.1, 1.2));

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i].real() == Approx((std::complex<float>(i,i) * std::complex<float>(1.1, 1.2)).real()));
        REQUIRE(cpu_vec[i].imag() == Approx((std::complex<float>(i,i) * std::complex<float>(1.1, 1.2)).imag()));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/c/1", "[float][scalar_mul]") {
    const size_t N = 321;

    std::complex<float>* cpu_vec = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<float>(2 * i, -3 * i);
    }

    std::complex<float>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_scalar_cmul(reinterpret_cast<cuComplex*>(gpu_vec), N, 1, make_cuComplex(1.3, -1.2));

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i].real() == Approx((std::complex<float>(2 * i,-3 * i) * std::complex<float>(1.3, -1.2)).real()));
        REQUIRE(cpu_vec[i].imag() == Approx((std::complex<float>(2 * i,-3 * i) * std::complex<float>(1.3, -1.2)).imag()));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/z/0", "[double][scalar_mul]") {
    const size_t N = 128;

    std::complex<double>* cpu_vec = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<double>(i, i);
    }

    std::complex<double>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_scalar_zmul(reinterpret_cast<cuDoubleComplex*>(gpu_vec), N, 1, make_cuDoubleComplex(1.1, 1.2));

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i].real() == Approx((std::complex<double>(i,i) * std::complex<double>(1.1, 1.2)).real()));
        REQUIRE(cpu_vec[i].imag() == Approx((std::complex<double>(i,i) * std::complex<double>(1.1, 1.2)).imag()));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}

TEST_CASE("scalar_mul/z/1", "[double][scalar_mul]") {
    const size_t N = 1025;

    std::complex<double>* cpu_vec = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        cpu_vec[i] = std::complex<double>(2 * i, -3 * i);
    }

    std::complex<double>* gpu_vec;
    cuda_check(cudaMalloc((void**)&gpu_vec, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(gpu_vec, cpu_vec, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_scalar_zmul(reinterpret_cast<cuDoubleComplex*>(gpu_vec), N, 1, make_cuDoubleComplex(1.3, -1.2));

    cuda_check(cudaMemcpy(cpu_vec, gpu_vec, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(cpu_vec[i].real() == Approx((std::complex<double>(2 * i,-3 * i) * std::complex<double>(1.3, -1.2)).real()));
        REQUIRE(cpu_vec[i].imag() == Approx((std::complex<double>(2 * i,-3 * i) * std::complex<double>(1.3, -1.2)).imag()));
    }

    cuda_check(cudaFree(gpu_vec));

    delete[] cpu_vec;
}
