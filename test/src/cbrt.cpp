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

template<typename T>
std::complex<T> cbrt(std::complex<T> z){
    T z_abs = std::abs(z);
    T z_arg = std::arg(z);

    T new_abs = std::cbrt(z_abs);
    T new_arg = z_arg / T(3);

    return {new_abs * std::cos(new_arg), new_abs * std::sin(new_arg)};
}

TEST_CASE("cbrt/s/0", "[float][cbrt]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_scbrt(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * std::cbrt(1.0f * i)));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/s/1", "[float][cbrt]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 1.1f * (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_scbrt(N, 0.2f, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * std::cbrt(1.1f * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/d/0", "[double][cbrt]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dcbrt(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0 * std::cbrt(1.0 * i)));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/d/1", "[double][cbrt]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 1.1 * (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dcbrt(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2 * std::cbrt(1.1 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/c/0", "[float][cbrt]") {
    const size_t N = 137;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu = new std::complex<float>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(1.0f + i / 1000.0f, 0.1 + 0.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_ccbrt(N, make_cuComplex(1.0f, 0.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(cbrt(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/c/1", "[float][cbrt]") {
    const size_t N = 338;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu = new std::complex<float>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(0.1 + i / 998.0f, 0.1 + -0.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_ccbrt(N, make_cuComplex(1.0f, 1.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(std::complex<float>(1.0f, 1.0f) * cbrt(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/z/0", "[double][cbrt]") {
    const size_t N = 137;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu = new std::complex<double>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i / 999.0, -0.1 * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zcbrt(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(cbrt(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("cbrt/z/1", "[double][cbrt]") {
    const size_t N = 338;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu = new std::complex<double>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i / 996.0, -0.4 * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zcbrt(N, make_cuDoubleComplex(0.1, 2.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(std::complex<double>(0.1, 2.0) * cbrt(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
