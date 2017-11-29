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

TEST_CASE("axdbpy/s/0", "[float][axdbpy]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.1f * (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdbpy(N, 1.01f, x_gpu, 1, 2.0f, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx((1.01f * (i + 1)) / (2.0f + 2.1f * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdbpy/d/0", "[double][axdbpy]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.1f * (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdbpy(N, 1.4, x_gpu, 1, 1.5, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx((1.4 * (i + 1)) / (1.5 + 2.1 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdbpy/c/0", "[float][axdbpy]") {
    const size_t N = 99;

    std::complex<float>* x_cpu   = new std::complex<float>[ N ];
    std::complex<float>* y_cpu   = new std::complex<float>[ N ];
    std::complex<float>* y_cpu_b = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i]   = std::complex<float>(0.1 + 0.1 * i, 0.1 + 0.001f * i);
        y_cpu[i]   = std::complex<float>(1.1f * i, 0.01f * i);
        y_cpu_b[i] = std::complex<float>(1.1f * i, 0.01f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    std::complex<float> alpha(1.0f, 2.0f);
    std::complex<float> beta(1.01f, 2.01f);
    egblas_caxdbpy(N, make_cuComplex(1.0f, 2.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, make_cuComplex(1.01f, 2.01f), reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(((alpha * x_cpu[i]) / (beta + y_cpu_b[i]))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_cpu_b;
}

TEST_CASE("axdbpy/z/0", "[double][axdbpy]") {
    const size_t N = 99;

    std::complex<double>* x_cpu   = new std::complex<double>[ N ];
    std::complex<double>* y_cpu   = new std::complex<double>[ N ];
    std::complex<double>* y_cpu_b = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i]   = std::complex<double>(0.1 + 0.1 * i, 0.1 + 0.001f * i);
        y_cpu[i]   = std::complex<double>(1.1f * i, 0.01f * i);
        y_cpu_b[i] = std::complex<double>(1.1f * i, 0.01f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    std::complex<double> alpha(1.0f, 2.0f);
    std::complex<double> beta(1.01f, 2.01f);
    egblas_zaxdbpy(N, make_cuDoubleComplex(1.0f, 2.0f), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, make_cuDoubleComplex(1.01f, 2.01f), reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(((alpha * x_cpu[i]) / (beta + y_cpu_b[i]))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_cpu_b;
}
