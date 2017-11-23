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

TEST_CASE("axdy_3/s/0", "[float][axdy_3]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];
    float* yy_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.1f * (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    float* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy_3(N, 1.0, x_gpu, 1, y_gpu, 1, yy_gpu, 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == Approx(1.0f * (i + 1) / (2.1f * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/s/1", "[float][axdy_3]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];
    float* yy_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    float* x_gpu;
    float* y_gpu;
    float* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy_3(N, 0.2, x_gpu, 1, y_gpu, 1, yy_gpu, 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == Approx((i + 1) / (0.2 * 2.3 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/s/2", "[float][axdy_3]") {
    const size_t N = 120;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];
    float* yy_cpu = new float[N / 3];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    float* x_gpu;
    float* y_gpu;
    float* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&yy_gpu, (N / 3)* sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy_3(N, 0.2, x_gpu, 3, y_gpu, 3, yy_gpu, 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, (N / 3) * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy_cpu[i] == Approx(((3 * i + 1) / (0.2f * 2.3f * (3 * i + 1)))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/d/0", "[double][axdy_3]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];
    double* yy_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.1f * (i+1);
    }

    double* x_gpu;
    double* y_gpu;
    double* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy_3(N, 1.0, x_gpu, 1, y_gpu, 1, yy_gpu, 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == Approx((1.0f * (i+1)) / (2.1f * (i+1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/d/1", "[double][axdy_3]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];
    double* yy_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    double* x_gpu;
    double* y_gpu;
    double* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy_3(N, 0.2, x_gpu, 1, y_gpu, 1, yy_gpu, 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == Approx((i+1) / (0.2 * 2.3 * (i+1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/d/2", "[double][axdy_3]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];
    double* yy_cpu = new double[N / 3];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.3f * (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    double* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&yy_gpu, (N / 3) * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy_3(N, 0.2, x_gpu, 3, y_gpu, 3, yy_gpu, 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, (N / 3) * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy_cpu[i] == Approx((3 * i + 1) / (0.2f * 2.3f * (3 * i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/c/0", "[float][axdy_3]") {
    const size_t N = 27;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];
    std::complex<float>* yy_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i, i);
        y_cpu[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    std::complex<float>* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_caxdy_3(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1, reinterpret_cast<cuComplex*>(yy_gpu), 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == TestComplex<double>(x_cpu[i] / y_cpu[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/c/1", "[float][axdy_3]") {
    const size_t N = 33;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];
    std::complex<float>* yy_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i, -1.0 * i);
        y_cpu[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    std::complex<float>* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_caxdy_3(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1, reinterpret_cast<cuComplex*>(yy_gpu), 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == TestComplex<double>(x_cpu[i] / y_cpu[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/z/0", "[double][axdy_3]") {
    const size_t N = 27;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];
    std::complex<double>* yy_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i, i);
        y_cpu[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    std::complex<double>* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zaxdy_3(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1, reinterpret_cast<cuDoubleComplex*>(yy_gpu), 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == TestComplex<double>(x_cpu[i] / y_cpu[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}

TEST_CASE("axdy_3/z/1", "[double][axdy_3]") {
    const size_t N = 33;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];
    std::complex<double>* yy_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i, -1.0 * i);
        y_cpu[i] = std::complex<double>(-1.0f * i, 2.1f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    std::complex<double>* yy_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&yy_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zaxdy_3(N, make_cuDoubleComplex(1.0, 0.1), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1, reinterpret_cast<cuDoubleComplex*>(yy_gpu), 1);

    cuda_check(cudaMemcpy(yy_cpu, yy_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy_cpu[i] == TestComplex<double>(x_cpu[i] / (std::complex<double>(1.0, 0.1) * y_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(yy_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] yy_cpu;
}
