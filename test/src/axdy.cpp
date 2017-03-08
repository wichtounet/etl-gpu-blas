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

TEST_CASE("axdy/s/0", "[float][axdy]") {
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

    egblas_saxdy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx((2.1f * (i + 1)) / (1.0 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdy/s/1", "[float][axdy]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.3f * (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdy/s/2", "[float][axdy]") {
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.3f * (i + 1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * (i + 1)));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdy/d/0", "[double][axdy]") {
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

    egblas_daxdy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx((2.1f * (i + 1)) / (1.0 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdy/d/1", "[double][axdy]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.3f * (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdy/d/2", "[double][axdy]") {
    const size_t N = 111;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i + 1);
        y_cpu[i] = 2.3f * (i + 1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * (i + 1)));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axdy/c/0", "[float][axdy]") {
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
    egblas_caxdy(N, make_cuComplex(1.0f, 2.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).real()));
        REQUIRE(y_cpu[i].imag() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).imag()));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_cpu_b;
}

TEST_CASE("axdy/c/1", "[float][axdy]") {
    const size_t N = 33;

    std::complex<float>* x_cpu   = new std::complex<float>[ N ];
    std::complex<float>* y_cpu   = new std::complex<float>[ N ];
    std::complex<float>* y_cpu_b = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i]   = std::complex<float>(-0.01 + 0.1 * i, 0.2 + 0.004f * i);
        y_cpu[i]   = std::complex<float>(1.3f * i, 0.29f * i);
        y_cpu_b[i] = std::complex<float>(1.3f * i, 0.29f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    std::complex<float> alpha(1.5f, 2.5f);
    egblas_caxdy(N, make_cuComplex(1.5f, 2.5f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).real()));
        REQUIRE(y_cpu[i].imag() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).imag()));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_cpu_b;
}

TEST_CASE("axdy/z/0", "[double][axdy]") {
    const size_t N = 48;

    std::complex<double>* x_cpu   = new std::complex<double>[ N ];
    std::complex<double>* y_cpu   = new std::complex<double>[ N ];
    std::complex<double>* y_cpu_b = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i]   = std::complex<double>(0.2 + 0.2 * i, 0.1 + 0.001 * i);
        y_cpu[i]   = std::complex<double>(1.1 * i, 0.01 * i);
        y_cpu_b[i] = std::complex<double>(1.1 * i, 0.01 * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    std::complex<double> alpha(1.0, 2.0);
    egblas_zaxdy(N, make_cuDoubleComplex(1.0, 2.0),
                 reinterpret_cast<cuDoubleComplex*>(x_gpu), 1,
                 reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).real()));
        REQUIRE(y_cpu[i].imag() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).imag()));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_cpu_b;
}

TEST_CASE("axdy/z/1", "[double][axdy]") {
    const size_t N = 39;

    std::complex<double>* x_cpu   = new std::complex<double>[ N ];
    std::complex<double>* y_cpu   = new std::complex<double>[ N ];
    std::complex<double>* y_cpu_b = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i]   = std::complex<double>(-0.01 + 0.1 * i, 0.2 + 0.004 * i);
        y_cpu[i]   = std::complex<double>(1.5 * i, 0.59 * i);
        y_cpu_b[i] = std::complex<double>(1.5 * i, 0.59 * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    std::complex<double> alpha(1.5, 2.5);
    egblas_zaxdy(N, make_cuDoubleComplex(1.5, 2.5),
                 reinterpret_cast<cuDoubleComplex*>(x_gpu), 1,
                 reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).real()));
        REQUIRE(y_cpu[i].imag() == Approx((y_cpu_b[i] / (alpha * x_cpu[i])).imag()));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_cpu_b;
}
