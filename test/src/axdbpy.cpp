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

#ifdef TEST_HALF

TEST_CASE_HALF("axdbpy/h/0") {
    const size_t N = 137;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = fromFloat<T>(i + 0.1f);
        y_cpu[i] = fromFloat<T>(2.1f * i + 0.1f);
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxdbpy(N, fromFloat<T>(1.0), x_gpu, 1, fromFloat<T>(1.0), y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxdbpy(N, fromFloat<T>(1.0), x_gpu, 1, fromFloat<T>(1.0), y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx((i + 0.1f) / (1.0f + 2.1f * i + 0.1f)).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx((i + 0.1f) / (1.0f + 2.1f * i + 0.1f)).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_HALF("axdbpy/h/1") {
    const size_t N = 79;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = fromFloat<T>(i);
        y_cpu[i] = fromFloat<T>(3.1f * i);
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxdbpy(N, fromFloat<T>(0.5), x_gpu, 1, fromFloat<T>(1.0), y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxdbpy(N, fromFloat<T>(0.5), x_gpu, 1, fromFloat<T>(1.0), y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx((0.5f * i) / (1.0f + 3.1f * i)).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx((0.5f * i) / (1.0f + 3.1f * i)).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_HALF("axdbpy/h/2") {
    const size_t N = 192;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = fromFloat<T>(i);
        y_cpu[i] = fromFloat<T>(-2.1f * i);
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxdbpy(N, fromFloat<T>(0.0), x_gpu, 1, fromFloat<T>(0.1), y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxdbpy(N, fromFloat<T>(0.0), x_gpu, 1, fromFloat<T>(0.1), y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

 //Compute y = (alpha * x) / (beta + y) (element wise), in single-precision

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(0.0f).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(0.0f).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

#endif
