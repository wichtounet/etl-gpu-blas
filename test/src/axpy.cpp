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
#include "cuda_fp16.h"

#include "egblas.hpp"
#include "test.hpp"

TEST_CASE("axpy/s/0", "[float][axpy]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.1f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * i + 2.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/s/1", "[float][axpy]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/s/2", "[float][axpy]") {
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3f * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/s/3", "[float][axpy]") {
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 0.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.0f * i + 2.3f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/d/0", "[double][axpy]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.1f * i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxpy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * i + 2.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/d/1", "[double][axpy]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxpy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/d/2", "[double][axpy]") {
    const size_t N = 111;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxpy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3f * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/c/0", "[float][axpy]") {
    const size_t N = 27;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i, i);
        y_cpu[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_caxpy(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -2.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(i +  0.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/c/1", "[float][axpy]") {
    const size_t N = 33;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i, -1.0 * i);
        y_cpu[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_caxpy(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -1.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(i +  0.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/z/0", "[double][axpy]") {
    const size_t N = 27;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i, i);
        y_cpu[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zaxpy(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -2.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(i +  0.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/z/1", "[double][axpy]") {
    const size_t N = 33;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i, -1.0 * i);
        y_cpu[i] = std::complex<double>(-1.0f * i, 2.1f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zaxpy(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -1.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(-1.0 * i +  2.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_TEMPLATE("axpy/i/0", T, int32_t, int64_t) {
    const size_t N = 137;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 21 * i;
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 1, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 1, x_gpu, 1, y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1 * i + 21 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_TEMPLATE("axpy/i/1", T, int32_t, int64_t) {
    const size_t N = 333;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 23 * i;
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 2, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 2, x_gpu, 1, y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(2 * i + 23 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_TEMPLATE("axpy/i/2", T, int32_t, int64_t ) {
    const size_t N = 111;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 23 * i;
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 2, x_gpu, 3, y_gpu, 3);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 2, x_gpu, 3, y_gpu, 3);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(2 * i + 23 * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(23 * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

#ifdef TEST_HALF

TEST_CASE_HALF("axpy/h/0") {
    const size_t N = 137;

    T* x_cpu = new T[N];
    T* y_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = fromFloat<T>(i);
        y_cpu[i] = fromFloat<T>(2.1f * i);
    }

    T* x_gpu;
    T* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(T)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxpy(N, fromFloat<T>(1.0), x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpy(N, fromFloat<T>(1.0), x_gpu, 1, y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(1.0f * i + 2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(1.0f * i + 2.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_HALF("axpy/h/1") {
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
        egblas_haxpy(N, fromFloat<T>(0.5), x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpy(N, fromFloat<T>(0.5), x_gpu, 1, y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(0.5f * i + 3.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(0.5f * i + 3.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_HALF("axpy/h/2") {
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
        egblas_haxpy(N, fromFloat<T>(0.0), x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpy(N, fromFloat<T>(0.0), x_gpu, 1, y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(-2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(-2.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

#endif
