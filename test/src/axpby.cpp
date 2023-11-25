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

TEST_CASE("axpby/s/0", "[float][axpby]") {
    const size_t N = 137;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpby(N, 1.0, x.gpu(), 1, 1.0, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1.0f * i + 2.1f * i));
    }
}

TEST_CASE("axpby/s/1", "[float][axpby]") {
    const size_t N = 333;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpby(N, 0.2, x.gpu(), 1, 0.3, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.2f * i + 0.3 * 2.3 * i));
    }
}

TEST_CASE("axpby/s/2", "[float][axpby]") {
    const size_t N = 111;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpby(N, 0.2, x.gpu(), 3, 0.5, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(0.2f * i + 0.5f * 2.3f * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * i));
        }
    }
}

TEST_CASE("axpby/d/0", "[double][axpby]") {
    const size_t N = 137;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpby(N, 1.0, x.gpu(), 1, 1.1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1.0f * i + 1.1 * 2.1f * i));
    }
}

TEST_CASE("axpby/d/1", "[double][axpby]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpby(N, 0.2, x.gpu(), 1, 1.0, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.2f * i + 1.0 * 2.3 * i));
    }
}

TEST_CASE("axpby/d/2", "[double][axpby]") {
    const size_t N = 111;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpby(N, 0.2, x.gpu(), 3, 3.0, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(0.2f * i + 3.0 * 2.3f * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * i));
        }
    }
}

TEST_CASE("axpby/c/0", "[float][axpby]") {
    const size_t N = 27;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, i);
        y.cpu()[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxpby(N,
        make_cuComplex(1.0, 0.0), x.complex_gpu(), 1,
        make_cuComplex(1.0, 0.0), y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == TestComplex<float>(std::complex<float>(i, i) + std::complex<float>(-2.0f * i, 0.1f * i)));
    }
}

TEST_CASE("axpby/c/1", "[float][axpby]") {
    const size_t N = 33;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, -1.0f * i);
        y.cpu()[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxpby(N,
        make_cuComplex(1.0, 0.0), x.complex_gpu(), 1,
        make_cuComplex(1.0, 0.1), y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == TestComplex<float>(std::complex<float>(i, -1.0f * i) + std::complex<float>(1.0, 0.1) * std::complex<float>(-1.0f * i, 2.1f * i)));
    }
}

TEST_CASE("axpby/z/0", "[double][axpby]") {
    const size_t N = 27;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, i);
        y.cpu()[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxpby(N,
        make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1,
        make_cuDoubleComplex(1.0, 0.0), y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == TestComplex<double>(std::complex<double>(i, i) + std::complex<double>(-2.0f * i, 0.1f * i)));
    }
}

TEST_CASE("axpby/z/1", "[double][axpby]") {
    const size_t N = 33;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, -1.0 * i);
        y.cpu()[i] = std::complex<double>(-1.0 * i, 2.1 * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxpby(N,
        make_cuDoubleComplex(-1.0, 0.2), x.complex_gpu(), 1,
        make_cuDoubleComplex(1.1, 0.1), y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == TestComplex<double>(std::complex<double>(-1.0, 0.2) * std::complex<double>(i, -1.0 * i) + std::complex<double>(1.1, 0.1) * std::complex<double>(-1.0 * i, 2.1 * i)));
    }
}

TEST_CASE_TEMPLATE("axpby/i/0", T, int32_t, int64_t) {
    const size_t N = 137;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 21 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpby(N, 1, x.gpu(), 1, 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpby(N, 1, x.gpu(), 1, 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1 * i + 21 * i));
    }
}

TEST_CASE_TEMPLATE("axpby/i/1", T, int32_t, int64_t) {
    const size_t N = 333;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpby(N, 2, x.gpu(), 1, 3, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpby(N, 2, x.gpu(), 1, 3, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(2 * i + 3 * 23 * i));
    }
}

TEST_CASE_TEMPLATE("axpby/i/2", T, int32_t, int64_t) {
    const size_t N = 111;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 232 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpby(N, 2, x.gpu(), 3, 5, y.gpu(), 3);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpby(N, 2, x.gpu(), 3, 5, y.gpu(), 3);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(2 * i + 5 * 232 * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(232 * i));
        }
    }
}

#ifdef TEST_HALF

TEST_CASE_HALF("axpby/h/0") {
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
        egblas_haxpby(N, fromFloat<T>(1.0), x_gpu, 1, fromFloat<T>(1.0), y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpby(N, fromFloat<T>(1.0), x_gpu, 1, fromFloat<T>(1.0), y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(i + 2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(i + 2.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_HALF("axpby/h/1") {
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
        egblas_haxpby(N, fromFloat<T>(0.5), x_gpu, 1, fromFloat<T>(0.2), y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpby(N, fromFloat<T>(0.5), x_gpu, 1, fromFloat<T>(0.2), y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(0.5f * i + 0.2f * 3.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(0.5f * i + 0.2f * 3.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE_HALF("axpby/h/2") {
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
        egblas_haxpby(N, fromFloat<T>(0.0), x_gpu, 1, fromFloat<T>(0.0), y_gpu, 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpby(N, fromFloat<T>(0.0), x_gpu, 1, fromFloat<T>(0.0), y_gpu, 1);
    }

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

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
