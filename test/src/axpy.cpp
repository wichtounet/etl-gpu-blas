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

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpy(N, 1.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1.0f * i + 2.1f * i));
    }
}

TEST_CASE("axpy/s/1", "[float][axpy]") {
    const size_t N = 333;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpy(N, 0.2, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.2f * i + 2.3f * i));
    }
}

TEST_CASE("axpy/s/2", "[float][axpy]") {
    const size_t N = 111;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpy(N, 0.2, x.gpu(), 3, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(0.2f * i + 2.3f * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * i));
        }
    }
}

TEST_CASE("axpy/s/3", "[float][axpy]") {
    const size_t N = 111;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxpy(N, 0.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.0f * i + 2.3f * i));
    }
}

TEST_CASE("axpy/d/0", "[double][axpy]") {
    const size_t N = 137;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpy(N, 1.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1.0f * i + 2.1f * i));
    }
}

TEST_CASE("axpy/d/1", "[double][axpy]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpy(N, 0.2, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.2f * i + 2.3 * i));
    }
}

TEST_CASE("axpy/d/2", "[double][axpy]") {
    const size_t N = 111;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpy(N, 0.2, x.gpu(), 3, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(0.2f * i + 2.3f * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * i));
        }
    }
}

TEST_CASE("axpy/c/0", "[float][axpy]") {
    const size_t N = 27;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, i);
        y.cpu()[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxpy(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x.gpu()), 1, reinterpret_cast<cuComplex*>(y.gpu()), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx(i + -2.0f * i));
        REQUIRE(y.cpu()[i].imag() == Approx(i +  0.1f * i));
    }
}

TEST_CASE("axpy/c/1", "[float][axpy]") {
    const size_t N = 33;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, -1.0 * i);
        y.cpu()[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxpy(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x.gpu()), 1, reinterpret_cast<cuComplex*>(y.gpu()), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx(i + -1.0f * i));
        REQUIRE(y.cpu()[i].imag() == Approx(i +  0.1f * i));
    }
}

TEST_CASE("axpy/z/0", "[double][axpy]") {
    const size_t N = 27;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, i);
        y.cpu()[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxpy(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x.gpu()), 1, reinterpret_cast<cuDoubleComplex*>(y.gpu()), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx(i + -2.0f * i));
        REQUIRE(y.cpu()[i].imag() == Approx(i +  0.1f * i));
    }
}

TEST_CASE("axpy/z/1", "[double][axpy]") {
    const size_t N = 33;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, -1.0 * i);
        y.cpu()[i] = std::complex<double>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxpy(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x.gpu()), 1, reinterpret_cast<cuDoubleComplex*>(y.gpu()), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx(i + -1.0f * i));
        REQUIRE(y.cpu()[i].imag() == Approx(-1.0 * i +  2.1f * i));
    }
}

TEST_CASE_TEMPLATE("axpy/i/0", T, int8_t, int16_t, int32_t, int64_t) {
    const size_t N = 256;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = T(3) * T(i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int8_t>) {
        egblas_oaxpy(N, 1, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        egblas_waxpy(N, 1, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 1, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 1, x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == T(T(T(3) * T(i)) + T(i)));
    }
}

TEST_CASE_TEMPLATE("axpy/i/1", T, int8_t, int16_t, int32_t, int64_t) {
    const size_t N = 333;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = T(3) * T(i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int8_t>) {
        egblas_oaxpy(N, 2, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        egblas_waxpy(N, 2, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 2, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 2, x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == T(T(2) * T(i) + T(T(3) * T(i))));
    }
}

TEST_CASE_TEMPLATE("axpy/i/2", T, int8_t, int16_t, int32_t, int64_t ) {
    const size_t N = 111;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = T(3) * T(i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int8_t>) {
        egblas_oaxpy(N, 2, x.gpu(), 3, y.gpu(), 3);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        egblas_waxpy(N, 2, x.gpu(), 3, y.gpu(), 3);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 2, x.gpu(), 3, y.gpu(), 3);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 2, x.gpu(), 3, y.gpu(), 3);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == T(T(2) * T(i) + T(3) * T(i)));
        } else {
            REQUIRE(y.cpu()[i] == T(T(3) * T(i)));
        }
    }
}

#ifdef TEST_HALF

TEST_CASE_HALF("axpy/h/0") {
    const size_t N = 137;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = fromFloat<T>(i);
        y.cpu()[i] = fromFloat<T>(2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxpy(N, fromFloat<T>(1.0), x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpy(N, fromFloat<T>(1.0), x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y.cpu()[i]) == Approx(1.0f * i + 2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y.cpu()[i]) == Approx(1.0f * i + 2.1f * i).epsilon(half_eps));
    }
}

TEST_CASE_HALF("axpy/h/1") {
    const size_t N = 79;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = fromFloat<T>(i);
        y.cpu()[i] = fromFloat<T>(3.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxpy(N, fromFloat<T>(0.5), x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpy(N, fromFloat<T>(0.5), x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y.cpu()[i]) == Approx(0.5f * i + 3.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y.cpu()[i]) == Approx(0.5f * i + 3.1f * i).epsilon(half_eps));
    }
}

TEST_CASE_HALF("axpy/h/2") {
    const size_t N = 192;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = fromFloat<T>(i);
        y.cpu()[i] = fromFloat<T>(-2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxpy(N, fromFloat<T>(0.0), x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxpy(N, fromFloat<T>(0.0), x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y.cpu()[i]) == Approx(-2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y.cpu()[i]) == Approx(-2.1f * i).epsilon(half_eps));
    }
}

#endif
