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

TEST_CASE("axmy_3/s/0", "[float][axmy_3]") {
    const size_t N = 137;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy_3(N, 1.0, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * i * 2.1f * i));
    }
}

TEST_CASE("axmy_3/s/1", "[float][axmy_3]") {
    const size_t N = 333;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy_3(N, 0.2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * i * 2.3 * i));
    }
}

TEST_CASE("axmy_3/s/2", "[float][axmy_3]") {
    const size_t N = 120;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy_3(N, 0.2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * (3 * i) * 2.3f * (3 * i)));
    }
}

TEST_CASE("axmy_3/s/3", "[float][axmy_3]") {
    const size_t N = 199;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = -1.1f * i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy_3(N, 0.0f, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.0f));
    }
}

TEST_CASE("axmy_3/d/0", "[double][axmy_3]") {
    const size_t N = 137;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxmy_3(N, 1.0, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * i * 2.1f * i));
    }
}

TEST_CASE("axmy_3/d/1", "[double][axmy_3]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxmy_3(N, 0.2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * i * 2.3 * i));
    }
}

TEST_CASE("axmy_3/d/2", "[double][axmy_3]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxmy_3(N, 0.2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * (3 * i) * 2.3f * (3 * i)));
    }
}

TEST_CASE("axmy_3/c/0", "[float][axmy_3]") {
    const size_t N = 27;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);
    dual_array<std::complex<float>> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, i);
        y.cpu()[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxmy_3(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx((x.cpu()[i] * y.cpu()[i]).real()));
        REQUIRE(yy.cpu()[i].imag() == Approx((x.cpu()[i] * y.cpu()[i]).imag()));
    }
}

TEST_CASE("axmy_3/c/1", "[float][axmy_3]") {
    const size_t N = 33;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);
    dual_array<std::complex<float>> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, -1.0 * i);
        y.cpu()[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxmy_3(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx((x.cpu()[i] * y.cpu()[i]).real()));
        REQUIRE(yy.cpu()[i].imag() == Approx((x.cpu()[i] * y.cpu()[i]).imag()));
    }
}

TEST_CASE("axmy_3/z/0", "[double][axmy_3]") {
    const size_t N = 27;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);
    dual_array<std::complex<double>> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, i);
        y.cpu()[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxmy_3(N, make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx((x.cpu()[i] * y.cpu()[i]).real()));
        REQUIRE(yy.cpu()[i].imag() == Approx((x.cpu()[i] * y.cpu()[i]).imag()));
    }
}

TEST_CASE("axmy_3/z/1", "[double][axmy_3]") {
    const size_t N = 33;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);
    dual_array<std::complex<double>> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, -1.0 * i);
        y.cpu()[i] = std::complex<double>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxmy_3(N, make_cuDoubleComplex(1.0, 0.1), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx((std::complex<double>(1.0, 0.1) * x.cpu()[i] * y.cpu()[i]).real()));
        REQUIRE(yy.cpu()[i].imag() == Approx((std::complex<double>(1.0, 0.1) * x.cpu()[i] * y.cpu()[i]).imag()));
    }
}

TEST_CASE_TEMPLATE("axmy_3/i/0", T, int32_t, int64_t) {
    const size_t N = 137;

    dual_array<T> x(N);
    dual_array<T> y(N);
    dual_array<T> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 21 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxmy_3(N, 1, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxmy_3(N, 1, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    }

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1 * i * 21 * i));
    }
}

TEST_CASE_TEMPLATE("axmy_3/i/1", T, int32_t, int64_t) {
    const size_t N = 333;

    dual_array<T> x(N);
    dual_array<T> y(N);
    dual_array<T> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 43 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxmy_3(N, 20, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxmy_3(N, 20, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    }

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(20 * i * 43* i));
    }
}

TEST_CASE_TEMPLATE("axmy_3/i/2", T, int32_t, int64_t) {
    const size_t N = 120;

    dual_array<T> x(N);
    dual_array<T> y(N);
    dual_array<T> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxmy_3(N, 10, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxmy_3(N, 10, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);
    }

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(10 * (3 * i) * 23 * (3 * i)));
    }
}

#ifdef TEST_HALF

TEST_CASE_HALF("axmy_3/h/0") {
    const size_t N = 137;

    dual_array<T> x(N);
    dual_array<T> y(N);
    dual_array<T> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = fromFloat<T>(i);
        y.cpu()[i] = fromFloat<T>(2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxmy_3(N, fromFloat<T>(1.0), x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxmy_3(N, fromFloat<T>(1.0), x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    }

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(yy.cpu()[i]) == Approx(1.0f * i * 2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(yy.cpu()[i]) == Approx(1.0f * i * 2.1f * i).epsilon(half_eps));
    }
}

TEST_CASE_HALF("axmy_3/h/1") {
    const size_t N = 339;

    dual_array<T> x(N);
    dual_array<T> y(N);
    dual_array<T> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = fromFloat<T>(i);
        y.cpu()[i] = fromFloat<T>(1.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxmy_3(N, fromFloat<T>(-0.1), x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxmy_3(N, fromFloat<T>(-0.1), x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    }

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(yy.cpu()[i]) == Approx(-0.1f * i * 1.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(yy.cpu()[i]) == Approx(-0.1f * i * 1.1f * i).epsilon(half_eps));
    }
}

TEST_CASE_HALF("axmy_3/h/2") {
    const size_t N = 256;

    dual_array<T> x(N);
    dual_array<T> y(N);
    dual_array<T> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = fromFloat<T>(i);
        y.cpu()[i] = fromFloat<T>(4.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, fp16>) {
        egblas_haxmy_3(N, fromFloat<T>(0), x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    } else if constexpr (std::is_same_v<T, bf16>) {
        egblas_baxmy_3(N, fromFloat<T>(0), x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);
    }

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(yy.cpu()[i]) == Approx(0.0f).epsilon(half_eps));
        REQUIRE(__low2float(yy.cpu()[i]) == Approx(0.0f).epsilon(half_eps));
    }
}

#endif
