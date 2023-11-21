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

TEST_CASE("axdy/s/0", "[float][axdy]") {
    const size_t N = 137;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.1f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxdy(N, 1.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx((2.1f * (i + 1)) / (1.0 * (i + 1))));
    }
}

TEST_CASE("axdy/s/1", "[float][axdy]") {
    const size_t N = 333;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.3f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxdy(N, 0.2, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
    }
}

TEST_CASE("axdy/s/2", "[float][axdy]") {
    const size_t N = 111;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.3f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxdy(N, 0.2, x.gpu(), 3, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * (i + 1)));
        }
    }
}

TEST_CASE("axdy/d/0", "[double][axdy]") {
    const size_t N = 137;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.1f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxdy(N, 1.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx((2.1f * (i + 1)) / (1.0 * (i + 1))));
    }
}

TEST_CASE("axdy/d/1", "[double][axdy]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.3f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxdy(N, 0.2, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
    }
}

TEST_CASE("axdy/d/2", "[double][axdy]") {
    const size_t N = 111;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.3f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxdy(N, 0.2, x.gpu(), 3, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx((2.3f * (i + 1)) / (0.2 * (i + 1))));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * (i + 1)));
        }
    }
}

TEST_CASE("axdy/c/0", "[float][axdy]") {
    const size_t N = 99;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);
    dual_array<std::complex<float>> y_b(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i]   = std::complex<float>(0.1 + 0.1 * i, 0.1 + 0.001f * i);
        y.cpu()[i]   = std::complex<float>(1.1f * i, 0.01f * i);
        y_b.cpu()[i] = std::complex<float>(1.1f * i, 0.01f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    std::complex<float> alpha(1.0f, 2.0f);
    egblas_caxdy(N, make_cuComplex(1.0f, 2.0f), x.complex_gpu(), 1, y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).real()));
        REQUIRE(y.cpu()[i].imag() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).imag()));
    }
}

TEST_CASE("axdy/c/1", "[float][axdy]") {
    const size_t N = 33;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);
    dual_array<std::complex<float>> y_b(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i]   = std::complex<float>(-0.01 + 0.1 * i, 0.2 + 0.004f * i);
        y.cpu()[i]   = std::complex<float>(1.3f * i, 0.29f * i);
        y_b.cpu()[i] = std::complex<float>(1.3f * i, 0.29f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    std::complex<float> alpha(1.5f, 2.5f);
    egblas_caxdy(N, make_cuComplex(1.5f, 2.5f), x.complex_gpu(), 1, y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).real()));
        REQUIRE(y.cpu()[i].imag() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).imag()));
    }
}

TEST_CASE("axdy/z/0", "[double][axdy]") {
    const size_t N = 48;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);
    dual_array<std::complex<double>> y_b(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i]   = std::complex<double>(0.2 + 0.2 * i, 0.1 + 0.001 * i);
        y.cpu()[i]   = std::complex<double>(1.1 * i, 0.01 * i);
        y_b.cpu()[i] = std::complex<double>(1.1 * i, 0.01 * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    std::complex<double> alpha(1.0, 2.0);
    egblas_zaxdy(N, make_cuDoubleComplex(1.0, 2.0),
                 x.complex_gpu(), 1,
                 y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).real()));
        REQUIRE(y.cpu()[i].imag() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).imag()));
    }
}

TEST_CASE("axdy/z/1", "[double][axdy]") {
    const size_t N = 39;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);
    dual_array<std::complex<double>> y_b(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i]   = std::complex<double>(-0.01 + 0.1 * i, 0.2 + 0.004 * i);
        y.cpu()[i]   = std::complex<double>(1.5 * i, 0.59 * i);
        y_b.cpu()[i] = std::complex<double>(1.5 * i, 0.59 * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    std::complex<double> alpha(1.5, 2.5);
    egblas_zaxdy(N, make_cuDoubleComplex(1.5, 2.5),
                 x.complex_gpu(), 1,
                 y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).real()));
        REQUIRE(y.cpu()[i].imag() == Approx((y_b.cpu()[i] / (alpha * x.cpu()[i])).imag()));
    }
}

TEST_CASE_TEMPLATE("axdy/i/0", T, int32_t, int64_t) {
    const size_t N = 137;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 21 * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxdy(N, 1, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxdy(N, 1, x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx((21 * (i + 1)) / (1 * (i + 1))));
    }
}

TEST_CASE_TEMPLATE("axdy/i/1", T, int32_t, int64_t) {
    const size_t N = 333;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 23 * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxdy(N, 2, x.gpu(), 1, y.gpu(), 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxdy(N, 2, x.gpu(), 1, y.gpu(), 1);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx((23 * (i + 1)) / (2 * (i + 1))));
    }
}

TEST_CASE_TEMPLATE("axdy/i/2", T, int32_t, int64_t) {
    const size_t N = 111;

    dual_array<T> x(N);
    dual_array<T> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 233 * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxdy(N, 21, x.gpu(), 3, y.gpu(), 3);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxdy(N, 21, x.gpu(), 3, y.gpu(), 3);
    }

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx((233 * (i + 1)) / (21 * (i + 1))));
        } else {
            REQUIRE(y.cpu()[i] == Approx(233 * (i + 1)));
        }
    }
}
