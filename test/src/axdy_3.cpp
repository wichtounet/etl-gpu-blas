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

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.1f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxdy_3(N, 1.0, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * (i + 1) / (2.1f * (i + 1))));
    }
}

TEST_CASE("axdy_3/s/1", "[float][axdy_3]") {
    const size_t N = 333;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i+1);
        y.cpu()[i] = 2.3f * (i+1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxdy_3(N, 0.2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx((i + 1) / (0.2 * 2.3 * (i + 1))));
    }
}

TEST_CASE("axdy_3/s/2", "[float][axdy_3]") {
    const size_t N = 120;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i+1);
        y.cpu()[i] = 2.3f * (i+1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxdy_3(N, 0.2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(((3 * i + 1) / (0.2f * 2.3f * (3 * i + 1)))));
    }
}

TEST_CASE("axdy_3/d/0", "[double][axdy_3]") {
    const size_t N = 137;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i+1);
        y.cpu()[i] = 2.1f * (i+1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxdy_3(N, 1.0, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx((1.0f * (i+1)) / (2.1f * (i+1))));
    }
}

TEST_CASE("axdy_3/d/1", "[double][axdy_3]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i+1);
        y.cpu()[i] = 2.3f * (i+1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxdy_3(N, 0.2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx((i+1) / (0.2 * 2.3 * (i+1))));
    }
}

TEST_CASE("axdy_3/d/2", "[double][axdy_3]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = (i + 1);
        y.cpu()[i] = 2.3f * (i + 1);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxdy_3(N, 0.2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx((3 * i + 1) / (0.2f * 2.3f * (3 * i + 1))));
    }
}

TEST_CASE("axdy_3/c/0", "[float][axdy_3]") {
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

    egblas_caxdy_3(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<double>(x.cpu()[i] / y.cpu()[i]));
    }
}

TEST_CASE("axdy_3/c/1", "[float][axdy_3]") {
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

    egblas_caxdy_3(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<double>(x.cpu()[i] / y.cpu()[i]));
    }
}

TEST_CASE("axdy_3/z/0", "[double][axdy_3]") {
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

    egblas_zaxdy_3(N, make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<double>(x.cpu()[i] / y.cpu()[i]));
    }
}

TEST_CASE("axdy_3/z/1", "[double][axdy_3]") {
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

    egblas_zaxdy_3(N, make_cuDoubleComplex(1.0, 0.1), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<double>(x.cpu()[i] / (std::complex<double>(1.0, 0.1) * y.cpu()[i])));
    }
}
