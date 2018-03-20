//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("axmy/s/0", "[float][axmy]") {
    const size_t N = 137;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy(N, 1.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1.0f * i * 2.1f * i));
    }
}

TEST_CASE("axmy/s/1", "[float][axmy]") {
    const size_t N = 333;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy(N, 0.2, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.2f * i * 2.3 * i));
    }
}

TEST_CASE("axmy/s/2", "[float][axmy]") {
    const size_t N = 111;

    dual_array<float> x(N);
    dual_array<float> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_saxmy(N, 0.2, x.gpu(), 3, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(0.2f * i * 2.3f * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * i));
        }
    }
}

TEST_CASE("axmy/d/0", "[double][axmy]") {
    const size_t N = 137;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.1f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxmy(N, 1.0, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1.0f * i * 2.1f * i));
    }
}

TEST_CASE("axmy/d/1", "[double][axmy]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxmy(N, 0.2, x.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(0.2f * i * 2.3 * i));
    }
}

TEST_CASE("axmy/d/2", "[double][axmy]") {
    const size_t N = 111;

    dual_array<double> x(N);
    dual_array<double> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxmy(N, 0.2, x.gpu(), 3, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(0.2f * i * 2.3f * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(2.3f * i));
        }
    }
}

TEST_CASE("axmy/c/0", "[float][axmy]") {
    const size_t N = 27;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, i);
        y.cpu()[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxmy(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((i * -2.0f * i) - (i * 0.1f * i)));
        REQUIRE(y.cpu()[i].imag() == Approx((i * -2.0f * i) + (i * 0.1f * i)));
    }
}

TEST_CASE("axmy/c/1", "[float][axmy]") {
    const size_t N = 33;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, -1.0 * i);
        y.cpu()[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxmy(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((i * -1.0f * i) - (-1.0 * i * 2.1f * i)));
        REQUIRE(y.cpu()[i].imag() == Approx((-1.0 * i * -1.0f * i) + (i * 2.1f * i)));
    }
}

TEST_CASE("axmy/z/0", "[double][axmy]") {
    const size_t N = 27;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, i);
        y.cpu()[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxmy(N, make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((i * -2.0f * i) - (i * 0.1f * i)));
        REQUIRE(y.cpu()[i].imag() == Approx((i * -2.0f * i) + (i * 0.1f * i)));
    }
}

TEST_CASE("axmy/z/1", "[double][axmy]") {
    const size_t N = 33;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, -1.0 * i);
        y.cpu()[i] = std::complex<double>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxmy(N, make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i].real() == Approx((i * -1.0f * i) - (-1.0 * i * 2.1f * i)));
        REQUIRE(y.cpu()[i].imag() == Approx((-1.0 * i * -1.0f * i) + (i * 2.1f * i)));
    }
}
