//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("axpy_3/s/0", "[float][axpy_3]") {
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

    egblas_saxpy_3(N, 1.0, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * i + 2.1f * i));
    }
}

TEST_CASE("axpy_3/s/1", "[float][axpy_3]") {
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

    egblas_saxpy_3(N, 0.2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * i + 2.3 * i));
    }
}

TEST_CASE("axpy_3/s/2", "[float][axpy_3]") {
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

    egblas_saxpy_3(N, 0.2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * (3 * i) + 2.3f * (3 * i)));
    }
}

TEST_CASE("axpy_3/d/0", "[double][axpy_3]") {
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

    egblas_daxpy_3(N, 1.0, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * i + 2.1f * i));
    }
}

TEST_CASE("axpy_3/d/1", "[double][axpy_3]") {
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

    egblas_daxpy_3(N, 0.2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * i + 2.3 * i));
    }
}

TEST_CASE("axpy_3/d/2", "[double][axpy_3]") {
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

    egblas_daxpy_3(N, 0.2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * (3 * i) + 2.3f * (3 * i)));
    }
}

TEST_CASE("axpy_3/c/0", "[float][axpy_3]") {
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

    egblas_caxpy_3(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx(i + -2.0f * i));
        REQUIRE(yy.cpu()[i].imag() == Approx(i +  0.1f * i));
    }
}

TEST_CASE("axpy_3/c/1", "[float][axpy_3]") {
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

    egblas_caxpy_3(N, make_cuComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx(i + -1.0f * i));
        REQUIRE(yy.cpu()[i].imag() == Approx(i +  0.1f * i));
    }
}

TEST_CASE("axpy_3/z/0", "[double][axpy_3]") {
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

    egblas_zaxpy_3(N, make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx(i + -2.0f * i));
        REQUIRE(yy.cpu()[i].imag() == Approx(i +  0.1f * i));
    }
}

TEST_CASE("axpy_3/z/1", "[double][axpy_3]") {
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

    egblas_zaxpy_3(N, make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1, y.complex_gpu(), 1, yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i].real() == Approx(i + -1.0f * i));
        REQUIRE(yy.cpu()[i].imag() == Approx(-1.0 * i +  2.1f * i));
    }
}

TEST_CASE("axpy_3/i/0", "[int32_t][axpy_3]") {
    const size_t N = 137;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);
    dual_array<int32_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 21 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_iaxpy_3(N, 1, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1 * i + 21 * i));
    }
}

TEST_CASE("axpy_3/i/1", "[int32_t][axpy_3]") {
    const size_t N = 333;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);
    dual_array<int32_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_iaxpy_3(N, 2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(2 * i + 23 * i));
    }
}

TEST_CASE("axpy_3/i/2", "[int32_t][axpy_3]") {
    const size_t N = 120;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);
    dual_array<int32_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_iaxpy_3(N, 2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(2 * (3 * i) + 23 * (3 * i)));
    }
}

TEST_CASE("axpy_3/l/0", "[int64_t][axpy_3]") {
    const size_t N = 137;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);
    dual_array<int64_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 21 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_laxpy_3(N, 1, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1 * i + 21 * i));
    }
}

TEST_CASE("axpy_3/l/1", "[int64_t][axpy_3]") {
    const size_t N = 333;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);
    dual_array<int64_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_laxpy_3(N, 2, x.gpu(), 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(2 * i + 23 * i));
    }
}

TEST_CASE("axpy_3/l/2", "[int64_t][axpy_3]") {
    const size_t N = 333;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);
    dual_array<int64_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_laxpy_3(N, 2, x.gpu(), 3, y.gpu(), 3, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N / 3; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(2 * (3 * i) + 23 * (3 * i)));
    }
}
