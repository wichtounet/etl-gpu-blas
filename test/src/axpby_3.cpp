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

TEST_CASE("axpby_3/s/0", "[float][axpby_3]") {
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

    egblas_saxpby_3(N, 1.0, x.gpu(), 1, 1.0, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * i + 2.1f * i));
    }
}

TEST_CASE("axpby_3/s/1", "[float][axpby_3]") {
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

    egblas_saxpby_3(N, 0.2, x.gpu(), 1, 0.3, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2f * i + 0.3 * 2.3 * i));
    }
}

TEST_CASE("axpby_3/s/2", "[float][axpby_3]") {
    const size_t N = 111;

    dual_array<float> x(N);
    dual_array<float> y(N);
    dual_array<float> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
        yy.cpu()[i] = 0.3f * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();
    yy.cpu_to_gpu();

    egblas_saxpby_3(N, 0.2, x.gpu(), 3, 0.5, y.gpu(), 3, yy.gpu(), 3);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(yy.cpu()[i] == Approx(0.2f * i + 0.5f * 2.3f * i));
        } else {
            REQUIRE(yy.cpu()[i] == Approx(0.3f * i));
        }
    }
}

TEST_CASE("axpby_3/d/0", "[double][axpby_3]") {
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

    egblas_daxpby_3(N, 1.0, x.gpu(), 1, 1.1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1.0f * i + 1.1 * 2.1f * i));
    }
}

TEST_CASE("axpby_3/d/1", "[double][axpby_3]") {
    const size_t N = 333;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_daxpby_3(N, 0.2, x.gpu(), 1, 1.0, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(0.2 * i + 1.0 * 2.3 * i));
    }
}

TEST_CASE("axpby_3/d/2", "[double][axpby_3]") {
    const size_t N = 111;

    dual_array<double> x(N);
    dual_array<double> y(N);
    dual_array<double> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 2.3f * i;
        yy.cpu()[i] = i + 1;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();
    yy.cpu_to_gpu();

    egblas_daxpby_3(N, 0.2, x.gpu(), 3, 3.0, y.gpu(), 3, yy.gpu(), 3);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(yy.cpu()[i] == Approx(0.2f * i + 3.0 * 2.3f * i));
        } else {
            REQUIRE(yy.cpu()[i] == Approx(i + 1.0f));
        }
    }
}

TEST_CASE("axpby_3/c/0", "[float][axpby_3]") {
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

    egblas_caxpby_3(N,
        make_cuComplex(1.0, 0.0), x.complex_gpu(), 1,
        make_cuComplex(1.0, 0.0), y.complex_gpu(), 1,
        yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<float>(std::complex<float>(i, i) + std::complex<float>(-2.0f * i, 0.1f * i)));
    }
}

TEST_CASE("axpby_3/c/1", "[float][axpby_3]") {
    const size_t N = 33;

    dual_array<std::complex<float>> x(N);
    dual_array<std::complex<float>> y(N);
    dual_array<std::complex<float>> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<float>(i, -1.0f * i);
        y.cpu()[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_caxpby_3(N,
        make_cuComplex(1.0, 0.0), x.complex_gpu(), 1,
        make_cuComplex(1.0, 0.1), y.complex_gpu(), 1,
        yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<float>(std::complex<float>(i, -1.0f * i) + std::complex<float>(1.0, 0.1) * std::complex<float>(-1.0f * i, 2.1f * i)));
    }
}

TEST_CASE("axpby_3/z/0", "[double][axpby_3]") {
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

    egblas_zaxpby_3(N,
        make_cuDoubleComplex(1.0, 0.0), x.complex_gpu(), 1,
        make_cuDoubleComplex(1.0, 0.0), y.complex_gpu(), 1,
        yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<double>(std::complex<double>(i, i) + std::complex<double>(-2.0f * i, 0.1f * i)));
    }
}

TEST_CASE("axpby_3/z/1", "[double][axpby_3]") {
    const size_t N = 33;

    dual_array<std::complex<double>> x(N);
    dual_array<std::complex<double>> y(N);
    dual_array<std::complex<double>> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = std::complex<double>(i, -1.0 * i);
        y.cpu()[i] = std::complex<double>(-1.0 * i, 2.1 * i);
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_zaxpby_3(N,
        make_cuDoubleComplex(-1.0, 0.2), x.complex_gpu(), 1,
        make_cuDoubleComplex(1.1, 0.1), y.complex_gpu(), 1,
        yy.complex_gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == TestComplex<double>(std::complex<double>(-1.0, 0.2) * std::complex<double>(i, -1.0 * i) + std::complex<double>(1.1, 0.1) * std::complex<double>(-1.0 * i, 2.1 * i)));
    }
}

TEST_CASE("axpby_3/i/0", "[int32_t][axpby_3]") {
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

    egblas_iaxpby_3(N, 1, x.gpu(), 1, 1, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1 * i + 21 * i));
    }
}

TEST_CASE("axpby_3/i/1", "[int32_t][axpby_3]") {
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

    egblas_iaxpby_3(N, 2, x.gpu(), 1, 3, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(2 * i + 3 * 23 * i));
    }
}

TEST_CASE("axpby_3/i/2", "[int32_t][axpby_3]") {
    const size_t N = 111;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);
    dual_array<int32_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
        yy.cpu()[i] = 3 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();
    yy.cpu_to_gpu();

    egblas_iaxpby_3(N, 2, x.gpu(), 3, 5, y.gpu(), 3, yy.gpu(), 3);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(yy.cpu()[i] == Approx(2 * i + 5 * 23 * i));
        } else {
            REQUIRE(yy.cpu()[i] == Approx(3 * i));
        }
    }
}

TEST_CASE("axpby_3/l/0", "[int64_t][axpby_3]") {
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

    egblas_laxpby_3(N, 1, x.gpu(), 1, 11, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(1 * i + 11 * 21 * i));
    }
}

TEST_CASE("axpby_3/l/1", "[int64_t][axpby_3]") {
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

    egblas_laxpby_3(N, 2, x.gpu(), 1, 10, y.gpu(), 1, yy.gpu(), 1);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(yy.cpu()[i] == Approx(2 * i + 10 * 23 * i));
    }
}

TEST_CASE("axpby_3/l/2", "[int64_t][axpby_3]") {
    const size_t N = 111;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);
    dual_array<int64_t> yy(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 235 * i;
        yy.cpu()[i] = i + 1;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();
    yy.cpu_to_gpu();

    egblas_laxpby_3(N, 2, x.gpu(), 3, 30, y.gpu(), 3, yy.gpu(), 3);

    yy.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(yy.cpu()[i] == Approx(2 * i + 30 * 235 * i));
        } else {
            REQUIRE(yy.cpu()[i] == Approx(i + 1));
        }
    }
}
