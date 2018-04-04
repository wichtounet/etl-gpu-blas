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

TEST_CASE("axpby/i/0", "[int32_t][axpby]") {
    const size_t N = 137;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 21 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_iaxpby(N, 1, x.gpu(), 1, 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1 * i + 21 * i));
    }
}

TEST_CASE("axpby/i/1", "[int32_t][axpby]") {
    const size_t N = 333;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_iaxpby(N, 2, x.gpu(), 1, 3, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(2 * i + 3 * 23 * i));
    }
}

TEST_CASE("axpby/i/2", "[int32_t][axpby]") {
    const size_t N = 111;

    dual_array<int32_t> x(N);
    dual_array<int32_t> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 232 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_iaxpby(N, 2, x.gpu(), 3, 5, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(2 * i + 5 * 232 * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(232 * i));
        }
    }
}

TEST_CASE("axpby/l/0", "[int64_t][axpby]") {
    const size_t N = 137;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 21 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_laxpby(N, 1, x.gpu(), 1, 11, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(1 * i + 11 * 21 * i));
    }
}

TEST_CASE("axpby/l/1", "[int64_t][axpby]") {
    const size_t N = 333;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_laxpby(N, 2, x.gpu(), 1, 1, y.gpu(), 1);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y.cpu()[i] == Approx(2 * i + 1 * 23 * i));
    }
}

TEST_CASE("axpby/l/2", "[int64_t][axpby]") {
    const size_t N = 111;

    dual_array<int64_t> x(N);
    dual_array<int64_t> y(N);

    for (size_t i = 0; i < N; ++i) {
        x.cpu()[i] = i;
        y.cpu()[i] = 23 * i;
    }

    x.cpu_to_gpu();
    y.cpu_to_gpu();

    egblas_laxpby(N, 2, x.gpu(), 3, 33, y.gpu(), 3);

    y.gpu_to_cpu();

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y.cpu()[i] == Approx(2 * i + 33 * 23 * i));
        } else {
            REQUIRE(y.cpu()[i] == Approx(23 * i));
        }
    }
}
