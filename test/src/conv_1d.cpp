//=======================================================================
// Copyright (c) 2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("conv1_valid/s/0", "[float][conv1_valid]") {
    const size_t N = 3;
    const size_t K = 3;

    dual_array<float> x(N);
    dual_array<float> k(N);
    dual_array<float> y(N - K + 1);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;

    k.cpu()[0] = 0.0;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 0.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_sconv1_valid(N, K, 1.0f, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(2.5f));
}

TEST_CASE("conv1_valid/s/1", "[float][conv1_valid]") {
    const size_t N = 5;
    const size_t K = 3;

    dual_array<float> x(N);
    dual_array<float> k(N);
    dual_array<float> y(N - K + 1);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;
    x.cpu()[3] = 4.0;
    x.cpu()[4] = 5.0;

    k.cpu()[0] = 0.5;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 1.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_sconv1_valid(N, K, 1.0f, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(5.0f));
    REQUIRE(y.cpu()[1] == Approx(8.0f));
    REQUIRE(y.cpu()[2] == Approx(11.0f));
}

TEST_CASE("conv1_valid/d/0", "[double][conv1_valid]") {
    const size_t N = 3;
    const size_t K = 3;

    dual_array<double> x(N);
    dual_array<double> k(N);
    dual_array<double> y(N - K + 1);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;

    k.cpu()[0] = 0.0;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 0.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_dconv1_valid(N, K, 1.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(2.5));
}

TEST_CASE("conv1_valid/d/1", "[double][conv1_valid]") {
    const size_t N = 5;
    const size_t K = 3;

    dual_array<double> x(N);
    dual_array<double> k(N);
    dual_array<double> y(N - K + 1);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;
    x.cpu()[3] = 4.0;
    x.cpu()[4] = 5.0;

    k.cpu()[0] = 0.5;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 1.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_dconv1_valid(N, K, 2.0f, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(10.0));
    REQUIRE(y.cpu()[1] == Approx(16.0));
    REQUIRE(y.cpu()[2] == Approx(22.0));
}
