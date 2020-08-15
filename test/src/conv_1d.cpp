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
    dual_array<float> k(K);
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
    dual_array<float> k(K);
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
    dual_array<double> k(K);
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
    dual_array<double> k(K);
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

TEST_CASE("conv1_same/s/0", "[float][conv1_same]") {
    const size_t N = 3;
    const size_t K = 3;

    dual_array<float> x(N);
    dual_array<float> k(K);
    dual_array<float> y(N);

    x.cpu()[0] = 1.0f;
    x.cpu()[1] = 2.0f;
    x.cpu()[2] = 3.0f;

    k.cpu()[0] = 0.0f;
    k.cpu()[1] = 1.0f;
    k.cpu()[2] = 0.5f;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_sconv1_same(N, K, 1.0f, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(1.0f));
    REQUIRE(y.cpu()[1] == Approx(2.5f));
    REQUIRE(y.cpu()[2] == Approx(4.0f));
}

TEST_CASE("conv1_same/s/1", "[float][conv1_same]") {
    const size_t N = 6;
    const size_t K = 4;

    dual_array<float> x(N);
    dual_array<float> k(K);
    dual_array<float> y(N);

    x.cpu()[0] = 1.0f;
    x.cpu()[1] = 2.0f;
    x.cpu()[2] = 3.0f;
    x.cpu()[3] = 0.0f;
    x.cpu()[4] = 0.5f;
    x.cpu()[5] = 2.0f;

    k.cpu()[0] = 0.0f;
    k.cpu()[1] = 0.5f;
    k.cpu()[2] = 1.0f;
    k.cpu()[3] = 0.0f;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_sconv1_same(N, K, 1.0f, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(2.0f));
    REQUIRE(y.cpu()[1] == Approx(3.5f));
    REQUIRE(y.cpu()[2] == Approx(3.0f));
    REQUIRE(y.cpu()[3] == Approx(0.25f));
    REQUIRE(y.cpu()[4] == Approx(1.5f));
    REQUIRE(y.cpu()[5] == Approx(2.0f));
}

TEST_CASE("conv1_same/d/0", "[double][conv1_same]") {
    const size_t N = 3;
    const size_t K = 3;

    dual_array<double> x(N);
    dual_array<double> k(K);
    dual_array<double> y(N);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;

    k.cpu()[0] = 0.0;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 0.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_dconv1_same(N, K, 1.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(1.0));
    REQUIRE(y.cpu()[1] == Approx(2.5));
    REQUIRE(y.cpu()[2] == Approx(4.0));
}

TEST_CASE("conv1_same/d/1", "[double][conv1_same]") {
    const size_t N = 6;
    const size_t K = 4;

    dual_array<double> x(N);
    dual_array<double> k(K);
    dual_array<double> y(N);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;
    x.cpu()[3] = 0.0;
    x.cpu()[4] = 0.5;
    x.cpu()[5] = 2.0;

    k.cpu()[0] = 0.0;
    k.cpu()[1] = 0.5;
    k.cpu()[2] = 1.0;
    k.cpu()[3] = 0.0;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_dconv1_same(N, K, 2.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(2.0 * 2.0));
    REQUIRE(y.cpu()[1] == Approx(2.0 * 3.5));
    REQUIRE(y.cpu()[2] == Approx(2.0 * 3.0));
    REQUIRE(y.cpu()[3] == Approx(2.0 * 0.25));
    REQUIRE(y.cpu()[4] == Approx(2.0 * 1.5));
    REQUIRE(y.cpu()[5] == Approx(2.0 * 2.0));
}

// Full Convolution

TEST_CASE("conv1_full/s/0", "[float][conv1_full]") {
    const size_t N = 3;
    const size_t K = 3;

    dual_array<float> x(N);
    dual_array<float> k(K);
    dual_array<float> y(N + K - 1);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;

    k.cpu()[0] = 0.0;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 0.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_sconv1_full(N, K, 1.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(0.0));
    REQUIRE(y.cpu()[1] == Approx(1.0));
    REQUIRE(y.cpu()[2] == Approx(2.5));
    REQUIRE(y.cpu()[3] == Approx(4.0));
    REQUIRE(y.cpu()[4] == Approx(1.5));
}

TEST_CASE("conv1_full/s/1", "[float][conv1_full]") {
    const size_t N = 5;
    const size_t K = 3;

    dual_array<float> x(N);
    dual_array<float> k(K);
    dual_array<float> y(N + K - 1);

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

    egblas_sconv1_full(N, K, 1.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(0.5));
    REQUIRE(y.cpu()[1] == Approx(2.0));
    REQUIRE(y.cpu()[2] == Approx(5.0));
    REQUIRE(y.cpu()[3] == Approx(8.0));
    REQUIRE(y.cpu()[4] == Approx(11.0));
    REQUIRE(y.cpu()[5] == Approx(11.0));
    REQUIRE(y.cpu()[6] == Approx(7.5));
}

TEST_CASE("conv1_full/d/0", "[double][conv1_full]") {
    const size_t N = 3;
    const size_t K = 3;

    dual_array<double> x(N);
    dual_array<double> k(K);
    dual_array<double> y(N + K - 1);

    x.cpu()[0] = 1.0;
    x.cpu()[1] = 2.0;
    x.cpu()[2] = 3.0;

    k.cpu()[0] = 0.0;
    k.cpu()[1] = 1.0;
    k.cpu()[2] = 0.5;

    x.cpu_to_gpu();
    k.cpu_to_gpu();

    egblas_dconv1_full(N, K, 1.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(0.0));
    REQUIRE(y.cpu()[1] == Approx(1.0));
    REQUIRE(y.cpu()[2] == Approx(2.5));
    REQUIRE(y.cpu()[3] == Approx(4.0));
    REQUIRE(y.cpu()[4] == Approx(1.5));
}

TEST_CASE("conv1_full/d/1", "[double][conv1_full]") {
    const size_t N = 5;
    const size_t K = 3;

    dual_array<double> x(N);
    dual_array<double> k(K);
    dual_array<double> y(N + K - 1);

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

    egblas_dconv1_full(N, K, 1.0, x.gpu(), 1, k.gpu(), 1, y.gpu(), 1);

    y.gpu_to_cpu();

    REQUIRE(y.cpu()[0] == Approx(0.5));
    REQUIRE(y.cpu()[1] == Approx(2.0));
    REQUIRE(y.cpu()[2] == Approx(5.0));
    REQUIRE(y.cpu()[3] == Approx(8.0));
    REQUIRE(y.cpu()[4] == Approx(11.0));
    REQUIRE(y.cpu()[5] == Approx(11.0));
    REQUIRE(y.cpu()[6] == Approx(7.5));
}
