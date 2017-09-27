//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

namespace {

template <typename T>
T softplus(T x) {
    return std::log(T(1) + std::exp(x));
}

} // anonymous namespace

TEST_CASE("softplus/s/0", "[float][softplus]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = (i - 28.0f) / 100.9f;;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_ssoftplus(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/s/1", "[float][softplus]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 0.00211f * (i - 24.0f);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_ssoftplus(N, 0.2f, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/d/0", "[double][softplus]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 2.2 * (i - 19.5f) / 100.05f;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dsoftplus(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0 * softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/d/1", "[double][softplus]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 1.1 * (i - 99.9f) / 99.99f;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dsoftplus(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2 * softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/c/0", "[float][softplus]") {
    const size_t N = 137;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu = new std::complex<float>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(1.0f * (i / 1000.0f), 1.0f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_csoftplus(N, make_cuComplex(1.0f, 0.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/c/1", "[float][softplus]") {
    const size_t N = 338;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu = new std::complex<float>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i / 998.0f, -2.0f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_csoftplus(N, make_cuComplex(1.0f, 1.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(std::complex<float>(1.0f, 1.0f) * softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/z/0", "[double][softplus]") {
    const size_t N = 137;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu = new std::complex<double>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i / 999.0, -1.1 * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zsoftplus(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("softplus/z/1", "[double][softplus]") {
    const size_t N = 338;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu = new std::complex<double>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(0.1 + i / 996.0, 0.1 + 2.4 * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zsoftplus(N, make_cuDoubleComplex(0.1, 2.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(std::complex<double>(0.1, 2.0) * softplus(x_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
