//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("imag/c/0", "[float][imag]") {
    const size_t N = 137;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(1.0f + i / 1000.0f, 1.0f * i);
    }

    std::complex<float>* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_cimag(N, 1.0f, reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<float*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (std::imag(std::complex<float>(1.0f + i / 1000.0f, 1.0f * i))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("imag/c/1", "[float][imag]") {
    const size_t N = 338;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i / 998.0f, -2.0f * i);
    }

    std::complex<float>* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_cimag(N, 2.2f, reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<float*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (2.2f * std::imag(std::complex<float>(i / 998.0f, -2.0f * i))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("imag/z/0", "[double][imag]") {
    const size_t N = 137;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i / 999.0, -1.1 * i);
    }

    std::complex<double>* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_zimag(N, 1.0, reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<double*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (std::imag(std::complex<double>(i / 999.0, -1.1 * i))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("imag/z/1", "[double][imag]") {
    const size_t N = 338;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i / 996.0, -2.4 * i);
    }

    std::complex<double>* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_zimag(N, 0.2, reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<double*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == (0.2 * std::imag(std::complex<double>(i / 996.0, -2.4 * i))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
