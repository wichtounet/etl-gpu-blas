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

TEST_CASE("min/s/0", "[float][min]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];
    float* z_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 1.0 + i;
        y_cpu[i] = 0.021f * i;
    }

    float* x_gpu;
    float* y_gpu;
    float* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_smin(N, 1.0, x_gpu, 1, y_gpu, 1, z_gpu, 1);
    egblas_smin(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * std::min(1.0f + i, 0.021f * i)));
        REQUIRE(z_cpu[i] == Approx(1.0f * std::min(1.0f + i, 0.021f * i)));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/s/1", "[float][min]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];
    float* z_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 1.0 + i;
        y_cpu[i] = 0.023f * i;
    }

    float* x_gpu;
    float* y_gpu;
    float* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_smin(N, 0.2, x_gpu, 1, y_gpu, 1, z_gpu, 1);
    egblas_smin(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * std::min(1.0f + i, 0.023f * i)));
        REQUIRE(z_cpu[i] == Approx(0.2f * std::min(1.0f + i, 0.023f * i)));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/s/2", "[float][min]") {
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];
    float* z_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = 1.0 + i;
        y_cpu[i] = 0.023f * i;
    }

    float* x_gpu;
    float* y_gpu;
    float* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_smin(N, 0.2, x_gpu, 3, y_gpu, 3, z_gpu, 3);
    egblas_smin(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(0.2f * std::min(1.0f + i, 0.023f * i)));
            REQUIRE(z_cpu[i] == Approx(0.2f * std::min(1.0f + i, 0.023f * i)));
        } else {
            REQUIRE(y_cpu[i] == Approx(0.023f * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/d/0", "[double][min]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];
    double* z_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 0.021f * i;
    }

    double* x_gpu;
    double* y_gpu;
    double* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dmin(N, 1.0, x_gpu, 1, y_gpu, 1, z_gpu, 1);
    egblas_dmin(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0 * std::min(1.0 * i, 0.021 * i)));
        REQUIRE(z_cpu[i] == Approx(1.0 * std::min(1.0 * i, 0.021 * i)));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/d/1", "[double][min]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];
    double* z_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 0.023f * i;
    }

    double* x_gpu;
    double* y_gpu;
    double* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dmin(N, 0.2, x_gpu, 1, y_gpu, 1, z_gpu, 1);
    egblas_dmin(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2 * std::min(1.0 * i, 0.023 * i)));
        REQUIRE(z_cpu[i] == Approx(0.2 * std::min(1.0 * i, 0.023 * i)));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/d/2", "[double][min]") {
    const size_t N = 111;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];
    double* z_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 0.023f * i;
    }

    double* x_gpu;
    double* y_gpu;
    double* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dmin(N, 0.2, x_gpu, 3, y_gpu, 3, z_gpu, 3);
    egblas_dmin(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(0.2 * std::min(1.0 * i, 0.023 * i)));
            REQUIRE(z_cpu[i] == Approx(0.2 * std::min(1.0 * i, 0.023 * i)));
        } else {
            REQUIRE(y_cpu[i] == Approx(0.023 * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] z_cpu;
}

namespace {

template<typename T>
std::complex<T> my_min(std::complex<T> lhs, std::complex<T> rhs){
    if (lhs.real() < rhs.real()) {
        return lhs;
    } else if (rhs.real() < lhs.real()) {
        return rhs;
    } else {
        if (lhs.imag() < rhs.imag()) {
            return lhs;
        } else {
            return rhs;
        }
    }
}

} // end of anonymous namespace

TEST_CASE("min/c/0", "[float][min]") {
    const size_t N = 27;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu0 = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];
    std::complex<float>* z_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(1.0 + i, 0.0f);
        y_cpu[i] = std::complex<float>(0.01f * i, 0.1f * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    std::complex<float>* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_cmin(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1, reinterpret_cast<cuComplex*>(z_gpu), 1);
    egblas_cmin(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(std::complex<float>(1.0, 0.0) * my_min(x_cpu[i], y_cpu0[i])));
        REQUIRE(z_cpu[i] == TestComplex<float>(std::complex<float>(1.0, 0.0) * my_min(x_cpu[i], y_cpu0[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu0;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/c/1", "[float][min]") {
    const size_t N = 33;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu0 = new std::complex<float>[ N ];
    std::complex<float>* z_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(1.0 + 0.1 * i, 1.0 * i);
        y_cpu[i] = std::complex<float>(1.1f * i, 0.021f * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    std::complex<float>* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_cmin(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1, reinterpret_cast<cuComplex*>(z_gpu), 1);
    egblas_cmin(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(std::complex<float>(1.0, 0.0) * my_min(x_cpu[i], y_cpu0[i])));
        REQUIRE(z_cpu[i] == TestComplex<float>(std::complex<float>(1.0, 0.0) * my_min(x_cpu[i], y_cpu0[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu0;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/z/0", "[double][min]") {
    const size_t N = 27;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu0 = new std::complex<double>[ N ];
    std::complex<double>* z_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(1.0 + 0.1 * i, 0.1 * i);
        y_cpu[i] = std::complex<double>(1.0f * i, 0.013f * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    std::complex<double>* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zmin(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1, reinterpret_cast<cuDoubleComplex*>(z_gpu), 1);
    egblas_zmin(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(std::complex<double>(1.0, 0.0) * my_min(x_cpu[i], y_cpu0[i])));
        REQUIRE(z_cpu[i] == TestComplex<double>(std::complex<double>(1.0, 0.0) * my_min(x_cpu[i], y_cpu0[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu0;
    delete[] y_cpu;
    delete[] z_cpu;
}

TEST_CASE("min/z/1", "[double][min]") {
    const size_t N = 33;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu0 = new std::complex<double>[ N ];
    std::complex<double>* z_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(1.0 + i, -1.0 * i);
        y_cpu[i] = std::complex<double>(-1.0f * i, 2.1f * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    std::complex<double>* z_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zmin(N, make_cuDoubleComplex(1.0, 0.1), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1, reinterpret_cast<cuDoubleComplex*>(z_gpu), 1);
    egblas_zmin(N, make_cuDoubleComplex(1.0, 0.1), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(z_cpu, z_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(std::complex<double>(1.0, 0.1) * my_min(x_cpu[i], y_cpu0[i])));
        REQUIRE(z_cpu[i] == TestComplex<double>(std::complex<double>(1.0, 0.1) * my_min(x_cpu[i], y_cpu0[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));
    cuda_check(cudaFree(z_gpu));

    delete[] x_cpu;
    delete[] y_cpu0;
    delete[] y_cpu;
    delete[] z_cpu;
}
