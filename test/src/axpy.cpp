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
#include "cuda_fp16.h"

#include "egblas.hpp"
#include "test.hpp"

#include "catch.hpp"

TEST_CASE("axpy/s/0", "[float][axpy]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.1f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * i + 2.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/s/1", "[float][axpy]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/s/2", "[float][axpy]") {
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3f * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/s/3", "[float][axpy]") {
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxpy(N, 0.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.0f * i + 2.3f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/d/0", "[double][axpy]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.1f * i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxpy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * i + 2.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/d/1", "[double][axpy]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxpy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/d/2", "[double][axpy]") {
    const size_t N = 111;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 2.3f * i;
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxpy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(0.2f * i + 2.3f * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/c/0", "[float][axpy]") {
    const size_t N = 27;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i, i);
        y_cpu[i] = std::complex<float>(-2.0f * i, 0.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_caxpy(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -2.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(i +  0.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/c/1", "[float][axpy]") {
    const size_t N = 33;

    std::complex<float>* x_cpu = new std::complex<float>[ N ];
    std::complex<float>* y_cpu = new std::complex<float>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<float>(i, -1.0 * i);
        y_cpu[i] = std::complex<float>(-1.0f * i, 2.1f * i);
    }

    std::complex<float>* x_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_caxpy(N, make_cuComplex(1.0, 0.0), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -1.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(i +  0.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/z/0", "[double][axpy]") {
    const size_t N = 27;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i, i);
        y_cpu[i] = std::complex<double>(-2.0f * i, 0.1f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zaxpy(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -2.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(i +  0.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/z/1", "[double][axpy]") {
    const size_t N = 33;

    std::complex<double>* x_cpu = new std::complex<double>[ N ];
    std::complex<double>* y_cpu = new std::complex<double>[ N ];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>(i, -1.0 * i);
        y_cpu[i] = std::complex<double>(-1.0f * i, 2.1f * i);
    }

    std::complex<double>* x_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zaxpy(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i].real() == Approx(i + -1.0f * i));
        REQUIRE(y_cpu[i].imag() == Approx(-1.0 * i +  2.1f * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/i/0", "[int][axpy]") {
    const size_t N = 137;

    int32_t* x_cpu = new int32_t[N];
    int32_t* y_cpu = new int32_t[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 21 * i;
    }

    int32_t* x_gpu;
    int32_t* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(int32_t)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(int32_t)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(int32_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(int32_t), cudaMemcpyHostToDevice));

    egblas_iaxpy(N, 1, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1 * i + 21 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/i/1", "[int][axpy]") {
    const size_t N = 333;

    int32_t* x_cpu = new int32_t[N];
    int32_t* y_cpu = new int32_t[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 23 * i;
    }

    int32_t* x_gpu;
    int32_t* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(int32_t)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(int32_t)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(int32_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(int32_t), cudaMemcpyHostToDevice));

    egblas_iaxpy(N, 2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(2 * i + 23 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/i/2", "[int][axpy]") {
    const size_t N = 111;

    int32_t* x_cpu = new int32_t[N];
    int32_t* y_cpu = new int32_t[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 23 * i;
    }

    int32_t* x_gpu;
    int32_t* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(int32_t)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(int32_t)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(int32_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(int32_t), cudaMemcpyHostToDevice));

    egblas_iaxpy(N, 2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(2 * i + 23 * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(23 * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/l/0", "[long][axpy]") {
    const size_t N = 137;

    int64_t* x_cpu = new int64_t[N];
    int64_t* y_cpu = new int64_t[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 21 * i;
    }

    int64_t* x_gpu;
    int64_t* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(int64_t)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(int64_t)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(int64_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(int64_t), cudaMemcpyHostToDevice));

    egblas_laxpy(N, 1, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(int64_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1 * i + 21 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/l/1", "[long][axpy]") {
    const size_t N = 333;

    int64_t* x_cpu = new int64_t[N];
    int64_t* y_cpu = new int64_t[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 23 * i;
    }

    int64_t* x_gpu;
    int64_t* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(int64_t)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(int64_t)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(int64_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(int64_t), cudaMemcpyHostToDevice));

    egblas_laxpy(N, 2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(int64_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(2 * i + 23 * i));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/l/2", "[long][axpy]") {
    const size_t N = 111;

    int64_t* x_cpu = new int64_t[N];
    int64_t* y_cpu = new int64_t[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = i;
        y_cpu[i] = 23 * i;
    }

    int64_t* x_gpu;
    int64_t* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(int64_t)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(int64_t)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(int64_t), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(int64_t), cudaMemcpyHostToDevice));

    egblas_laxpy(N, 2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(int64_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            REQUIRE(y_cpu[i] == Approx(2 * i + 23 * i));
        } else {
            REQUIRE(y_cpu[i] == Approx(23 * i));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

#ifndef DISABLE_FP16

TEST_CASE("axpy/h/0", "[half][axpy]") {
    const size_t N = 137;

    __half2* x_cpu = new __half2[N];
    __half2* y_cpu = new __half2[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = __float2half2_rn(i);
        y_cpu[i] = __float2half2_rn(2.1f * i);
    }

    __half2* x_gpu;
    __half2* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(__half2)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(__half2)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(__half2), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(__half2), cudaMemcpyHostToDevice));

    egblas_haxpy(N, __float2half2_rn(1.0), x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(__half2), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(1.0f * i + 2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(1.0f * i + 2.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/h/1", "[half][axpy]") {
    const size_t N = 79;

    __half2* x_cpu = new __half2[N];
    __half2* y_cpu = new __half2[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = __float2half2_rn(i);
        y_cpu[i] = __float2half2_rn(3.1f * i);
    }

    __half2* x_gpu;
    __half2* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(__half2)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(__half2)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(__half2), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(__half2), cudaMemcpyHostToDevice));

    egblas_haxpy(N, __float2half2_rn(0.5), x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(__half2), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(0.5f * i + 3.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(0.5f * i + 3.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("axpy/h/2", "[half][axpy]") {
    const size_t N = 192;

    __half2* x_cpu = new __half2[N];
    __half2* y_cpu = new __half2[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = __float2half2_rn(i);
        y_cpu[i] = __float2half2_rn(-2.1f * i);
    }

    __half2* x_gpu;
    __half2* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(__half2)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(__half2)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(__half2), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(__half2), cudaMemcpyHostToDevice));

    egblas_haxpy(N, __float2half2_rn(0.0), x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(__half2), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(__high2float(y_cpu[i]) == Approx(-2.1f * i).epsilon(half_eps));
        REQUIRE(__low2float(y_cpu[i]) == Approx(-2.1f * i).epsilon(half_eps));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

#endif
