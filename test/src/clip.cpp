//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

namespace {

template <typename T>
T clip(T x, T min, T max){
    if(x < min){
        return min;
    } else if(x > max){
        return max;
    } else {
        return x;
    }
}

template <>
std::complex<float> clip(std::complex<float> x, std::complex<float> min, std::complex<float> max){
    if(x.real() < min.real() || (x.real() == min.real() && x.imag() < min.imag())){
        return min;
    } else if(x.real() > max.real() || (x.real() == max.real() && x.imag() > max.imag())){
        return max;
    } else {
        return x;
    }
}

template <>
std::complex<double> clip(std::complex<double> x, std::complex<double> min, std::complex<double> max){
    if(x.real() < min.real() || (x.real() == min.real() && x.imag() < min.imag())){
        return min;
    } else if(x.real() > max.real() || (x.real() == max.real() && x.imag() > max.imag())){
        return max;
    } else {
        return x;
    }
}

} // end of anonymous namespace

TEST_CASE("clip/s/0", "[float][clip]") {
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* z_cpu = new float[N];
    float* y_cpu = new float[N];
    float* y_cpu0 = new float[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = 0.5 * (i + 1);
        y_cpu0[i] = y_cpu[i];
        x_cpu[i] = -9.0 + i + 1;
        z_cpu[i] = 12.0 + (i + 1);
    }

    float* x_gpu;
    float* z_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sclip(N, 1.0, x_gpu, 1, z_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/s/1", "[float][clip]") {
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* z_cpu = new float[N];
    float* y_cpu = new float[N];
    float* y_cpu0 = new float[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = 0.7f * (i + 1);
        y_cpu0[i] = y_cpu[i];
        x_cpu[i] = -12.5f + i + 1;
        z_cpu[i] = 1.24f * (25.0f + (i + 1));
    }

    float* x_gpu;
    float* z_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sclip(N, 0.3f, x_gpu, 1, z_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.3f * clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/d/0", "[double][clip]") {
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* z_cpu = new double[N];
    double* y_cpu = new double[N];
    double* y_cpu0 = new double[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = 0.24 * (i + 1);
        x_cpu[i] = -12.0 + i + 1;
        z_cpu[i] = 13.33 + i + 1 - 100.00;
        y_cpu0[i] = y_cpu[i];
    }

    double* x_gpu;
    double* z_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dclip(N, 1.0, x_gpu, 1, z_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0 * clip(y_cpu0[i], x_cpu[i], y_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/d/1", "[double][clip]") {
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* z_cpu = new double[N];
    double* y_cpu = new double[N];
    double* y_cpu0 = new double[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = 0.24 * ((i - 10) * i);
        x_cpu[i] = -12.0 + i + 1;
        z_cpu[i] = 13.33 + i + 1 - 100.00;
        y_cpu0[i] = y_cpu[i];
    }

    double* x_gpu;
    double* z_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dclip(N, 0.2, x_gpu, 1, z_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.2 * clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/c/0", "[float][clip]") {
    const size_t N = 137;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    std::complex<float>* z_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu0 = new std::complex<float>[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = std::complex<float>(0.8f + i / 1000.0f, 1.01f * i);
        x_cpu[i] = std::complex<float>(-2.0f + i / 1000.0f, 1.02f * i);
        z_cpu[i] = std::complex<float>(1.0f + i / 1000.0f, 1.03f * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<float>* x_gpu;
    std::complex<float>* z_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_cclip(N, make_cuComplex(1.0f, 0.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(z_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/c/1", "[float][clip]") {
    const size_t N = 338;

    std::complex<float>* x_cpu = new std::complex<float>[N];
    std::complex<float>* z_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu = new std::complex<float>[N];
    std::complex<float>* y_cpu0 = new std::complex<float>[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = std::complex<float>(0.5 + i / 900.0f, -2.0f * i);
        x_cpu[i] = std::complex<float>(-0.5 + i / 998.0f, -2.0f * i);
        z_cpu[i] = std::complex<float>(1.5 + i / 996.0f, -2.0f * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<float>* x_gpu;
    std::complex<float>* z_gpu;
    std::complex<float>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<float>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<float>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<float>), cudaMemcpyHostToDevice));

    egblas_cclip(N, make_cuComplex(1.0f, 1.0f), reinterpret_cast<cuComplex*>(x_gpu), 1, reinterpret_cast<cuComplex*>(z_gpu), 1, reinterpret_cast<cuComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<float>(std::complex<float>(1.0f, 1.0f) * clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/z/0", "[double][clip]") {
    const size_t N = 137;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    std::complex<double>* z_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu0 = new std::complex<double>[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = std::complex<double>((i-1) / 900.0, -1.1 * i);
        z_cpu[i] = std::complex<double>((i+1) / 999.0, -1.1 * i);
        y_cpu[i] = std::complex<double>((i+3) / 999.0, -1.1 * i);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<double>* x_gpu;
    std::complex<double>* z_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zclip(N, make_cuDoubleComplex(1.0, 0.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(z_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip/z/1", "[double][clip]") {
    const size_t N = 338;

    std::complex<double>* x_cpu = new std::complex<double>[N];
    std::complex<double>* z_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu = new std::complex<double>[N];
    std::complex<double>* y_cpu0 = new std::complex<double>[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = std::complex<double>(i / 996.0, 50);
        x_cpu[i] = std::complex<double>(i / 996.0, 2.4 * i);
        z_cpu[i] = std::complex<double>(i / 996.0, 100.0);
        y_cpu0[i] = y_cpu[i];
    }

    std::complex<double>* x_gpu;
    std::complex<double>* z_gpu;
    std::complex<double>* y_gpu;
    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&z_gpu, N * sizeof(std::complex<double>)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(std::complex<double>)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(z_gpu, z_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

    egblas_zclip(N, make_cuDoubleComplex(0.1, 2.0), reinterpret_cast<cuDoubleComplex*>(x_gpu), 1, reinterpret_cast<cuDoubleComplex*>(z_gpu), 1, reinterpret_cast<cuDoubleComplex*>(y_gpu), 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == TestComplex<double>(std::complex<double>(0.1, 2.0) * clip(y_cpu0[i], x_cpu[i], z_cpu[i])));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(z_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] z_cpu;
    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip_value/s/0", "[float][clip]") {
    const size_t N = 137;

    float* y_cpu = new float[N];
    float* y_cpu0 = new float[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = 0.5 * (i + 1);
        y_cpu0[i] = y_cpu[i];
    }

    float* y_gpu;
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sclip_value(N, 1.0f, 9.0f, 16.0f, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(1.0f * clip(y_cpu0[i], 9.0f, 16.0f)));
    }

    cuda_check(cudaFree(y_gpu));

    delete[] y_cpu;
    delete[] y_cpu0;
}

TEST_CASE("clip_value/d/0", "[float][clip]") {
    const size_t N = 137;

    double* y_cpu = new double[N];
    double* y_cpu0 = new double[N];

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = 0.5 * (i + 1);
        y_cpu0[i] = y_cpu[i];
    }

    double* y_gpu;
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(y_gpu, y_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dclip_value(N, 0.5, 4.5, 22.4, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(0.5 * clip(y_cpu0[i], 4.5, 22.4)));
    }

    cuda_check(cudaFree(y_gpu));

    delete[] y_cpu;
    delete[] y_cpu0;
}
