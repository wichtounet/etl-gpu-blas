//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"
#include "test.hpp"

#include "catch.hpp"

TEST_CASE( "axdy/float/0", "[float][axdy]"){
    const size_t N = 137;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for(size_t i = 0; i < N; ++i){
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.1f * (i+1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void **)&x_gpu, N*sizeof(float)));
    cuda_check(cudaMalloc((void **)&y_gpu, N*sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N*sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N*sizeof(float), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < N; ++i){
        REQUIRE(y_cpu[i] == Approx((2.1f * (i+1)) / (1.0 * (i+1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE( "axdy/float/1", "[float][axdy]"){
    const size_t N = 333;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for(size_t i = 0; i < N; ++i){
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void **)&x_gpu, N*sizeof(float)));
    cuda_check(cudaMalloc((void **)&y_gpu, N*sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N*sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N*sizeof(float), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < N; ++i){
        REQUIRE(y_cpu[i] == Approx((2.3f * (i+1)) / (0.2 * (i+1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE( "axdy/float/2", "[float][axdy]"){
    const size_t N = 111;

    float* x_cpu = new float[N];
    float* y_cpu = new float[N];

    for(size_t i = 0; i < N; ++i){
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    float* x_gpu;
    float* y_gpu;
    cuda_check(cudaMalloc((void **)&x_gpu, N*sizeof(float)));
    cuda_check(cudaMalloc((void **)&y_gpu, N*sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N*sizeof(float), cudaMemcpyHostToDevice));

    egblas_saxdy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N*sizeof(float), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < N; ++i){
        if(i % 3 == 0){
            REQUIRE(y_cpu[i] == Approx((2.3f * (i+1)) / (0.2 * (i+1))));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * (i+1)));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE( "axdy/double/0", "[double][axdy]"){
    const size_t N = 137;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for(size_t i = 0; i < N; ++i){
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.1f * (i+1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void **)&x_gpu, N*sizeof(double)));
    cuda_check(cudaMalloc((void **)&y_gpu, N*sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N*sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N*sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy(N, 1.0, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < N; ++i){
        REQUIRE(y_cpu[i] == Approx((2.1f * (i+1)) / (1.0 * (i+1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE( "axdy/double/1", "[double][axdy]"){
    const size_t N = 333;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for(size_t i = 0; i < N; ++i){
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void **)&x_gpu, N*sizeof(double)));
    cuda_check(cudaMalloc((void **)&y_gpu, N*sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N*sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N*sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy(N, 0.2, x_gpu, 1, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < N; ++i){
        REQUIRE(y_cpu[i] == Approx((2.3f * (i+1)) / (0.2 * (i+1))));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE( "axdy/double/2", "[double][axdy]"){
    const size_t N = 111;

    double* x_cpu = new double[N];
    double* y_cpu = new double[N];

    for(size_t i = 0; i < N; ++i){
        x_cpu[i] = (i+1);
        y_cpu[i] = 2.3f * (i+1);
    }

    double* x_gpu;
    double* y_gpu;
    cuda_check(cudaMalloc((void **)&x_gpu, N*sizeof(double)));
    cuda_check(cudaMalloc((void **)&y_gpu, N*sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, N*sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(y_gpu, y_cpu, N*sizeof(double), cudaMemcpyHostToDevice));

    egblas_daxdy(N, 0.2, x_gpu, 3, y_gpu, 3);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < N; ++i){
        if(i % 3 == 0){
            REQUIRE(y_cpu[i] == Approx((2.3f * (i+1)) / (0.2 * (i+1))));
        } else {
            REQUIRE(y_cpu[i] == Approx(2.3f * (i+1)));
        }
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}
