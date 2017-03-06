#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE( "sum/float/0", "[float][sum]"){
    const size_t N = 123;

    float* cpu_vec = new float[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = i;
    }

    float* gpu_vec;
    cudaMalloc((void **)&gpu_vec, N*sizeof(float));

    cudaMemcpy(cpu_vec, gpu_vec, N*sizeof(float), cudaMemcpyHostToDevice);

    REQUIRE(egblas_ssum(cpu_vec, N, 1) == Approx(float(7503)));

    cudaFree(gpu_vec);

    delete[] cpu_vec;
}

TEST_CASE( "sum/float/1", "[float][sum]"){
    const size_t N = 389;

    float* cpu_vec = new float[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    float* gpu_vec;
    cudaMalloc((void **)&gpu_vec, N*sizeof(float));

    cudaMemcpy(cpu_vec, gpu_vec, N*sizeof(float), cudaMemcpyHostToDevice);

    REQUIRE(egblas_ssum(cpu_vec, N, 1) == Approx(float(116972)));

    cudaFree(gpu_vec);

    delete[] cpu_vec;
}

TEST_CASE( "sum/double/0", "[double][sum]"){
    const size_t N = 123;

    double* cpu_vec = new double[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = i;
    }

    double* gpu_vec;
    cudaMalloc((void **)&gpu_vec, N*sizeof(double));

    cudaMemcpy(cpu_vec, gpu_vec, N*sizeof(double), cudaMemcpyHostToDevice);

    REQUIRE(egblas_dsum(cpu_vec, N, 1) == Approx(double(7503)));

    cudaFree(gpu_vec);

    delete[] cpu_vec;
}

TEST_CASE( "sum/double/1", "[double][sum]"){
    const size_t N = 389;

    double* cpu_vec = new double[N];

    for(size_t i = 0; i < N; ++i){
        cpu_vec[i] = (3.1 * i) / 2.0;
    }

    double* gpu_vec;
    cudaMalloc((void **)&gpu_vec, N*sizeof(double));

    cudaMemcpy(cpu_vec, gpu_vec, N*sizeof(double), cudaMemcpyHostToDevice);

    REQUIRE(egblas_dsum(cpu_vec, N, 1) == Approx(double(116972)));

    cudaFree(gpu_vec);

    delete[] cpu_vec;
}
