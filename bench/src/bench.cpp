//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <chrono>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"

#define cuda_check(call)                                                                                \
    {                                                                                                   \
        auto status = call;                                                                             \
        if (status != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                           \
            exit(1);                                                                                    \
        }                                                                                               \
    }

namespace {

using timer = std::chrono::high_resolution_clock;
using microseconds =  std::chrono::microseconds;

float* prepare_cpu(size_t N, float s){
    float* x_cpu = new float[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = s * (i + 1);
    }

    return x_cpu;
}

float* prepare_gpu(size_t N, float* x_cpu){
    float* x_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(float)));
    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    return x_gpu;
}

void release(float* x_cpu, float* x_gpu){
    delete[] x_cpu;

    cuda_check(cudaFree(x_gpu));
}

void bench_saxpy(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_saxpy(N, 2.1f, x_gpu, 1, y_gpu, 1);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    std::cout << "saxpy(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_saxpy(){
    bench_saxpy(100);
    bench_saxpy(1000);
    bench_saxpy(10000);
    bench_saxpy(100000);
    bench_saxpy(1000000);
    bench_saxpy(10000000);
    std::cout << std::endl;
}

void bench_saxpby(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.2f);
    auto* y_cpu = prepare_cpu(N, 3.1f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_saxpby(N, 2.54f, x_gpu, 1, 3.49f, y_gpu, 1);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    std::cout << "saxpby(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_saxpby(){
    bench_saxpby(100);
    bench_saxpby(1000);
    bench_saxpby(10000);
    bench_saxpby(100000);
    bench_saxpby(1000000);
    bench_saxpy(10000000);
    std::cout << std::endl;
}

} // End of anonymous namespace

int main(){
    bench_saxpy();
    bench_saxpby();
}
