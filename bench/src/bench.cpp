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

void bench_inv_dropout(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_sinv_dropout_seed(N, 0.5f, 1.0f, x_gpu, 1, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sinv_dropout_seed(N, 0.5f, 1.0f, x_gpu, 1, 42);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);

    std::cout << "inv_dropout(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_inv_dropout(){
    bench_inv_dropout(100);
    bench_inv_dropout(1000);
    bench_inv_dropout(10000);
    bench_inv_dropout(100000);
    bench_inv_dropout(1000000);
    bench_inv_dropout(10000000);
    bench_inv_dropout(100000000);
    std::cout << std::endl;
}

void bench_saxpy(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_saxpy(N, 2.1f, x_gpu, 1, y_gpu, 1);

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
    bench_saxpy(100000000);
    std::cout << std::endl;
}

void bench_saxpby(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.2f);
    auto* y_cpu = prepare_cpu(N, 3.1f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_saxpby(N, 2.54f, x_gpu, 1, 3.49f, y_gpu, 1);

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
    bench_saxpby(10000000);
    bench_saxpby(100000000);
    std::cout << std::endl;
}

void bench_shuffle(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.2f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_shuffle_seed(N, x_gpu, 4, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_shuffle_seed(N, x_gpu, 4, 42);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);

    std::cout << "shuffle(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_shuffle(){
    bench_shuffle(100);
    bench_shuffle(1000);
    bench_shuffle(10000);
    bench_shuffle(100000);
    std::cout << std::endl;
}

void bench_par_shuffle(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.2f);
    auto* y_cpu = prepare_cpu(N, 3.1f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_par_shuffle_seed(N, x_gpu, 4, y_gpu, 4, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_par_shuffle_seed(N, x_gpu, 4, y_gpu, 4, 42);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    std::cout << "par_shuffle(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_par_shuffle(){
    bench_par_shuffle(100);
    bench_par_shuffle(1000);
    bench_par_shuffle(10000);
    bench_par_shuffle(100000);
    std::cout << std::endl;
}

void bench_big_shuffle(size_t N, size_t repeat = 100) {
    auto* x_cpu = prepare_cpu(N * 1024, 2.2f);
    auto* x_gpu = prepare_gpu(N * 1024, x_cpu);

    egblas_shuffle_seed(N, x_gpu, 4 * 1024, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_shuffle_seed(N, x_gpu, 4 * 1024, 42);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * 1024 * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);

    std::cout << "big_shuffle(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_big_shuffle(){
    bench_big_shuffle(100);
    bench_big_shuffle(1000);
    bench_big_shuffle(10000);
    std::cout << std::endl;
}

void bench_par_big_shuffle(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N * 1024, 2.2f);
    auto* y_cpu = prepare_cpu(N * 1024, 3.1f);

    auto* x_gpu = prepare_gpu(N * 1024, x_cpu);
    auto* y_gpu = prepare_gpu(N * 1024, y_cpu);

    egblas_par_shuffle_seed(N, x_gpu, 4 * 1024, y_gpu, 4, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_par_shuffle_seed(N, x_gpu, 4 * 1024, y_gpu, 4, 42);
    }

    auto t1 = timer::now();
    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * 1024 * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * 1024 * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    std::cout << "par_big_shuffle(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
        << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
}

void bench_par_big_shuffle(){
    bench_par_big_shuffle(100);
    bench_par_big_shuffle(1000);
    bench_par_big_shuffle(10000);
    std::cout << std::endl;
}

} // End of anonymous namespace

int main(){
    bench_inv_dropout();
    bench_shuffle();
    bench_par_shuffle();
    bench_big_shuffle();
    bench_par_big_shuffle();
    bench_saxpy();
    bench_saxpby();
}
