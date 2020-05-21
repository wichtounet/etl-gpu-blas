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
#include "cublas_v2.h"

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

#define cublas_check(call)                                                          \
    {                                                                               \
        auto status = call;                                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                      \
            std::cerr << "CUDA error: " << status << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;       \
        }                                                                           \
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

template<typename T>
inline void report(const std::string& name, const T& t0, size_t repeat, size_t N, bool us_unit = true){
    cudaDeviceSynchronize();

    auto t1 = timer::now();

    auto us = std::chrono::duration_cast<microseconds>(t1 - t0).count();
    auto us_avg = us / double(repeat);

    if (us_unit) {
        std::cout << name << "(" << N << "): Tot: " << us << "us Avg: " << us_avg << "us Throughput: "
            << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
    } else { // ms_unit
        auto ms = (us / 1000.0);
        auto ms_avg = ms / double(repeat);

        std::cout << name << "(" << N << "): Tot: " << ms << "ms Avg: " << ms_avg << "ms Throughput: "
            << (1e6 / double(us_avg)) * N << "E/s" << std::endl;
    }
}

void bench_sum(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_ssum(x_gpu, N, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_ssum(x_gpu, N, 1);
    }

    report("sum", t0, repeat, N, false);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
}

void bench_sum(){
    bench_sum(100);
    bench_sum(1000);
    bench_sum(10000);
    bench_sum(100000);
    bench_sum(1000000);
    bench_sum(10000000);
    bench_sum(100000000);
    std::cout << std::endl;
}

void bench_stddev(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_sstddev(x_gpu, N, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sstddev(x_gpu, N, 1);
    }

    report("stddev", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
}

void bench_stddev(){
    bench_stddev(100);
    bench_stddev(1000);
    bench_stddev(10000);
    bench_stddev(100000);
    bench_stddev(1000000);
    bench_stddev(10000000);
    bench_stddev(100000000);
    std::cout << std::endl;
}

void bench_cce_loss(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* y_cpu = prepare_cpu(N, 2.2f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_cce_sloss(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_cce_sloss(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    }

    report("cce_loss", t0, repeat, N, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_cce_loss(){
    bench_cce_loss(100);
    bench_cce_loss(1000);
    bench_cce_loss(10000);
    bench_cce_loss(100000);
    bench_cce_loss(1000000);
    bench_cce_loss(10000000);
    bench_cce_loss(100000000);
    std::cout << std::endl;
}

void bench_cce_error(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N * 10, 2.1f);
    auto* y_cpu = prepare_cpu(N * 10, 2.2f);

    auto* x_gpu = prepare_gpu(N * 10, x_cpu);
    auto* y_gpu = prepare_gpu(N * 10, y_cpu);

    egblas_cce_serror(N, 10, 1.0f/ float(N), x_gpu, y_gpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_cce_serror(N, 10, 1.0f/ float(N), x_gpu, y_gpu);
    }

    report("cce_error", t0, repeat, N * 10, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_cce_error(){
    bench_cce_error(10);
    bench_cce_error(100);
    bench_cce_error(1000);
    bench_cce_error(10000);
    bench_cce_error(100000);
    bench_cce_error(1000000);
    bench_cce_error(10000000);
    std::cout << std::endl;
}

void bench_cce(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N * 10, 2.1f);
    auto* y_cpu = prepare_cpu(N * 10, 2.2f);

    auto* x_gpu = prepare_gpu(N * 10, x_cpu);
    auto* y_gpu = prepare_gpu(N * 10, y_cpu);

    egblas_cce_sloss(10 * N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    egblas_cce_serror(N, 10, 1.0f/ float(N), x_gpu, y_gpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_cce_sloss(10 * N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
        egblas_cce_serror(N, 10, 1.0f/ float(N), x_gpu, y_gpu);
    }

    report("cce_both", t0, repeat, N * 10, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_cce(){
    bench_cce(10);
    bench_cce(100);
    bench_cce(1000);
    bench_cce(10000);
    bench_cce(100000);
    bench_cce(1000000);
    bench_cce(10000000);
    std::cout << std::endl;
}

void bench_scce(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N * 10, 2.1f);
    auto* y_cpu = prepare_cpu(N * 10, 2.2f);

    auto* x_gpu = prepare_gpu(N * 10, x_cpu);
    auto* y_gpu = prepare_gpu(N * 10, y_cpu);

    egblas_scce(N, 10, 1.0f/ float(N), 1.0f/ float(N), x_gpu, y_gpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_scce(N, 10, 1.0f / float(N), 1.0f / float(N), x_gpu, y_gpu);
    }

    report("scce", t0, repeat, N * 10, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_scce(){
    bench_scce(10);
    bench_scce(100);
    bench_scce(1000);
    bench_scce(10000);
    bench_scce(100000);
    bench_scce(1000000);
    bench_scce(10000000);
    std::cout << std::endl;
}

void bench_bce_loss(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* y_cpu = prepare_cpu(N, 2.2f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_bce_sloss(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_bce_sloss(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    }

    report("bce_loss", t0, repeat, N, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bce_loss(){
    bench_bce_loss(100);
    bench_bce_loss(1000);
    bench_bce_loss(10000);
    bench_bce_loss(100000);
    bench_bce_loss(1000000);
    bench_bce_loss(10000000);
    bench_bce_loss(100000000);
    std::cout << std::endl;
}

void bench_bce_error(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* y_cpu = prepare_cpu(N, 2.2f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_bce_serror(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_bce_serror(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    }

    report("bce_error", t0, repeat, N, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bce_error(){
    bench_bce_error(100);
    bench_bce_error(1000);
    bench_bce_error(10000);
    bench_bce_error(100000);
    bench_bce_error(1000000);
    bench_bce_error(10000000);
    bench_bce_error(100000000);
    std::cout << std::endl;
}

void bench_bce(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* y_cpu = prepare_cpu(N, 2.2f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_bce_serror(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    egblas_bce_sloss(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_bce_serror(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
        egblas_bce_sloss(N, 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    }

    report("bce_both", t0, repeat, N, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bce(){
    bench_bce(100);
    bench_bce(1000);
    bench_bce(10000);
    bench_bce(100000);
    bench_bce(1000000);
    bench_bce(10000000);
    bench_bce(100000000);
    std::cout << std::endl;
}

void bench_sbce(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* y_cpu = prepare_cpu(N, 2.2f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_sbce(N, 1.0f/ float(N), 1.0f/ float(N), x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sbce(N, 1.0f/ float(N), 1.0f/ float(N), x_gpu, 1, y_gpu, 1);
    }

    report("sbce", t0, repeat, N, false);

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_sbce(){
    bench_sbce(100);
    bench_sbce(1000);
    bench_sbce(10000);
    bench_sbce(100000);
    bench_sbce(1000000);
    bench_sbce(10000000);
    bench_sbce(100000000);
    std::cout << std::endl;
}

void bench_normalize_flat(size_t N, size_t repeat = 100) {
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_snormalize_flat(N, x_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_snormalize_flat(N, x_gpu, 1);
    }

    report("normalize_flat", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
}

void bench_normalize_flat(){
    bench_normalize_flat(100);
    bench_normalize_flat(1000);
    bench_normalize_flat(10000);
    bench_normalize_flat(100000);
    bench_normalize_flat(1000000);
    bench_normalize_flat(10000000);
    bench_normalize_flat(100000000);
    std::cout << std::endl;
}

void bench_normalize_sub(size_t N, size_t N2, size_t repeat = 100) {
    auto* x_cpu = prepare_cpu(N * N2, 2.0f);
    auto* x_gpu = prepare_gpu(N * N2, x_cpu);

    egblas_snormalize_sub(N, x_gpu, N2, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_snormalize_sub(N, x_gpu, N2, 1);
    }

    report("normalize_sub", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
}

void bench_normalize_sub(){
    bench_normalize_sub(10, 784);
    bench_normalize_sub(100, 784);
    bench_normalize_sub(1000, 784);
    bench_normalize_sub(10000, 784);
    bench_normalize_sub(100000, 784);
    bench_normalize_sub(1000000, 784);
    bench_normalize_sub(10000000, 784);
    std::cout << std::endl;
}

void bench_inv_dropout(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_sinv_dropout_seed(N, 0.5f, 1.0f, x_gpu, 1, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sinv_dropout_seed(N, 0.5f, 1.0f, x_gpu, 1, 42);
    }

    report("inv_dropout", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
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

    report("saxpy", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
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

void bench_cublas_saxpy(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 2.1f;

    cublas_check(cublasSaxpy(handle, N, &alpha, x_gpu, 1, y_gpu, 1));

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        cublas_check(cublasSaxpy(handle, N, &alpha, x_gpu, 1, y_gpu, 1));
    }

    report("cublas_saxpy", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    cublasDestroy(handle);
}

void bench_cublas_saxpy(){
    bench_cublas_saxpy(100);
    bench_cublas_saxpy(1000);
    bench_cublas_saxpy(10000);
    bench_cublas_saxpy(100000);
    bench_cublas_saxpy(1000000);
    bench_cublas_saxpy(10000000);
    bench_cublas_saxpy(100000000);
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

    report("saxpby", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
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

    report("shuffle", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
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

    report("par_shuffle", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
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

    report("big_shuffle", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * 1024 * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
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

    report("par_big_shuffle", t0, repeat, N);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * 1024 * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * 1024 * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_par_big_shuffle(){
    bench_par_big_shuffle(100);
    bench_par_big_shuffle(1000);
    bench_par_big_shuffle(10000);
    std::cout << std::endl;
}

} // End of anonymous namespace

int main(int argc, char* argv[]){
    std::string sub = "all";

    if(argc > 1){
        sub = std::string(argv[1]);
    }

    if (sub == "sum" || sub == "all") {
        bench_sum();
    }

    if (sub == "stddev" || sub == "all") {
        bench_stddev();
    }

    if (sub == "cce" || sub == "all") {
        bench_cce_loss();
        bench_cce_error();
        bench_cce();
        bench_scce();
    }

    if (sub == "bce" || sub == "all") {
        bench_bce_loss();
        bench_bce_error();
        bench_bce();
        bench_sbce();
    }

    if (sub == "normalize" || sub == "all") {
        bench_normalize_flat();
        bench_normalize_sub();
    }

    if (sub == "dropout" || sub == "all") {
        bench_inv_dropout();
    }

    if (sub == "shuffle" || sub == "all") {
        bench_shuffle();
        bench_par_shuffle();
        bench_big_shuffle();
        bench_par_big_shuffle();
    }

    if (sub == "axpy" || sub == "all") {
        bench_saxpy();
        bench_cublas_saxpy();
        bench_saxpby();
    }
}
