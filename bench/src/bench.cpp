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

#if __has_include("cudnn.h")
#define ENABLE_CUDNN
#include "cudnn.h"
#endif

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

#define cudnn_check(call)                                                                                 \
    {                                                                                                     \
        cudnnStatus_t status = call;                                                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                                                             \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                             \
        }                                                                                                 \
    }

namespace {

using timer = std::chrono::high_resolution_clock;
using microseconds =  std::chrono::microseconds;

template <typename T = float>
T* prepare_cpu(size_t N, T s){
    T* x_cpu = new T[N];

    for (size_t i = 0; i < N; ++i) {
        x_cpu[i] = s * (i + 1);
    }

    return x_cpu;
}

template <typename T = float>
T* prepare_gpu(size_t N, T* x_cpu){
    T* x_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, N * sizeof(T)));
    cuda_check(cudaMemcpy(x_gpu, x_cpu, N * sizeof(T), cudaMemcpyHostToDevice));

    return x_gpu;
}

template <typename T>
void release(T* x_cpu, T* x_gpu){
    delete[] x_cpu;

    cuda_check(cudaFree(x_gpu));
}

template<typename T>
inline void report(std::string_view name, const T& t0, size_t repeat, size_t N, bool us_unit = true){
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

void bench_asum(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_sasum(x_gpu, N, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sasum(x_gpu, N, 1);
    }

    report("asum", t0, repeat, N, false);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
}

void bench_asum(){
    bench_asum(100);
    bench_asum(1000);
    bench_asum(10000);
    bench_asum(100000);
    bench_asum(1000000);
    bench_asum(10000000);
    bench_asum(100000000);
    std::cout << std::endl;
}

void bench_cublas_asum(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float prod = 0.1f;
    cublas_check(cublasSasum(handle, N, x_gpu, 1, &prod));

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        cublas_check(cublasSasum(handle, N, x_gpu, 1, &prod));
    }

    report("cublas_asum", t0, repeat, N, false);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);

    cublasDestroy(handle);
}

void bench_cublas_asum(){
    bench_cublas_asum(100);
    bench_cublas_asum(1000);
    bench_cublas_asum(10000);
    bench_cublas_asum(100000);
    bench_cublas_asum(1000000);
    bench_cublas_asum(10000000);
    bench_cublas_asum(100000000);
    std::cout << std::endl;
}

void bench_max(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.1f);
    auto* x_gpu = prepare_gpu(N, x_cpu);

    egblas_smax(x_gpu, N, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_smax(x_gpu, N, 1);
    }

    report("max", t0, repeat, N, false);

    cuda_check(cudaMemcpy(x_cpu, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
}

void bench_max(){
    bench_max(100);
    bench_max(1000);
    bench_max(10000);
    bench_max(100000);
    bench_max(1000000);
    bench_max(10000000);
    bench_max(100000000);
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

template <typename T>
void bench_axpy(std::string_view name, size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu<T>(N, T(2));
    auto* y_cpu = prepare_cpu<T>(N, T(3));

    auto* x_gpu = prepare_gpu<T>(N, x_cpu);
    auto* y_gpu = prepare_gpu<T>(N, y_cpu);

    if constexpr (std::is_same_v<T, float>) {
        egblas_saxpy(N, 2.1f, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, double>) {
        egblas_daxpy(N, 2.1, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        egblas_oaxpy(N, 2, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        egblas_waxpy(N, 2, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        egblas_iaxpy(N, 2, x_gpu, 1, y_gpu, 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        egblas_laxpy(N, 2, x_gpu, 1, y_gpu, 1);
    }

    auto t0 = timer::now();

    for (size_t i = 0; i < repeat; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            egblas_saxpy(N, 2.1f, x_gpu, 1, y_gpu, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            egblas_daxpy(N, 2.1, x_gpu, 1, y_gpu, 1);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            egblas_oaxpy(N, 2, x_gpu, 1, y_gpu, 1);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            egblas_waxpy(N, 2, x_gpu, 1, y_gpu, 1);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            egblas_iaxpy(N, 2, x_gpu, 1, y_gpu, 1);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            egblas_laxpy(N, 2, x_gpu, 1, y_gpu, 1);
        }
    }

    report(name, t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(T), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

template <typename T>
void bench_axpy(std::string_view name){
    bench_axpy<T>(name, 100);
    bench_axpy<T>(name, 1000);
    bench_axpy<T>(name, 10000);
    bench_axpy<T>(name, 100000);
    bench_axpy<T>(name, 1000000);
    bench_axpy<T>(name, 10000000);
    bench_axpy<T>(name, 100000000);
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

void bench_sigmoid(float alpha, size_t N,size_t repeat = 10){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_ssigmoid(N, alpha, x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_ssigmoid(N, alpha, x_gpu, 1, y_gpu, 1);
    }

    report("sigmoid", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_sigmoid(float alpha){
    bench_sigmoid(alpha, 100);
    bench_sigmoid(alpha, 1000);
    bench_sigmoid(alpha, 10000);
    bench_sigmoid(alpha, 100000);
    bench_sigmoid(alpha, 1000000);
    bench_sigmoid(alpha, 10000000);
    bench_sigmoid(alpha, 100000000);
    std::cout << std::endl;
}

#ifdef ENABLE_CUDNN

void bench_cudnn_sigmoid_lazy(float alpha, size_t N,size_t repeat = 10){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t x_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&x_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(x_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    cudnnTensorDescriptor_t y_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&y_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(y_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    cudnnActivationDescriptor_t func_tensor;
    cudnn_check(cudnnCreateActivationDescriptor(&func_tensor));
    cudnn_check(cudnnSetActivationDescriptor(func_tensor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

    float beta = 0.0f;

    cudnn_check(cudnnActivationForward(handle, func_tensor, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        cudnn_check(cudnnActivationForward(handle, func_tensor, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));
    }

    report("cudnn_sigmoid_lazy", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudnn_check(cudnnDestroyTensorDescriptor(x_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(y_tensor));
    cudnn_check(cudnnDestroyActivationDescriptor(func_tensor));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    cudnnDestroy(handle);
}

void bench_cudnn_sigmoid_lazy(float alpha){
    bench_cudnn_sigmoid_lazy(alpha, 100);
    bench_cudnn_sigmoid_lazy(alpha, 1000);
    bench_cudnn_sigmoid_lazy(alpha, 10000);
    bench_cudnn_sigmoid_lazy(alpha, 100000);
    bench_cudnn_sigmoid_lazy(alpha, 1000000);
    bench_cudnn_sigmoid_lazy(alpha, 10000000);
    bench_cudnn_sigmoid_lazy(alpha, 100000000);
    std::cout << std::endl;
}

void bench_cudnn_sigmoid(float alpha, size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t x_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&x_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(x_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    cudnnTensorDescriptor_t y_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&y_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(y_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    cudnnActivationDescriptor_t func_tensor;
    cudnn_check(cudnnCreateActivationDescriptor(&func_tensor));
    cudnn_check(cudnnSetActivationDescriptor(func_tensor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

    float beta = 0.0f;

    cudnn_check(cudnnActivationForward(handle, func_tensor, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        cudnnTensorDescriptor_t x_tensor;
        cudnn_check(cudnnCreateTensorDescriptor(&x_tensor));
        cudnn_check(cudnnSetTensor4dDescriptor(x_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

        cudnnTensorDescriptor_t y_tensor;
        cudnn_check(cudnnCreateTensorDescriptor(&y_tensor));
        cudnn_check(cudnnSetTensor4dDescriptor(y_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

        cudnnActivationDescriptor_t func_tensor;
        cudnn_check(cudnnCreateActivationDescriptor(&func_tensor));
        cudnn_check(cudnnSetActivationDescriptor(func_tensor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

        cudnn_check(cudnnActivationForward(handle, func_tensor, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));

        cudnn_check(cudnnDestroyTensorDescriptor(x_tensor));
        cudnn_check(cudnnDestroyTensorDescriptor(y_tensor));
        cudnn_check(cudnnDestroyActivationDescriptor(func_tensor));
    }

    report("cudnn_sigmoid", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    cudnn_check(cudnnDestroyTensorDescriptor(x_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(y_tensor));
    cudnn_check(cudnnDestroyActivationDescriptor(func_tensor));

    cudnnDestroy(handle);
}

void bench_cudnn_sigmoid(float alpha){
    bench_cudnn_sigmoid(alpha, 100);
    bench_cudnn_sigmoid(alpha, 1000);
    bench_cudnn_sigmoid(alpha, 10000);
    bench_cudnn_sigmoid(alpha, 100000);
    bench_cudnn_sigmoid(alpha, 1000000);
    bench_cudnn_sigmoid(alpha, 10000000);
    bench_cudnn_sigmoid(alpha, 100000000);
    std::cout << std::endl;
}

#endif

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

void bench_saxmy_3(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.2f);
    auto* y_cpu = prepare_cpu(N, 3.1f);
    auto* yy_cpu = prepare_cpu(N, 3.4f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);
    auto* yy_gpu = prepare_gpu(N, yy_cpu);

    egblas_saxmy_3(N, 1.0f, x_gpu, 1, y_gpu, 1, yy_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_saxmy_3(N, 1.0f, x_gpu, 1, y_gpu, 1, yy_gpu, 1);
    }

    report("saxmy_3", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
    release(yy_cpu, yy_gpu);
}

void bench_saxmy_3(){
    bench_saxmy_3(100);
    bench_saxmy_3(1000);
    bench_saxmy_3(10000);
    bench_saxmy_3(100000);
    bench_saxmy_3(1000000);
    bench_saxmy_3(10000000);
    bench_saxmy_3(100000000);
    std::cout << std::endl;
}

void bench_sqrt(size_t N,size_t repeat = 100){
    auto* x_cpu = prepare_cpu(N, 2.2f);
    auto* y_cpu = prepare_cpu(N, 3.1f);

    auto* x_gpu = prepare_gpu(N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_ssqrt(N, 1.0f, x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_ssqrt(N, 1.0f, x_gpu, 1, y_gpu, 1);
    }

    report("sqrt", t0, repeat, N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_sqrt(){
    bench_sqrt(100);
    bench_sqrt(1000);
    bench_sqrt(10000);
    bench_sqrt(100000);
    bench_sqrt(1000000);
    bench_sqrt(10000000);
    bench_sqrt(100000000);
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
    bench_shuffle(1000000);
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
    bench_par_shuffle(1000000);
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

    egblas_par_shuffle_seed(N, x_gpu, 4 * 1024, y_gpu, 8, 42);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_par_shuffle_seed(N, x_gpu, 4 * 1024, y_gpu, 8, 42);
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

#ifdef ENABLE_CUDNN

void bench_cudnn_bias_batch_sum(size_t B, size_t N, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(B * N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t x_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&x_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(x_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, N, 1, 1));

    cudnnTensorDescriptor_t y_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&y_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(y_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, 1, 1));

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnn_check(cudnnConvolutionBackwardBias(handle, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        cudnn_check(cudnnConvolutionBackwardBias(handle, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));
    }

    report("cudnn_bias_batch_sum", t0, repeat, B * N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    cudnn_check(cudnnDestroyTensorDescriptor(x_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(y_tensor));

    cudnnDestroy(handle);
}

void bench_cudnn_bias_batch_sum(){
    bench_cudnn_bias_batch_sum(256, 10);
    bench_cudnn_bias_batch_sum(256, 100);
    bench_cudnn_bias_batch_sum(256, 1000);
    bench_cudnn_bias_batch_sum(256, 10000);
    std::cout << std::endl;
}

#endif

void bench_bias_batch_sum(size_t B, size_t N, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(B * N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_sbias_batch_sum(B, N, x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sbias_batch_sum(B, N, x_gpu, 1, y_gpu, 1);
    }

    report("bias_batch_sum", t0, repeat, B * N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bias_batch_sum(){
    bench_bias_batch_sum(256, 10);
    bench_bias_batch_sum(256, 100);
    bench_bias_batch_sum(256, 1000);
    bench_bias_batch_sum(256, 10000);
    std::cout << std::endl;
}

void bench_bias_batch_mean(size_t B, size_t N, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(B * N, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_sbias_batch_mean(B, N, x_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sbias_batch_mean(B, N, x_gpu, 1, y_gpu, 1);
    }

    report("bias_batch_mean", t0, repeat, B * N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bias_batch_mean(){
    bench_bias_batch_mean(256, 10);
    bench_bias_batch_mean(256, 100);
    bench_bias_batch_mean(256, 1000);
    bench_bias_batch_mean(256, 10000);
    std::cout << std::endl;
}

#ifdef ENABLE_CUDNN

void bench_cudnn_bias_batch_sum4(size_t B, size_t N, size_t W, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N * W * W, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(B * N * W * W, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t x_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&x_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(x_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, N, W, W));

    cudnnTensorDescriptor_t y_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&y_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(y_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, 1, 1));

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnn_check(cudnnConvolutionBackwardBias(handle, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        cudnn_check(cudnnConvolutionBackwardBias(handle, &alpha, x_tensor, x_gpu, &beta, y_tensor, y_gpu));
    }

    report("cudnn_bias_batch_sum4", t0, repeat, B * N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);

    cudnn_check(cudnnDestroyTensorDescriptor(x_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(y_tensor));

    cudnnDestroy(handle);
}

void bench_cudnn_bias_batch_sum4(){
    std::cout << "8x8" << std::endl;
    bench_cudnn_bias_batch_sum4(256, 16, 8);
    bench_cudnn_bias_batch_sum4(256, 32, 8);
    bench_cudnn_bias_batch_sum4(256, 64, 8);
    bench_cudnn_bias_batch_sum4(256, 128, 8);
    std::cout << "16x16" << std::endl;
    bench_cudnn_bias_batch_sum4(256, 16, 16);
    bench_cudnn_bias_batch_sum4(256, 32, 16);
    bench_cudnn_bias_batch_sum4(256, 64, 16);
    bench_cudnn_bias_batch_sum4(256, 128, 16);
    std::cout << "28x28" << std::endl;
    bench_cudnn_bias_batch_sum4(256, 16, 28);
    bench_cudnn_bias_batch_sum4(256, 32, 28);
    bench_cudnn_bias_batch_sum4(256, 64, 28);
    bench_cudnn_bias_batch_sum4(256, 128, 28);
    std::cout << "256 32x32" << std::endl;
    bench_cudnn_bias_batch_sum4(256, 16, 32);
    bench_cudnn_bias_batch_sum4(256, 32, 32);
    bench_cudnn_bias_batch_sum4(256, 64, 32);
    bench_cudnn_bias_batch_sum4(256, 128, 32);
    std::cout << "300 32x32" << std::endl;
    bench_cudnn_bias_batch_sum4(300, 16, 32);
    bench_cudnn_bias_batch_sum4(300, 32, 32);
    bench_cudnn_bias_batch_sum4(300, 64, 32);
    bench_cudnn_bias_batch_sum4(300, 128, 32);
    std::cout << "300 64x64" << std::endl;
    bench_cudnn_bias_batch_sum4(300, 16, 64);
    bench_cudnn_bias_batch_sum4(300, 32, 64);
    bench_cudnn_bias_batch_sum4(300, 64, 64);
    bench_cudnn_bias_batch_sum4(300, 128, 64);
    std::cout << "300 128x128" << std::endl;
    bench_cudnn_bias_batch_sum4(300, 3, 128);
    bench_cudnn_bias_batch_sum4(300, 8, 128);
    bench_cudnn_bias_batch_sum4(300, 16, 128);
    std::cout << "300 256x256" << std::endl;
    bench_cudnn_bias_batch_sum4(300, 3, 256);
    bench_cudnn_bias_batch_sum4(300, 8, 256);
    bench_cudnn_bias_batch_sum4(300, 16, 256);
    std::cout << std::endl;
}

#endif

void bench_bias_batch_sum4(size_t B, size_t N, size_t W, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N * W * W, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(B * N * W * W, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_sbias_batch_sum4(B, N, W, W, x_gpu, y_gpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sbias_batch_sum4(B, N, W, W, x_gpu, y_gpu);
    }

    report("bias_batch_sum4", t0, repeat, B * N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bias_batch_sum4(){
    std::cout << "8x8" << std::endl;
    bench_bias_batch_sum4(256, 16, 8);
    bench_bias_batch_sum4(256, 32, 8);
    bench_bias_batch_sum4(256, 64, 8);
    bench_bias_batch_sum4(256, 128, 8);
    std::cout << "16x16" << std::endl;
    bench_bias_batch_sum4(256, 16, 16);
    bench_bias_batch_sum4(256, 32, 16);
    bench_bias_batch_sum4(256, 64, 16);
    bench_bias_batch_sum4(256, 128, 16);
    std::cout << "28x28" << std::endl;
    bench_bias_batch_sum4(256, 16, 28);
    bench_bias_batch_sum4(256, 32, 28);
    bench_bias_batch_sum4(256, 64, 28);
    bench_bias_batch_sum4(256, 128, 28);
    std::cout << "256 32x32" << std::endl;
    bench_bias_batch_sum4(256, 16, 32);
    bench_bias_batch_sum4(256, 32, 32);
    bench_bias_batch_sum4(256, 64, 32);
    bench_bias_batch_sum4(256, 128, 32);
    std::cout << "300 32x32" << std::endl;
    bench_bias_batch_sum4(300, 16, 32);
    bench_bias_batch_sum4(300, 32, 32);
    bench_bias_batch_sum4(300, 64, 32);
    bench_bias_batch_sum4(300, 128, 32);
    std::cout << "300 64x64" << std::endl;
    bench_bias_batch_sum4(300, 16, 64);
    bench_bias_batch_sum4(300, 32, 64);
    bench_bias_batch_sum4(300, 64, 64);
    bench_bias_batch_sum4(300, 128, 64);
    std::cout << "300 128x128" << std::endl;
    bench_bias_batch_sum4(300, 3, 128);
    bench_bias_batch_sum4(300, 8, 128);
    bench_bias_batch_sum4(300, 16, 128);
    std::cout << "300 256x256" << std::endl;
    bench_bias_batch_sum4(300, 3, 256);
    bench_bias_batch_sum4(300, 8, 256);
    bench_bias_batch_sum4(300, 16, 256);
    std::cout << std::endl;
}

void bench_bias_batch_mean4(size_t B, size_t N, size_t W, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N * W * W, 2.0f);
    auto* y_cpu = prepare_cpu(N, 3.0f);

    auto* x_gpu = prepare_gpu(B * N * W * W, x_cpu);
    auto* y_gpu = prepare_gpu(N, y_cpu);

    egblas_sbias_batch_mean4(B, N, W, W, x_gpu, y_gpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sbias_batch_mean4(B, N, W, W, x_gpu, y_gpu);
    }

    report("bias_batch_mean4", t0, repeat, B * N);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_bias_batch_mean4(){
    std::cout << "16x16" << std::endl;
    bench_bias_batch_mean4(256, 16, 16);
    bench_bias_batch_mean4(256, 32, 16);
    bench_bias_batch_mean4(256, 64, 16);
    bench_bias_batch_mean4(256, 128, 16);
    std::cout << "28x28" << std::endl;
    bench_bias_batch_mean4(256, 16, 28);
    bench_bias_batch_mean4(256, 32, 28);
    bench_bias_batch_mean4(256, 64, 28);
    bench_bias_batch_mean4(256, 128, 28);
    std::cout << "32x32" << std::endl;
    bench_bias_batch_mean4(256, 16, 32);
    bench_bias_batch_mean4(256, 32, 32);
    bench_bias_batch_mean4(256, 64, 32);
    bench_bias_batch_mean4(256, 128, 32);
    std::cout << std::endl;
}

void bench_bias_add4(size_t B, size_t N, size_t W, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(B * N * W * W, 2.0f);
    auto* y_cpu = prepare_cpu(B * N * W * W, 3.0f);
    auto* b_cpu = prepare_cpu(N, 4.0f);

    auto* x_gpu = prepare_gpu(B * N * W * W, x_cpu);
    auto* y_gpu = prepare_gpu(B * N * W * W, x_cpu);
    auto* b_gpu = prepare_gpu(N, b_cpu);

    egblas_sbias_add_4d(B, N, W, W, x_gpu, 1, b_gpu, 1, y_gpu, 1);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_sbias_add_4d(B, N, W, W, x_gpu, 1, b_gpu, 1, y_gpu, 1);
    }

    report("bias_add4", t0, repeat, B * N * W * W);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
    release(b_cpu, b_gpu);
}

void bench_bias_add4(){
    bench_bias_add4(200, 15, 14);
    bench_bias_add4(256, 16, 16);
    bench_bias_add4(256, 32, 16);
    bench_bias_add4(256, 64, 16);
    bench_bias_add4(256, 128, 16);
    std::cout << std::endl;
}

void bench_transpose_front(size_t M, size_t N, size_t K, size_t repeat = 100){
    auto* x_cpu = prepare_cpu(M * N * K, 2.0f);
    auto* y_cpu = prepare_cpu(M * N * K, 3.0f);

    auto* x_gpu = prepare_gpu(M * N * K, x_cpu);
    auto* y_gpu = prepare_gpu(M * N * K, x_cpu);

    egblas_stranspose_front(M, N, K, x_gpu, y_gpu);

    auto t0 = timer::now();

    for(size_t i = 0; i < repeat; ++i){
        egblas_stranspose_front(M, N, K, x_gpu, y_gpu);
    }

    report("transpose_front", t0, repeat, M * N * K);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, M * N * K * sizeof(float), cudaMemcpyDeviceToHost));

    release(x_cpu, x_gpu);
    release(y_cpu, y_gpu);
}

void bench_transpose_front(){
    bench_transpose_front(200, 16, 16);
    bench_transpose_front(256, 16, 32);
    bench_transpose_front(256, 28, 28);
    bench_transpose_front(256, 28, 32);
    bench_transpose_front(256, 32, 64);
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
        bench_asum();
        bench_cublas_asum();
        bench_max();
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

    if (sub == "sigmoid" || sub == "all") {
        std::cout << "alpha=1.0f" << std::endl;
        bench_sigmoid(1.0f);
#ifdef ENABLE_CUDNN
        bench_cudnn_sigmoid(1.0f);
        bench_cudnn_sigmoid_lazy(1.0f);
#endif
        std::cout << "alpha=2.17f" << std::endl;
        bench_sigmoid(2.17f);
#ifdef ENABLE_CUDNN
        bench_cudnn_sigmoid(2.17f);
        bench_cudnn_sigmoid_lazy(2.17f);
#endif
    }

    if (sub == "shuffle" || sub == "all") {
        bench_shuffle();
        bench_par_shuffle();
        bench_big_shuffle();
        bench_par_big_shuffle();
    }

    if (sub == "axpy" || sub == "all") {
        bench_axpy<int8_t>("oaxpy");
        bench_axpy<int16_t>("waxpy");
        bench_axpy<int32_t>("iaxpy");
        bench_axpy<int64_t>("laxpy");
        bench_axpy<float>("saxpy");
        bench_axpy<double>("daxpy");
        bench_cublas_saxpy();
        bench_saxpby();
        bench_saxmy_3();
    }

    if (sub == "sqrt" || sub == "all") {
        bench_sqrt();
    }

    if (sub == "bias_batch") {
        bench_bias_batch_sum();
        bench_bias_batch_mean();
#ifdef ENABLE_CUDNN
        bench_cudnn_bias_batch_sum();
#endif
    }

    if (sub == "bias_batch4") {
        bench_bias_batch_sum4();
        //bench_bias_batch_mean4();
#ifdef ENABLE_CUDNN
        bench_cudnn_bias_batch_sum4();
#endif
    }

    if (sub == "bias_add4") {
        bench_bias_add4();
    }

    if (sub == "transpose_front") {
        bench_transpose_front();
    }
}
