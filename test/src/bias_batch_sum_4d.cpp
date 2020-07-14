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

TEST_CASE("bias_batch_sum4/s/0", "[float][bias_batch_sum4]") {
    const size_t B = 2;
    const size_t N = 3;
    const size_t W = 2;
    const size_t H = 2;

    float* x_cpu = new float[B * N * W * H];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < B * N * W * H; ++i) {
        x_cpu[i] = i + 1;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 8 * 8.5f);
    REQUIRE(y_cpu[1] == 8 * 12.5f);
    REQUIRE(y_cpu[2] == 8 * 16.5f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_sum4/d/0", "[double][bias_batch_sum4]") {
    const size_t B = 2;
    const size_t N = 3;
    const size_t W = 2;
    const size_t H = 2;

    double* x_cpu = new double[B * N * W * H];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < B * N * W * H; ++i) {
        x_cpu[i] = i + 1;
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 8 * 8.5f);
    REQUIRE(y_cpu[1] == 8 * 12.5f);
    REQUIRE(y_cpu[2] == 8 * 16.5f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_mean4/s/0", "[float][bias_batch_mean4]") {
    const size_t B = 2;
    const size_t N = 3;
    const size_t W = 2;
    const size_t H = 2;

    float* x_cpu = new float[B * N * W * H];
    float* y_cpu = new float[N];

    for (size_t i = 0; i < B * N * W * H; ++i) {
        x_cpu[i] = i + 1;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_mean4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 8.5f);
    REQUIRE(y_cpu[1] == 12.5f);
    REQUIRE(y_cpu[2] == 16.5f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_mean4/d/0", "[double][bias_batch_mean4]") {
    const size_t B = 2;
    const size_t N = 3;
    const size_t W = 2;
    const size_t H = 2;

    double* x_cpu = new double[B * N * W * H];
    double* y_cpu = new double[N];

    for (size_t i = 0; i < B * N * W * H; ++i) {
        x_cpu[i] = i + 1;
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_mean4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 8.5f);
    REQUIRE(y_cpu[1] == 12.5f);
    REQUIRE(y_cpu[2] == 16.5f);

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
}

TEST_CASE("bias_batch_sum4/s/1", "[float][bias_batch_sum4]") {
    const size_t B = 11;
    const size_t N = 13;
    const size_t W = 33;
    const size_t H = 33;

    float* x_cpu = new float[B * N * W * H];
    float* y_cpu = new float[N];
    float* y_res = new float[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    x_cpu[i] = (i + 1) / 111.0f;
                    y_res[n] += x_cpu[i];
                    ++i;
                }
            }
        }
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_res;
}

TEST_CASE("bias_batch_sum4/d/1", "[float][bias_batch_sum4]") {
    const size_t B = 13;
    const size_t N = 13;
    const size_t W = 37;
    const size_t H = 33;

    float* x_cpu = new float[B * N * W * H];
    float* y_cpu = new float[N];
    float* y_res = new float[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    x_cpu[i] = (i + 1) / 123.0f;
                    y_res[n] += x_cpu[i];
                    ++i;
                }
            }
        }
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_res;
}

TEST_CASE("bias_batch_sum4/d/2", "[float][bias_batch_sum4]") {
    const size_t B = 9;
    const size_t N = 15;
    const size_t W = 41;
    const size_t H = 45;

    float* x_cpu = new float[B * N * W * H];
    float* y_cpu = new float[N];
    float* y_res = new float[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    x_cpu[i] = (i + 1) / 123.0f;
                    y_res[n] += x_cpu[i];
                    ++i;
                }
            }
        }
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_res;
}

TEST_CASE("bias_batch_sum4/d/3", "[double][bias_batch_sum4]") {
    const size_t B = 132;
    const size_t N = 21;
    const size_t W = 49;
    const size_t H = 145;

    double* x_cpu = new double[B * N * W * H];
    double* y_cpu = new double[N];
    double* y_res = new double[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    x_cpu[i] = (i + 1) / 123.0f;
                    y_res[n] += x_cpu[i];
                    ++i;
                }
            }
        }
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_res;
}

TEST_CASE("bias_batch_sum4/d/4", "[double][bias_batch_sum4]") {
    const size_t B = 132;
    const size_t N = 21;
    const size_t W = 7;
    const size_t H = 8;

    double* x_cpu = new double[B * N * W * H];
    double* y_cpu = new double[N];
    double* y_res = new double[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    x_cpu[i] = (i + 1) / 123.0f;
                    y_res[n] += x_cpu[i];
                    ++i;
                }
            }
        }
    }

    double* x_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, B * N  * W * H* sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, B * N  * W * H* sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_sum4(B, N, W, H, x_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(x_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] x_cpu;
    delete[] y_cpu;
    delete[] y_res;
}

TEST_CASE("bias_batch_var4/s/0", "[float][bias_batch_var4]") {
    const size_t B = 33;
    const size_t N = 19;
    const size_t W = 17;
    const size_t H = 9;

    float* a_cpu = new float[B * N * W * H];
    float* b_cpu = new float[N];
    float* y_cpu = new float[N];
    float* y_res = new float[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
        b_cpu[n] = n + 1;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    a_cpu[i] = (i + 1) / 123.0f;
                    y_res[n] += (a_cpu[i] - b_cpu[n]) * (a_cpu[i] - b_cpu[n]);
                    ++i;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        y_res[n] /= B * W * H;
    }

    float* a_gpu;
    float* b_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&a_gpu, B * N  * W * H* sizeof(float)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(float)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, B * N * W * H * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_var4(B, N, W, H, a_gpu, b_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] y_cpu;
    delete[] y_res;
}

TEST_CASE("bias_batch_var4/d/4", "[double][bias_batch_var4]") {
    const size_t B = 132;
    const size_t N = 21;
    const size_t W = 7;
    const size_t H = 8;

    double* a_cpu = new double[B * N * W * H];
    double* b_cpu = new double[N];
    double* y_cpu = new double[N];
    double* y_res = new double[N];

    for (size_t n = 0; n < N; ++n) {
        y_res[n] = 0;
        b_cpu[n] = n + 1;
    }

    size_t i = 0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t h = 0; h < H; ++h) {
                    a_cpu[i] = (i + 1) / 123.0f;
                    y_res[n] += (a_cpu[i] - b_cpu[n]) * (a_cpu[i] - b_cpu[n]);
                    ++i;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        y_res[n] /= B * W * H;
    }

    double* a_gpu;
    double* b_gpu;
    double* y_gpu;

    cuda_check(cudaMalloc((void**)&a_gpu, B * N  * W * H* sizeof(double)));
    cuda_check(cudaMalloc((void**)&b_gpu, N * sizeof(double)));
    cuda_check(cudaMalloc((void**)&y_gpu, N * sizeof(double)));

    cuda_check(cudaMemcpy(a_gpu, a_cpu, B * N * W * H * sizeof(double), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b_gpu, b_cpu, N * sizeof(double), cudaMemcpyHostToDevice));

    egblas_dbias_batch_var4(B, N, W, H, a_gpu, b_gpu, y_gpu);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        REQUIRE(y_cpu[i] == Approx(y_res[i]));
    }

    cuda_check(cudaFree(a_gpu));
    cuda_check(cudaFree(y_gpu));

    delete[] a_cpu;
    delete[] y_cpu;
    delete[] y_res;
}
