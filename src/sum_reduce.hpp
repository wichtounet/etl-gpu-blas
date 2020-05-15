//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef CUDART_VERSION
#error "Unsupported CUDA version"
#endif

template <class T, size_t blockSize>
__device__ void warpReduce(volatile T *shared_data, size_t tid){
    if (blockSize >= 64) shared_data[tid] += shared_data[tid + 32];
    if (blockSize >= 32) shared_data[tid] += shared_data[tid + 16];
    if (blockSize >= 16) shared_data[tid] += shared_data[tid +  8];
    if (blockSize >=  8) shared_data[tid] += shared_data[tid +  4];
    if (blockSize >=  4) shared_data[tid] += shared_data[tid +  2];
    if (blockSize >=  2) shared_data[tid] += shared_data[tid +  1];
}

template <class T, size_t blockSize>
__device__ void sum_reduce_impl(T* output, volatile T* shared_data){
    size_t tid      = threadIdx.x;

    // Do the reduction in shared memory
    // This part is fully unrolled

    if (blockSize >= 1024) {
        if (tid < 512) {
            shared_data[tid] += shared_data[tid + 512];
        }

        __syncthreads();
    }

    if (blockSize >= 512) {
        if (tid < 256) {
            shared_data[tid] += shared_data[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            shared_data[tid] += shared_data[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            shared_data[tid] += shared_data[tid + 64];
        }

        __syncthreads();
    }

    // Compute the reduction of the last warp

    if (tid < 32) {
        warpReduce<T, blockSize>(shared_data, tid);
    }

    // write result for this block to global mem
    if (tid == 0){
        output[blockIdx.x] = shared_data[0];
    }
}
