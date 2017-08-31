//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

template <class T, size_t blockSize>
__device__ void sum_reduce_impl(unsigned int n, const T* g_idata, size_t incx, T* g_odata, volatile T* sdata, T mySum){
    unsigned int tid      = threadIdx.x;

    // Do the reduction in shared memory
    // This part is fully unrolled

    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    __syncthreads();

    // Compute the reduction of the last warp

#if (__CUDA_ARCH__ >= 300 )
    // Compute last warp reduction with warp shuffling

    if (tid < 32) {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64){
            mySum += sdata[tid + 32];
        }

        // Reduce final warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // Fully unroll the reduction within a single warp

    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = mySum = mySum + sdata[tid + 8];
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = mySum = mySum + sdata[tid + 4];
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = mySum = mySum + sdata[tid + 2];
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = mySum = mySum + sdata[tid + 1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0){
        g_odata[blockIdx.x] = mySum;
    }
}
