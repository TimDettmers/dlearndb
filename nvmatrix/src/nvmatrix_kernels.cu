/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 * Copyright 2014 Nervana Systems Inc.  All rights reserved.
 *
 * * Added argmin, argmax support, other operations further fleshed out.
 * ---------------------------------------------------------------------------
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/nvmatrix_kernels.cuh"

__global__ void kTile(const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (uint i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
        const uint y = i / tgtWidth;
        const uint x = i % tgtWidth;
        const uint srcY = y % srcHeight;
        const uint srcX = x % srcWidth;
        tgt[i] = src[srcY * srcWidth + srcX];
    }
}

__global__ void kDotProduct_r(float* a, float* b, float* target,  const uint numElements) {
    __shared__ float shmem[DP_BLOCKSIZE];

    uint eidx = DP_BLOCKSIZE * blockIdx.x + threadIdx.x;
    shmem[threadIdx.x] = 0;
    if (eidx < gridDim.x * DP_BLOCKSIZE) {
        for (; eidx < numElements; eidx += gridDim.x * DP_BLOCKSIZE) {
            shmem[threadIdx.x] += a[eidx] * b[eidx];
        }
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 256];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        volatile float* mysh = &shmem[threadIdx.x];
        *mysh += mysh[32];
        *mysh += mysh[16];
        *mysh += mysh[8];
        *mysh += mysh[4];
        *mysh += mysh[2];
        *mysh += mysh[1];
        if (threadIdx.x == 0) {
            target[blockIdx.x] = *mysh;
        }
    }
}

__global__ void kSetupCurand(curandState *state, unsigned long long seed) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}


__global__ void kArgMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float max_vals[32];
    __shared__ unsigned int max_args[32];
    float cur_max = -2e38;
    unsigned int cur_arg = 0;
    float val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[i * width + blockIdx.x];

        if (val > cur_max) {
            cur_max = val;
            cur_arg = i;
        }
    }

    max_vals[threadIdx.x] = cur_max;
    max_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -2e38;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_arg = max_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kArgMaxRowwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float max_vals[32];
    __shared__ unsigned int max_args[32];
    float cur_max = -2e38;
    unsigned int cur_arg = 0;
    float val = 0;
 
    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[blockIdx.x * width + i];

        if (val > cur_max) {
            cur_max = val;
            cur_arg = i;
        }
    }

    max_vals[threadIdx.x] = cur_max;
    max_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -2e38;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_arg = max_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}


__global__ void kArgMinColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float min_vals[32];
    __shared__ unsigned int min_args[32];
    float cur_min = 2e38;
    unsigned int cur_arg = 0;
    float val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[i * width + blockIdx.x];

        if (val < cur_min) {
            cur_min = val;
            cur_arg = i;
        }
    }

    min_vals[threadIdx.x] = cur_min;
    min_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = 2e38;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min) {
                cur_min = min_vals[i];
                cur_arg = min_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kArgMinRowwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float min_vals[32];
    __shared__ unsigned int min_args[32];
    float cur_min = 2e38;
    unsigned int cur_arg = 0;
    float val = 0;
 
    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[blockIdx.x * width + i];

        if (val < cur_min) {
            cur_min = val;
            cur_arg = i;
        }
    }

    min_vals[threadIdx.x] = cur_min;
    min_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = 2e38;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min) {
                cur_min = min_vals[i];
                cur_arg = min_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kArange(float* target, unsigned int width, unsigned int height) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (uint i = idx; i < width * height; i += numThreads) {
       target[i] = (float)i;
    }
}
