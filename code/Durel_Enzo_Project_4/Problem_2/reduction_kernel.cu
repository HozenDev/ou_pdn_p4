#include "const.h"

#include <limits.h>

// Tree-based reduction to find min hash and corresponding nonce
__global__ void reduce_min_hash(
    const unsigned int* hash_array,
    const unsigned int* nonce_array,
    unsigned int* min_hash_out,
    unsigned int* min_nonce_out,
    int N
) {
    extern __shared__ unsigned int shared[];
    unsigned int* hash_shared = shared;
    unsigned int* nonce_shared = &shared[blockDim.x];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // Load data into shared memory
    if (global_idx < N) {
        hash_shared[local_idx] = hash_array[global_idx];
        nonce_shared[local_idx] = nonce_array[global_idx];
    } else {
        hash_shared[local_idx] = UINT_MAX;
        nonce_shared[local_idx] = 0;
    }

    __syncthreads();

    // Tree-based reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            if (hash_shared[local_idx + stride] < hash_shared[local_idx]) {
                hash_shared[local_idx] = hash_shared[local_idx + stride];
                nonce_shared[local_idx] = nonce_shared[local_idx + stride];
            }
        }
        __syncthreads();
    }

    // Write block-level result
    if (local_idx == 0) {
        min_hash_out[blockIdx.x] = hash_shared[0];
        min_nonce_out[blockIdx.x] = nonce_shared[0];
    }
}

