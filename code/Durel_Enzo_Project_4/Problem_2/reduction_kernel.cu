#include "const.h"

#include <limits.h>

__global__ void reduction_kernel(
    const unsigned int* hash_array,
    const unsigned int* nonce_array,
    unsigned int* block_min_hashes,
    unsigned int* block_min_nonces,
    int N
) {
    extern __shared__ unsigned int sdata[];

    unsigned int* s_hashes = sdata;
    unsigned int* s_nonces = &sdata[blockDim.x];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load input into shared memory
    unsigned int hash1 = (i < N) ? hash_array[i] : UINT_MAX;
    unsigned int hash2 = (i + blockDim.x < N) ? hash_array[i + blockDim.x] : UINT_MAX;
    unsigned int nonce1 = (i < N) ? nonce_array[i] : 0;
    unsigned int nonce2 = (i + blockDim.x < N) ? nonce_array[i + blockDim.x] : 0;

    if (hash1 < hash2) {
        s_hashes[tid] = hash1;
        s_nonces[tid] = nonce1;
    } else {
        s_hashes[tid] = hash2;
        s_nonces[tid] = nonce2;
    }

    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_hashes[tid + s] < s_hashes[tid]) {
                s_hashes[tid] = s_hashes[tid + s];
                s_nonces[tid] = s_nonces[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min_hashes[blockIdx.x] = s_hashes[0];
        block_min_nonces[blockIdx.x] = s_nonces[0];
    }
}


