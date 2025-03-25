#include "const.h"

#include <limits.h>

/* Hash Function -----------------------------------
   *       Generates a hash value from a nonce and an array of transactions.
*/
__device__
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions, unsigned int max)
{
    unsigned int hash = (nonce + transactions[0] * (index + 1)) % max;
    for (int j = 1; j < n_transactions; j++) {
	hash = (hash + transactions[j] * (index + 1)) % max;
    }
    return hash;
}

/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
__global__
void hash_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate hash value
    if (index < array_size) {
	hash_array[index] = generate_hash(nonce_array[index], index, transactions, n_transactions, mod);
    }

} // End Hash Kernel //

__global__
void reduce_min_hash(const unsigned int* hash_array,
		     const unsigned int* nonce_array,
		     unsigned int* min_hash_block,
		     unsigned int* min_nonce_block,
		     int array_size)
{
    __shared__ unsigned int shared[BLOCK_SIZE];

    unsigned int* hash_shared = shared;
    unsigned int* nonce_shared = &shared[blockDim.x];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // Load into shared memory
    if (global_idx < array_size) {
        hash_shared[local_idx] = hash_array[global_idx];
        nonce_shared[local_idx] = nonce_array[global_idx];
    } else {
        hash_shared[local_idx] = UINT_MAX;
        nonce_shared[local_idx] = 0;
    }

    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            if (hash_shared[local_idx + stride] < hash_shared[local_idx]) {
                hash_shared[local_idx] = hash_shared[local_idx + stride];
                nonce_shared[local_idx] = nonce_shared[local_idx + stride];
            }
        }
        __syncthreads();
    }

    // Write the result from each block
    if (local_idx == 0) {
        min_hash_block[blockIdx.x] = hash_shared[0];
        min_nonce_block[blockIdx.x] = nonce_shared[0];
    }
}

