#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void block_radix_sort_4bit_kernel(const uint64_t* __restrict__ input_keys, 
                                            const uint32_t* __restrict__ input_values,
                                            uint64_t* __restrict__ output_keys,
                                            uint32_t* __restrict__ output_values,
                                            uint32_t* __restrict__ block_offsets,
                                            int bit_position, 
                                            int num_elements) {
    
    const int block_size = blockDim.x;
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    
    extern __shared__ uint8_t shared_mem[];
    uint64_t* shared_keys = (uint64_t*)shared_mem;
    uint32_t* shared_values = (uint32_t*)(shared_keys + block_size);
    uint32_t* shared_bits = (uint32_t*)(shared_values + block_size);
    
    uint32_t* shared_scan[16];
    for (int i = 0; i < 16; i++) {
        shared_scan[i] = (uint32_t*)(shared_bits + block_size + i * block_size);
    }
    
    int elements_per_block = block_size;
    int block_start = block_id * elements_per_block;
    int block_end = min(block_start + elements_per_block, num_elements);
    int block_elements = block_end - block_start;
    
    uint64_t key = 0;
    uint32_t value = 0;
    uint32_t bits = 0;
    
    if (thread_id < block_elements) {
        int load_idx = block_start + thread_id;
        key = input_keys[load_idx];
        value = input_values[load_idx];
        bits = (key >> bit_position) & 15;
    }
    
    shared_keys[thread_id] = key;
    shared_values[thread_id] = value;
    shared_bits[thread_id] = bits;
    __syncthreads();
    
    for (int bucket = 0; bucket < 16; bucket++) {
        shared_scan[bucket][thread_id] = (bits == bucket) ? 1 : 0;
    }
    __syncthreads();
    
    for (int stride = 1; stride < block_size; stride *= 2) {
        uint32_t temp_vals[16];
        
        for (int bucket = 0; bucket < 16; bucket++) {
            temp_vals[bucket] = (thread_id >= stride) ? shared_scan[bucket][thread_id - stride] : 0;
        }
        __syncthreads();
        
        for (int bucket = 0; bucket < 16; bucket++) {
            if (thread_id >= stride) {
                shared_scan[bucket][thread_id] += temp_vals[bucket];
            }
        }
        __syncthreads();
    }
    
    uint32_t exc_scan[16];
    __shared__ uint32_t bucket_counts[16];
    
    for (int bucket = 0; bucket < 16; bucket++) {
        exc_scan[bucket] = (thread_id == 0) ? 0 : shared_scan[bucket][thread_id - 1];
    }
    
    if (thread_id == 0) {
        for (int bucket = 0; bucket < 16; bucket++) {
            bucket_counts[bucket] = (block_elements > 0) ? shared_scan[bucket][block_elements - 1] : 0;
            block_offsets[block_id * 16 + bucket] = bucket_counts[bucket];
        }
    }
    __syncthreads();
    
    __shared__ uint32_t bucket_boundaries[16];
    if (thread_id == 0) {
        bucket_boundaries[0] = 0;
        for (int bucket = 1; bucket < 16; bucket++) {
            bucket_boundaries[bucket] = bucket_boundaries[bucket - 1] + bucket_counts[bucket - 1];
        }
    }
    __syncthreads();
    
    uint32_t local_pos = bucket_boundaries[bits] + exc_scan[bits];
    
    if (thread_id < block_elements) {
        shared_keys[local_pos] = key;
        shared_values[local_pos] = value;
    }
    __syncthreads();
    
    if (thread_id < block_elements) {
        int store_idx = block_start + thread_id;
        output_keys[store_idx] = shared_keys[thread_id];
        output_values[store_idx] = shared_values[thread_id];
    }
}

__global__ void global_merge_4bit_kernel(const uint64_t* __restrict__ input_keys,
                                        const uint32_t* __restrict__ input_values,
                                        uint64_t* __restrict__ output_keys,
                                        uint32_t* __restrict__ output_values,
                                        const uint32_t* __restrict__ block_offsets,
                                        const uint32_t* __restrict__ global_bucket_offsets,
                                        int num_elements,
                                        int num_blocks) {
    
    const int block_size = blockDim.x;
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    
    int elements_per_block = block_size;
    int block_start = block_id * elements_per_block;
    int block_end = min(block_start + elements_per_block, num_elements);
    int block_elements = block_end - block_start;
    
    if (thread_id >= block_elements) return;
    
    int src_idx = block_start + thread_id;
    uint64_t key = input_keys[src_idx];
    uint32_t value = input_values[src_idx];
    
    __shared__ uint32_t bucket_counts[16];
    __shared__ uint32_t bucket_boundaries[16];
    
    if (thread_id < 16) {
        bucket_counts[thread_id] = block_offsets[block_id * 16 + thread_id];
    }
    __syncthreads();
    
    if (thread_id == 0) {
        bucket_boundaries[0] = 0;
        for (int bucket = 1; bucket < 16; bucket++) {
            bucket_boundaries[bucket] = bucket_boundaries[bucket - 1] + bucket_counts[bucket - 1];
        }
    }
    __syncthreads();
    
    uint32_t bucket = 0;
    uint32_t local_bucket_pos = thread_id;
    
    for (int b = 0; b < 16; b++) {
        if (thread_id < bucket_boundaries[b]) {
            bucket = (b > 0) ? b - 1 : 0;
            local_bucket_pos = (b > 0) ? thread_id - bucket_boundaries[b - 1] : thread_id;
            break;
        }
        if (b == 15) { // Last bucket
            bucket = 15;
            local_bucket_pos = thread_id - bucket_boundaries[15];
        }
    }
    
    uint32_t global_bucket_start = global_bucket_offsets[block_id * 16 + bucket];
    uint32_t global_pos = global_bucket_start + local_bucket_pos;
    
    output_keys[global_pos] = key;
    output_values[global_pos] = value;
}

void compute_global_bucket_offsets_4bit(uint32_t* block_offsets, uint32_t* global_bucket_offsets, 
                                       int num_blocks) {
    
    uint32_t total_counts[16] = {0};
    for (int i = 0; i < num_blocks; i++) {
        for (int bucket = 0; bucket < 16; bucket++) {
            total_counts[bucket] += block_offsets[i * 16 + bucket];
        }
    }
    
    uint32_t global_starts[16];
    global_starts[0] = 0;
    for (int bucket = 1; bucket < 16; bucket++) {
        global_starts[bucket] = global_starts[bucket - 1] + total_counts[bucket - 1];
    }
    
    uint32_t running_sums[16] = {0};
    
    for (int i = 0; i < num_blocks; i++) {
        for (int bucket = 0; bucket < 16; bucket++) {
            global_bucket_offsets[i * 16 + bucket] = global_starts[bucket] + running_sums[bucket];
            running_sums[bucket] += block_offsets[i * 16 + bucket];
        }
    }
}

void radix_sort_4bit_coalesced(uint64_t* unsorted_keys, uint64_t* sorted_keys, 
                              uint32_t* unsorted_values, uint32_t* sorted_values, 
                              int num_elements) {
    
    const int block_size = 256; 
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* block_offsets;           
    uint32_t* global_bucket_offsets;  
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&block_offsets, num_blocks * 16 * sizeof(uint32_t));
    cudaMalloc(&global_bucket_offsets, num_blocks * 16 * sizeof(uint32_t));
    
    uint32_t* h_block_offsets = new uint32_t[num_blocks * 16];
    uint32_t* h_global_bucket_offsets = new uint32_t[num_blocks * 16];
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int bit = 0; bit < 64; bit += 4) {
        
        size_t shared_mem_size = block_size * (sizeof(uint64_t) +      // keys
                                              sizeof(uint32_t) +       // values  
                                              sizeof(uint32_t) +       // bits
                                              16 * sizeof(uint32_t));  // 16 scan arrays
        
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        if (shared_mem_size > deviceProp.sharedMemPerBlock) {
            printf("Error: Shared memory requirement (%zu bytes) exceeds device limit (%zu bytes)\n", 
                   shared_mem_size, deviceProp.sharedMemPerBlock);
            return;
        }
        
        block_radix_sort_4bit_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            sorted_keys, sorted_values, temp_keys, temp_values, 
            block_offsets, bit, num_elements);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_block_offsets, block_offsets, num_blocks * 16 * sizeof(uint32_t), 
                   cudaMemcpyDeviceToHost);
        
        compute_global_bucket_offsets_4bit(h_block_offsets, h_global_bucket_offsets, num_blocks);
        
        cudaMemcpy(global_bucket_offsets, h_global_bucket_offsets, 
                   num_blocks * 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        global_merge_4bit_kernel<<<num_blocks, block_size>>>(
            temp_keys, temp_values, sorted_keys, sorted_values,
            block_offsets, global_bucket_offsets, num_elements, num_blocks);
        
        cudaDeviceSynchronize();
    }
    
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(block_offsets);
    cudaFree(global_bucket_offsets);
    
    delete[] h_block_offsets;
    delete[] h_global_bucket_offsets;
}