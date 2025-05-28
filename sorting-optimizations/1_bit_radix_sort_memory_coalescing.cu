#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void block_radix_sort_kernel(const uint64_t* __restrict__ input_keys, 
                                       const uint32_t* __restrict__ input_values,
                                       uint64_t* __restrict__ output_keys,
                                       uint32_t* __restrict__ output_values,
                                       uint32_t* __restrict__ block_offsets,
                                       int bit_position, 
                                       int num_elements) {
    
    const int block_size = blockDim.x;
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int global_id = block_id * block_size + thread_id;
    
    extern __shared__ uint8_t shared_mem[];
    uint64_t* shared_keys = (uint64_t*)shared_mem;
    uint32_t* shared_values = (uint32_t*)(shared_keys + block_size);
    uint32_t* shared_bits = (uint32_t*)(shared_values + block_size);
    uint32_t* shared_scan = shared_bits + block_size;
    
    int elements_per_block = block_size;
    int block_start = block_id * elements_per_block;
    int block_end = min(block_start + elements_per_block, num_elements);
    int block_elements = block_end - block_start;
    
    uint64_t key = 0;
    uint32_t value = 0;
    uint32_t bit = 0;
    
    if (thread_id < block_elements) {
        int load_idx = block_start + thread_id;
        key = input_keys[load_idx];
        value = input_values[load_idx];
        bit = (key >> bit_position) & 1;
    }
    
    shared_keys[thread_id] = key;
    shared_values[thread_id] = value;
    shared_bits[thread_id] = bit;
    __syncthreads();
    
    uint32_t scan_val = bit;
    shared_scan[thread_id] = scan_val;
    __syncthreads();
    
    for (int stride = 1; stride < block_size; stride *= 2) {
        uint32_t temp = 0;
        if (thread_id >= stride) {
            temp = shared_scan[thread_id - stride];
        }
        __syncthreads();
        
        if (thread_id >= stride) {
            shared_scan[thread_id] += temp;
        }
        __syncthreads();
    }
    
    uint32_t exclusive_scan = (thread_id == 0) ? 0 : shared_scan[thread_id - 1];
    
    __shared__ uint32_t block_zeros, block_ones;
    if (thread_id == 0) {
        block_ones = shared_scan[block_elements - 1];
        block_zeros = block_elements - block_ones;
        
        block_offsets[block_id * 2] = block_zeros;
        block_offsets[block_id * 2 + 1] = block_ones;
    }
    __syncthreads();
    
    uint32_t local_pos;
    if (bit == 0) {
        local_pos = thread_id - exclusive_scan;  // Position among zeros
    } else {
        local_pos = block_zeros + exclusive_scan;  // Position among ones
    }
    
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

__global__ void global_merge_kernel(const uint64_t* __restrict__ input_keys,
                                   const uint32_t* __restrict__ input_values,
                                   uint64_t* __restrict__ output_keys,
                                   uint32_t* __restrict__ output_values,
                                   const uint32_t* __restrict__ block_offsets,
                                   const uint32_t* __restrict__ global_zero_offsets,
                                   const uint32_t* __restrict__ global_one_offsets,
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
    
    uint32_t block_zeros = block_offsets[block_id * 2];
    uint32_t is_one = (thread_id >= block_zeros) ? 1 : 0;
    
    uint32_t global_pos;
    if (is_one == 0) {
        uint32_t global_zero_start = global_zero_offsets[block_id];
        global_pos = global_zero_start + thread_id;
    } else {
        uint32_t global_one_start = global_one_offsets[block_id];
        global_pos = global_one_start + (thread_id - block_zeros);
    }
    
    output_keys[global_pos] = key;
    output_values[global_pos] = value;
}

void compute_global_offsets(uint32_t* block_offsets, uint32_t* global_zero_offsets, 
                           uint32_t* global_one_offsets, int num_blocks, uint32_t total_zeros) {
    
    uint32_t zero_sum = 0;
    uint32_t one_sum = total_zeros;
    
    for (int i = 0; i < num_blocks; i++) {
        global_zero_offsets[i] = zero_sum;
        global_one_offsets[i] = one_sum;
        
        zero_sum += block_offsets[i * 2];      // zeros in block i
        one_sum += block_offsets[i * 2 + 1];   // ones in block i
    }
}

void radix_sort_coalesced(uint64_t* unsorted_keys, uint64_t* sorted_keys, 
                         uint32_t* unsorted_values, uint32_t* sorted_values, 
                         int num_elements) {
    
    const int block_size = 512;  
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* block_offsets;
    uint32_t* global_zero_offsets;
    uint32_t* global_one_offsets;
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&block_offsets, num_blocks * 2 * sizeof(uint32_t));
    cudaMalloc(&global_zero_offsets, num_blocks * sizeof(uint32_t));
    cudaMalloc(&global_one_offsets, num_blocks * sizeof(uint32_t));
    
    uint32_t* h_block_offsets = new uint32_t[num_blocks * 2];
    uint32_t* h_global_zero_offsets = new uint32_t[num_blocks];
    uint32_t* h_global_one_offsets = new uint32_t[num_blocks];
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int bit = 0; bit < 64; bit++) {
        
        size_t shared_mem_size = block_size * (sizeof(uint64_t) + 2 * sizeof(uint32_t)) + 
                                block_size * sizeof(uint32_t);
        
        block_radix_sort_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            sorted_keys, sorted_values, temp_keys, temp_values, 
            block_offsets, bit, num_elements);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_block_offsets, block_offsets, num_blocks * 2 * sizeof(uint32_t), 
                   cudaMemcpyDeviceToHost);
        
        uint32_t total_zeros = 0;
        for (int i = 0; i < num_blocks; i++) {
            total_zeros += h_block_offsets[i * 2];
        }
        
        compute_global_offsets(h_block_offsets, h_global_zero_offsets, 
                              h_global_one_offsets, num_blocks, total_zeros);
        
        cudaMemcpy(global_zero_offsets, h_global_zero_offsets, 
                   num_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(global_one_offsets, h_global_one_offsets, 
                   num_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        global_merge_kernel<<<num_blocks, block_size>>>(
            temp_keys, temp_values, sorted_keys, sorted_values,
            block_offsets, global_zero_offsets, global_one_offsets,
            num_elements, num_blocks);
        
        cudaDeviceSynchronize();
    }
    
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(block_offsets);
    cudaFree(global_zero_offsets);
    cudaFree(global_one_offsets);
    
    delete[] h_block_offsets;
    delete[] h_global_zero_offsets;
    delete[] h_global_one_offsets;
}