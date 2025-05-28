#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void block_radix_sort_2bit_kernel(const uint64_t* __restrict__ input_keys, 
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
    uint32_t* shared_scan_00 = shared_bits + block_size;
    uint32_t* shared_scan_01 = shared_scan_00 + block_size;
    uint32_t* shared_scan_10 = shared_scan_01 + block_size;
    uint32_t* shared_scan_11 = shared_scan_10 + block_size;
    
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
        bits = (key >> bit_position) & 3;
    }
    
    shared_keys[thread_id] = key;
    shared_values[thread_id] = value;
    shared_bits[thread_id] = bits;
    __syncthreads();
    
    uint32_t is_00 = (bits == 0) ? 1 : 0;
    uint32_t is_01 = (bits == 1) ? 1 : 0;
    uint32_t is_10 = (bits == 2) ? 1 : 0;
    uint32_t is_11 = (bits == 3) ? 1 : 0;
    
    shared_scan_00[thread_id] = is_00;
    shared_scan_01[thread_id] = is_01;
    shared_scan_10[thread_id] = is_10;
    shared_scan_11[thread_id] = is_11;
    __syncthreads();
    
    for (int stride = 1; stride < block_size; stride *= 2) {
        uint32_t temp_00 = 0, temp_01 = 0, temp_10 = 0, temp_11 = 0;
        
        if (thread_id >= stride) {
            temp_00 = shared_scan_00[thread_id - stride];
            temp_01 = shared_scan_01[thread_id - stride];
            temp_10 = shared_scan_10[thread_id - stride];
            temp_11 = shared_scan_11[thread_id - stride];
        }
        __syncthreads();
        
        if (thread_id >= stride) {
            shared_scan_00[thread_id] += temp_00;
            shared_scan_01[thread_id] += temp_01;
            shared_scan_10[thread_id] += temp_10;
            shared_scan_11[thread_id] += temp_11;
        }
        __syncthreads();
    }
    
    uint32_t exc_scan_00 = (thread_id == 0) ? 0 : shared_scan_00[thread_id - 1];
    uint32_t exc_scan_01 = (thread_id == 0) ? 0 : shared_scan_01[thread_id - 1];
    uint32_t exc_scan_10 = (thread_id == 0) ? 0 : shared_scan_10[thread_id - 1];
    uint32_t exc_scan_11 = (thread_id == 0) ? 0 : shared_scan_11[thread_id - 1];
    
    __shared__ uint32_t bucket_counts[4];
    if (thread_id == 0) {
        bucket_counts[0] = shared_scan_00[block_elements - 1]; // count_00
        bucket_counts[1] = shared_scan_01[block_elements - 1]; // count_01
        bucket_counts[2] = shared_scan_10[block_elements - 1]; // count_10
        bucket_counts[3] = shared_scan_11[block_elements - 1]; // count_11
        
        block_offsets[block_id * 4 + 0] = bucket_counts[0];
        block_offsets[block_id * 4 + 1] = bucket_counts[1];
        block_offsets[block_id * 4 + 2] = bucket_counts[2];
        block_offsets[block_id * 4 + 3] = bucket_counts[3];
    }
    __syncthreads();
    
    uint32_t local_pos;
    if (bits == 0) {
        local_pos = exc_scan_00;
    } else if (bits == 1) {
        local_pos = bucket_counts[0] + exc_scan_01;
    } else if (bits == 2) {
        local_pos = bucket_counts[0] + bucket_counts[1] + exc_scan_10;
    } else { // bits == 3
        local_pos = bucket_counts[0] + bucket_counts[1] + bucket_counts[2] + exc_scan_11;
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

__global__ void global_merge_2bit_kernel(const uint64_t* __restrict__ input_keys,
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
    
    uint32_t count_00 = block_offsets[block_id * 4 + 0];
    uint32_t count_01 = block_offsets[block_id * 4 + 1];
    uint32_t count_10 = block_offsets[block_id * 4 + 2];
    uint32_t count_11 = block_offsets[block_id * 4 + 3];
    
    uint32_t bucket;
    uint32_t local_bucket_pos;
    
    if (thread_id < count_00) {
        bucket = 0;
        local_bucket_pos = thread_id;
    } else if (thread_id < count_00 + count_01) {
        bucket = 1;
        local_bucket_pos = thread_id - count_00;
    } else if (thread_id < count_00 + count_01 + count_10) {
        bucket = 2;
        local_bucket_pos = thread_id - count_00 - count_01;
    } else {
        bucket = 3;
        local_bucket_pos = thread_id - count_00 - count_01 - count_10;
    }
    
    uint32_t global_bucket_start = global_bucket_offsets[block_id * 4 + bucket];
    uint32_t global_pos = global_bucket_start + local_bucket_pos;
    
    output_keys[global_pos] = key;
    output_values[global_pos] = value;
}

void compute_global_bucket_offsets_2bit(uint32_t* block_offsets, uint32_t* global_bucket_offsets, 
                                       int num_blocks) {
    
    uint32_t total_counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < num_blocks; i++) {
        total_counts[0] += block_offsets[i * 4 + 0];
        total_counts[1] += block_offsets[i * 4 + 1];
        total_counts[2] += block_offsets[i * 4 + 2];
        total_counts[3] += block_offsets[i * 4 + 3];
    }
    
    uint32_t global_starts[4];
    global_starts[0] = 0;
    global_starts[1] = total_counts[0];
    global_starts[2] = total_counts[0] + total_counts[1];
    global_starts[3] = total_counts[0] + total_counts[1] + total_counts[2];
    
    uint32_t running_sums[4] = {0, 0, 0, 0};
    
    for (int i = 0; i < num_blocks; i++) {
        global_bucket_offsets[i * 4 + 0] = global_starts[0] + running_sums[0];
        global_bucket_offsets[i * 4 + 1] = global_starts[1] + running_sums[1];
        global_bucket_offsets[i * 4 + 2] = global_starts[2] + running_sums[2];
        global_bucket_offsets[i * 4 + 3] = global_starts[3] + running_sums[3];
        
        running_sums[0] += block_offsets[i * 4 + 0];
        running_sums[1] += block_offsets[i * 4 + 1];
        running_sums[2] += block_offsets[i * 4 + 2];
        running_sums[3] += block_offsets[i * 4 + 3];
    }
}

void radix_sort_2bit_coalesced(uint64_t* unsorted_keys, uint64_t* sorted_keys, 
                              uint32_t* unsorted_values, uint32_t* sorted_values, 
                              int num_elements) {
    
    const int block_size = 512; 
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* block_offsets;           // 4 counts per block
    uint32_t* global_bucket_offsets;   // 4 global offsets per block
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&block_offsets, num_blocks * 4 * sizeof(uint32_t));
    cudaMalloc(&global_bucket_offsets, num_blocks * 4 * sizeof(uint32_t));
    
    uint32_t* h_block_offsets = new uint32_t[num_blocks * 4];
    uint32_t* h_global_bucket_offsets = new uint32_t[num_blocks * 4];
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int bit = 0; bit < 64; bit += 2) {
        
        size_t shared_mem_size = block_size * (sizeof(uint64_t) + sizeof(uint32_t) + // keys + values
                                              sizeof(uint32_t) +                      // bits
                                              4 * sizeof(uint32_t));                  // 4 scan arrays
        
        block_radix_sort_2bit_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            sorted_keys, sorted_values, temp_keys, temp_values, 
            block_offsets, bit, num_elements);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_block_offsets, block_offsets, num_blocks * 4 * sizeof(uint32_t), 
                   cudaMemcpyDeviceToHost);
        
        compute_global_bucket_offsets_2bit(h_block_offsets, h_global_bucket_offsets, num_blocks);
        
        cudaMemcpy(global_bucket_offsets, h_global_bucket_offsets, 
                   num_blocks * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        global_merge_2bit_kernel<<<num_blocks, block_size>>>(
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