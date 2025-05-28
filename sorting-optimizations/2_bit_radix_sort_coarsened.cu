#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void block_radix_sort_2bit_multi_kernel(const uint64_t* __restrict__ input_keys, 
                                                   const uint32_t* __restrict__ input_values,
                                                   uint64_t* __restrict__ output_keys,
                                                   uint32_t* __restrict__ output_values,
                                                   uint32_t* __restrict__ block_offsets,
                                                   int bit_position, 
                                                   int num_elements,
                                                   int elements_per_thread) {
    
    const int block_size = blockDim.x;
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int elements_per_block = block_size * elements_per_thread;
    
    extern __shared__ uint8_t shared_mem[];
    uint64_t* shared_keys = (uint64_t*)shared_mem;
    uint32_t* shared_values = (uint32_t*)(shared_keys + elements_per_block);
    uint32_t* shared_bits = (uint32_t*)(shared_values + elements_per_block);
    uint32_t* shared_scan_00 = shared_bits + elements_per_block;
    uint32_t* shared_scan_01 = shared_scan_00 + elements_per_block;
    uint32_t* shared_scan_10 = shared_scan_01 + elements_per_block;
    uint32_t* shared_scan_11 = shared_scan_10 + elements_per_block;
    
    int block_start = block_id * elements_per_block;
    int block_end = min(block_start + elements_per_block, num_elements);
    int block_elements = block_end - block_start;
    
    uint64_t thread_keys[8];  // Support up to 8 elements per thread
    uint32_t thread_values[8];
    uint32_t thread_bits[8];
    int thread_element_count = 0;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int global_idx = block_start + thread_id + i * block_size;
        int shared_idx = thread_id + i * block_size;
        
        if (global_idx < block_end && shared_idx < block_elements) {
            uint64_t key = input_keys[global_idx];
            uint32_t value = input_values[global_idx];
            uint32_t bits = (key >> bit_position) & 3;
            
            shared_keys[shared_idx] = key;
            shared_values[shared_idx] = value;
            shared_bits[shared_idx] = bits;
            
            // Store in thread-local arrays for processing
            thread_keys[thread_element_count] = key;
            thread_values[thread_element_count] = value;
            thread_bits[thread_element_count] = bits;
            thread_element_count++;
        }
    }
    __syncthreads();
    
    for (int i = 0; i < elements_per_thread; i++) {
        int shared_idx = thread_id + i * block_size;
        if (shared_idx < block_elements) {
            uint32_t bits = shared_bits[shared_idx];
            shared_scan_00[shared_idx] = (bits == 0) ? 1 : 0;
            shared_scan_01[shared_idx] = (bits == 1) ? 1 : 0;
            shared_scan_10[shared_idx] = (bits == 2) ? 1 : 0;
            shared_scan_11[shared_idx] = (bits == 3) ? 1 : 0;
        } else {
            shared_scan_00[shared_idx] = 0;
            shared_scan_01[shared_idx] = 0;
            shared_scan_10[shared_idx] = 0;
            shared_scan_11[shared_idx] = 0;
        }
    }
    __syncthreads();
    
    for (int stride = 1; stride < elements_per_block; stride *= 2) {
        for (int i = 0; i < elements_per_thread; i++) {
            int shared_idx = thread_id + i * block_size;
            uint32_t temp_00 = 0, temp_01 = 0, temp_10 = 0, temp_11 = 0;
            
            if (shared_idx >= stride && shared_idx < block_elements) {
                temp_00 = shared_scan_00[shared_idx - stride];
                temp_01 = shared_scan_01[shared_idx - stride];
                temp_10 = shared_scan_10[shared_idx - stride];
                temp_11 = shared_scan_11[shared_idx - stride];
            }
            
            thread_keys[i] = temp_00;  
            thread_values[i] = temp_01;
            thread_bits[i] = (temp_10 << 16) | temp_11; 
        }
        __syncthreads();
        
        for (int i = 0; i < elements_per_thread; i++) {
            int shared_idx = thread_id + i * block_size;
            if (shared_idx >= stride && shared_idx < block_elements) {
                shared_scan_00[shared_idx] += thread_keys[i];
                shared_scan_01[shared_idx] += thread_values[i];
                shared_scan_10[shared_idx] += (thread_bits[i] >> 16);
                shared_scan_11[shared_idx] += (thread_bits[i] & 0xFFFF);
            }
        }
        __syncthreads();
    }
    
    __shared__ uint32_t bucket_counts[4];
    if (thread_id == 0) {
        bucket_counts[0] = (block_elements > 0) ? shared_scan_00[block_elements - 1] : 0;
        bucket_counts[1] = (block_elements > 0) ? shared_scan_01[block_elements - 1] : 0;
        bucket_counts[2] = (block_elements > 0) ? shared_scan_10[block_elements - 1] : 0;
        bucket_counts[3] = (block_elements > 0) ? shared_scan_11[block_elements - 1] : 0;
        
        block_offsets[block_id * 4 + 0] = bucket_counts[0];
        block_offsets[block_id * 4 + 1] = bucket_counts[1];
        block_offsets[block_id * 4 + 2] = bucket_counts[2];
        block_offsets[block_id * 4 + 3] = bucket_counts[3];
    }
    __syncthreads();
    
    for (int i = 0; i < elements_per_thread; i++) {
        int shared_idx = thread_id + i * block_size;
        if (shared_idx < block_elements) {
            uint64_t key = shared_keys[shared_idx];
            uint32_t value = shared_values[shared_idx];
            uint32_t bits = shared_bits[shared_idx];
            
            uint32_t exc_scan;
            if (shared_idx == 0) {
                exc_scan = 0;
            } else {
                if (bits == 0) exc_scan = shared_scan_00[shared_idx - 1];
                else if (bits == 1) exc_scan = shared_scan_01[shared_idx - 1];
                else if (bits == 2) exc_scan = shared_scan_10[shared_idx - 1];
                else exc_scan = shared_scan_11[shared_idx - 1];
            }
            
            uint32_t local_pos;
            if (bits == 0) {
                local_pos = exc_scan;
            } else if (bits == 1) {
                local_pos = bucket_counts[0] + exc_scan;
            } else if (bits == 2) {
                local_pos = bucket_counts[0] + bucket_counts[1] + exc_scan;
            } else { // bits == 3
                local_pos = bucket_counts[0] + bucket_counts[1] + bucket_counts[2] + exc_scan;
            }
            
            thread_keys[i] = key;
            thread_values[i] = value;
            thread_bits[i] = local_pos;  // Store position instead of bits
        }
    }
    __syncthreads();
    
    for (int i = 0; i < elements_per_thread; i++) {
        int shared_idx = thread_id + i * block_size;
        if (shared_idx < block_elements) {
            uint32_t local_pos = thread_bits[i];
            shared_keys[local_pos] = thread_keys[i];
            shared_values[local_pos] = thread_values[i];
        }
    }
    __syncthreads();
    
    for (int i = 0; i < elements_per_thread; i++) {
        int shared_idx = thread_id + i * block_size;
        int global_idx = block_start + thread_id + i * block_size;
        
        if (global_idx < block_end && shared_idx < block_elements) {
            output_keys[global_idx] = shared_keys[shared_idx];
            output_values[global_idx] = shared_values[shared_idx];
        }
    }
}

__global__ void global_merge_2bit_multi_kernel(const uint64_t* __restrict__ input_keys,
                                              const uint32_t* __restrict__ input_values,
                                              uint64_t* __restrict__ output_keys,
                                              uint32_t* __restrict__ output_values,
                                              const uint32_t* __restrict__ block_offsets,
                                              const uint32_t* __restrict__ global_bucket_offsets,
                                              int num_elements,
                                              int num_blocks,
                                              int elements_per_thread) {
    
    const int block_size = blockDim.x;
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int elements_per_block = block_size * elements_per_thread;
    
    int block_start = block_id * elements_per_block;
    int block_end = min(block_start + elements_per_block, num_elements);
    int block_elements = block_end - block_start;
    
    uint32_t count_00 = block_offsets[block_id * 4 + 0];
    uint32_t count_01 = block_offsets[block_id * 4 + 1];
    uint32_t count_10 = block_offsets[block_id * 4 + 2];
    uint32_t count_11 = block_offsets[block_id * 4 + 3];
    
    uint32_t boundary_01 = count_00;
    uint32_t boundary_10 = count_00 + count_01;
    uint32_t boundary_11 = count_00 + count_01 + count_10;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int shared_idx = thread_id + i * block_size;
        int global_src_idx = block_start + thread_id + i * block_size;
        
        if (global_src_idx < block_end && shared_idx < block_elements) {
            uint64_t key = input_keys[global_src_idx];
            uint32_t value = input_values[global_src_idx];
            
            uint32_t bucket;
            uint32_t local_bucket_pos;
            
            if (shared_idx < boundary_01) {
                bucket = 0;
                local_bucket_pos = shared_idx;
            } else if (shared_idx < boundary_10) {
                bucket = 1;
                local_bucket_pos = shared_idx - boundary_01;
            } else if (shared_idx < boundary_11) {
                bucket = 2;
                local_bucket_pos = shared_idx - boundary_10;
            } else {
                bucket = 3;
                local_bucket_pos = shared_idx - boundary_11;
            }
            
            uint32_t global_bucket_start = global_bucket_offsets[block_id * 4 + bucket];
            uint32_t global_dest_idx = global_bucket_start + local_bucket_pos;
            
            output_keys[global_dest_idx] = key;
            output_values[global_dest_idx] = value;
        }
    }
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

void radix_sort_2bit_multi_coalesced(uint64_t* unsorted_keys, uint64_t* sorted_keys, 
                                     uint32_t* unsorted_values, uint32_t* sorted_values, 
                                     int num_elements) {
    
    const int block_size = 256;      
    const int elements_per_thread = 4; 
    const int elements_per_block = block_size * elements_per_thread;
    const int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* block_offsets;
    uint32_t* global_bucket_offsets;
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&block_offsets, num_blocks * 4 * sizeof(uint32_t));
    cudaMalloc(&global_bucket_offsets, num_blocks * 4 * sizeof(uint32_t));
    
    uint32_t* h_block_offsets = new uint32_t[num_blocks * 4];
    uint32_t* h_global_bucket_offsets = new uint32_t[num_blocks * 4];
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int bit = 0; bit < 64; bit += 2) {
        
        size_t shared_mem_size = elements_per_block * (sizeof(uint64_t) +     // keys
                                                       sizeof(uint32_t) +     // values  
                                                       sizeof(uint32_t) +     // bits
                                                       4 * sizeof(uint32_t)); // 4 scan arrays
        
        block_radix_sort_2bit_multi_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            sorted_keys, sorted_values, temp_keys, temp_values, 
            block_offsets, bit, num_elements, elements_per_thread);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_block_offsets, block_offsets, num_blocks * 4 * sizeof(uint32_t), 
                   cudaMemcpyDeviceToHost);
        
        compute_global_bucket_offsets_2bit(h_block_offsets, h_global_bucket_offsets, num_blocks);
        
        cudaMemcpy(global_bucket_offsets, h_global_bucket_offsets, 
                   num_blocks * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        global_merge_2bit_multi_kernel<<<num_blocks, block_size>>>(
            temp_keys, temp_values, sorted_keys, sorted_values,
            block_offsets, global_bucket_offsets, num_elements, num_blocks, elements_per_thread);
        
        cudaDeviceSynchronize();
    }
    
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(block_offsets);
    cudaFree(global_bucket_offsets);
    
    delete[] h_block_offsets;
    delete[] h_global_bucket_offsets;
}