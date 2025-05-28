#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void block_exclusive_scan(uint32_t* shared_data, uint32_t tid, uint32_t block_size) {
    for (int d = 1; d < block_size; d <<= 1) {
        __syncthreads();
        if (tid % (2 * d) == 0) {
            shared_data[tid + 2*d - 1] += shared_data[tid + d - 1];
        }
    }
    
    if (tid == 0) {
        shared_data[block_size - 1] = 0;
    }
    
    for (int d = block_size >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid % (2 * d) == 0) {
            uint32_t temp = shared_data[tid + d - 1];
            shared_data[tid + d - 1] = shared_data[tid + 2*d - 1];
            shared_data[tid + 2*d - 1] += temp;
        }
    }
    __syncthreads();
}

__global__ void radix_sort_4bit_kernel(
    const uint64_t* __restrict__ input_keys,
    const uint32_t* __restrict__ input_values,
    uint64_t* __restrict__ output_keys,
    uint32_t* __restrict__ output_values,
    uint32_t* __restrict__ global_histograms,
    int bit_shift,
    int num_elements) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int elements_per_block = (num_elements + gridDim.x - 1) / gridDim.x;
    const int start_idx = bid * elements_per_block;
    const int end_idx = min(start_idx + elements_per_block, num_elements);
    
    __shared__ uint32_t local_hist[16];
    __shared__ uint32_t scan_data[16];
    __shared__ uint32_t global_offsets[16];
    
    if (tid < 16) {
        local_hist[tid] = 0;
    }
    __syncthreads();
    
    for (int i = start_idx + tid; i < end_idx; i += block_size) {
        uint32_t digit = (input_keys[i] >> bit_shift) & 0xF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();
    
    if (tid < 16) {
        scan_data[tid] = local_hist[tid];
    }
    __syncthreads();
    
    if (tid < 16) {
        block_exclusive_scan(scan_data, tid, 16);
    }
    __syncthreads();
    
    if (tid < 16) {
        global_offsets[tid] = atomicAdd(&global_histograms[tid], local_hist[tid]);
    }
    __syncthreads();
    
    for (int i = start_idx + tid; i < end_idx; i += block_size) {
        uint64_t key = input_keys[i];
        uint32_t value = input_values[i];
        uint32_t digit = (key >> bit_shift) & 0xF;
        
        uint32_t local_pos = 0;
        for (int j = start_idx; j < i; j++) {
            if (((input_keys[j] >> bit_shift) & 0xF) == digit) {
                local_pos++;
            }
        }
        
        uint32_t global_pos = global_offsets[digit] + scan_data[digit] + local_pos;
        output_keys[global_pos] = key;
        output_values[global_pos] = value;
    }
}

__global__ void radix_sort_4bit_coalesced_kernel(
    const uint64_t* __restrict__ input_keys,
    const uint32_t* __restrict__ input_values,
    uint64_t* __restrict__ output_keys,
    uint32_t* __restrict__ output_values,
    int bit_shift,
    int num_elements) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int global_tid = bid * block_size + tid;
    
    const int elements_per_block = (num_elements + gridDim.x - 1) / gridDim.x;
    const int start_idx = bid * elements_per_block;
    const int end_idx = min(start_idx + elements_per_block, num_elements);
    const int block_elements = end_idx - start_idx;
    
    __shared__ uint32_t local_hist[16];
    __shared__ uint32_t prefix_sums[16];
    __shared__ uint64_t shared_keys[1024]; 
    __shared__ uint32_t shared_values[1024];
    __shared__ uint32_t shared_digits[1024];
    
    if (tid < 16) {
        local_hist[tid] = 0;
    }
    __syncthreads();
    
    uint64_t my_key = 0;
    uint32_t my_value = 0;
    uint32_t my_digit = 0;
    
    if (global_tid < num_elements) {
        my_key = input_keys[global_tid];
        my_value = input_values[global_tid];
        my_digit = (my_key >> bit_shift) & 0xF;
        
        shared_keys[tid] = my_key;
        shared_values[tid] = my_value;
        shared_digits[tid] = my_digit;
        
        atomicAdd(&local_hist[my_digit], 1);
    }
    __syncthreads();
    
    if (tid < 16) {
        prefix_sums[tid] = 0;
        for (int i = 0; i < tid; i++) {
            prefix_sums[tid] += local_hist[i];
        }
    }
    __syncthreads();
    
    if (global_tid < num_elements) {
        uint32_t local_pos = prefix_sums[my_digit];
        
        for (int i = 0; i < tid; i++) {
            if (shared_digits[i] == my_digit) {
                local_pos++;
            }
        }
        
        uint32_t output_pos = start_idx + local_pos;
        if (output_pos < num_elements) {
            output_keys[output_pos] = my_key;
            output_values[output_pos] = my_value;
        }
    }
}

void radix_sort_4bit_optimized(
    uint64_t* unsorted_keys,
    uint64_t* sorted_keys,
    uint32_t* unsorted_values,
    uint32_t* sorted_values,
    int num_elements) {
    
    const int block_size = 256;
    const int grid_size = min(256, (num_elements + block_size - 1) / block_size);
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* global_histograms;
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&global_histograms, 16 * sizeof(uint32_t));
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int pass = 0; pass < 16; pass++) {
        int bit_shift = pass * 4;
        
        cudaMemset(global_histograms, 0, 16 * sizeof(uint32_t));
        
        radix_sort_4bit_kernel<<<grid_size, block_size>>>(
            sorted_keys, sorted_values,
            temp_keys, temp_values,
            global_histograms,
            bit_shift, num_elements);
        
        cudaDeviceSynchronize();
        
        uint64_t* temp_k = sorted_keys;
        sorted_keys = temp_keys;
        temp_keys = temp_k;
        
        uint32_t* temp_v = sorted_values;
        sorted_values = temp_values;
        temp_values = temp_v;
    }
    
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(global_histograms);
}

__global__ void count_digits_kernel(
    const uint64_t* __restrict__ keys,
    uint32_t* __restrict__ digit_counts,
    int bit_shift,
    int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint32_t shared_counts[16];
    
    if (threadIdx.x < 16) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (idx < num_elements) {
        uint32_t digit = (keys[idx] >> bit_shift) & 0xF;
        atomicAdd(&shared_counts[digit], 1);
    }
    __syncthreads();
    
    if (threadIdx.x < 16) {
        atomicAdd(&digit_counts[threadIdx.x], shared_counts[threadIdx.x]);
    }
}

__global__ void scatter_digits_kernel(
    const uint64_t* __restrict__ input_keys,
    const uint32_t* __restrict__ input_values,
    uint64_t* __restrict__ output_keys,
    uint32_t* __restrict__ output_values,
    const uint32_t* __restrict__ digit_offsets,
    int bit_shift,
    int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        uint64_t key = input_keys[idx];
        uint32_t value = input_values[idx];
        uint32_t digit = (key >> bit_shift) & 0xF;
        
        uint32_t prefix = 0;
        for (int i = 0; i < idx; i++) {
            if (((input_keys[i] >> bit_shift) & 0xF) == digit) {
                prefix++;
            }
        }
        
        uint32_t output_pos = digit_offsets[digit] + prefix;
        output_keys[output_pos] = key;
        output_values[output_pos] = value;
    }
}

void radix_sort_4bit_super_optimized(
    uint64_t* unsorted_keys,
    uint64_t* sorted_keys,
    uint32_t* unsorted_values,
    uint32_t* sorted_values,
    int num_elements) {
    
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* digit_counts;
    uint32_t* digit_offsets;
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&digit_counts, 16 * sizeof(uint32_t));
    cudaMalloc(&digit_offsets, 16 * sizeof(uint32_t));
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int pass = 0; pass < 16; pass++) {
        int bit_shift = pass * 4;
        
        cudaMemset(digit_counts, 0, 16 * sizeof(uint32_t));
        count_digits_kernel<<<grid_size, block_size>>>(sorted_keys, digit_counts, bit_shift, num_elements);
        
        uint32_t host_counts[16];
        cudaMemcpy(host_counts, digit_counts, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        uint32_t host_offsets[16];
        host_offsets[0] = 0;
        for (int i = 1; i < 16; i++) {
            host_offsets[i] = host_offsets[i-1] + host_counts[i-1];
        }
        
        cudaMemcpy(digit_offsets, host_offsets, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        scatter_digits_kernel<<<grid_size, block_size>>>(
            sorted_keys, sorted_values,
            temp_keys, temp_values,
            digit_offsets, bit_shift, num_elements);
        
        uint64_t* temp_k = sorted_keys;
        sorted_keys = temp_keys;
        temp_keys = temp_k;
        
        uint32_t* temp_v = sorted_values;
        sorted_values = temp_values;
        temp_values = temp_v;
    }
    
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(digit_counts);
    cudaFree(digit_offsets);
}