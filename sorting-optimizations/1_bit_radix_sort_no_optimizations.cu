#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kogge_stone_exclusive_scan_ultra_warp_optimized_kernel(const uint32_t* __restrict__ input_array, uint32_t* __restrict__ output_array, uint32_t* __restrict__ block_sums, int num_of_elements) {
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ volatile uint32_t shared_output_array[];
    grid_group grid = this_grid();
    
    uint32_t val;
    if (threadIdx.x == 0) {
        val = 0; 
    } else {
        val = __ldg(&input_array[thread_id - 1]); 
    }
    
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    
    uint32_t temp;
    temp = __shfl_up_sync(0xffffffff, val, 1);  if (lane >= 1)  val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 2);  if (lane >= 2)  val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 4);  if (lane >= 4)  val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 8);  if (lane >= 8)  val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 16); if (lane >= 16) val += temp;
    
    if (lane == 31) {
        shared_output_array[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0 && threadIdx.x < num_warps) {
        uint32_t warp_sum = shared_output_array[threadIdx.x];
        
        temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 1);  if (threadIdx.x >= 1)  warp_sum += temp;
        temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 2);  if (threadIdx.x >= 2)  warp_sum += temp;
        temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 4);  if (threadIdx.x >= 4)  warp_sum += temp;
        temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 8);  if (threadIdx.x >= 8)  warp_sum += temp;
        temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 16); if (threadIdx.x >= 16) warp_sum += temp;
        
        shared_output_array[threadIdx.x] = warp_sum;
    }
    __syncthreads();
    
    if (warp_id > 0) {
        val += shared_output_array[warp_id - 1];
    }
    
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = val + __ldg(&input_array[thread_id]);
    }

    grid.sync();

    if (thread_id == 0) {
        const int num_blocks = gridDim.x;
        uint32_t running_sum = block_sums[0];
        
        if (num_blocks <= 8) {
            if (num_blocks > 1) { block_sums[1] += running_sum; running_sum = block_sums[1]; }
            if (num_blocks > 2) { block_sums[2] += running_sum; running_sum = block_sums[2]; }
            if (num_blocks > 3) { block_sums[3] += running_sum; running_sum = block_sums[3]; }
            if (num_blocks > 4) { block_sums[4] += running_sum; running_sum = block_sums[4]; }
            if (num_blocks > 5) { block_sums[5] += running_sum; running_sum = block_sums[5]; }
            if (num_blocks > 6) { block_sums[6] += running_sum; running_sum = block_sums[6]; }
            if (num_blocks > 7) { block_sums[7] += running_sum; }
        } else {
            #pragma unroll 4
            for (int i = 1; i < num_blocks; i++) {
                block_sums[i] += block_sums[i - 1];
            }
        }
    }
        
    grid.sync();

    const uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = val + block_sum_to_add;
}

__global__ void extract_bit_kernel(const uint64_t* __restrict__ keys, uint32_t* __restrict__ bits, int bit_position, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    bits[idx] = (keys[idx] >> bit_position) & 1;
}

__global__ void scatter_radix_kernel(const uint64_t* __restrict__ input_keys, const uint32_t* __restrict__ input_values, const uint32_t* __restrict__ bits, const uint32_t* __restrict__ prefix_sums, uint64_t* __restrict__ output_keys, uint32_t* __restrict__ output_values, uint32_t total_zeros, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    uint32_t bit = bits[idx];
    uint32_t new_pos;
    
    if (bit == 0) {
        new_pos = idx - prefix_sums[idx];
    } else {
        new_pos = total_zeros + prefix_sums[idx];
    }
    
    output_keys[new_pos] = input_keys[idx];
    output_values[new_pos] = input_values[idx];
}

void exclusive_scan_wrapper(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    int block_size = 1024;
    int shared_memory_size = block_size * sizeof(uint32_t);
    int num_blocks = (num_of_elements + block_size - 1) / block_size;
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    if (!deviceProp.cooperativeLaunch) {
        printf("Error: Device does not support cooperative kernel launch\n");
        return;
    }
    
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, 
                                                   kogge_stone_exclusive_scan_ultra_warp_optimized_kernel, 
                                                   block_size, 
                                                   shared_memory_size);
    
    int max_blocks = max_blocks_per_sm * deviceProp.multiProcessorCount;
    
    if (num_blocks > max_blocks) {
        printf("Error: Requested %d blocks, but maximum cooperative blocks is %d\n", num_blocks, max_blocks);
        return;
    }
    
    uint32_t* block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(uint32_t));
    
    void* args[] = {&input_array, &output_array, &block_sums, &num_of_elements};
    
    dim3 grid_size(num_blocks);
    dim3 block_size_dim(block_size);
    
    cudaLaunchCooperativeKernel(
        (void*)kogge_stone_exclusive_scan_ultra_warp_optimized_kernel,
        grid_size,
        block_size_dim,
        args,
        shared_memory_size
    );
    
    cudaFree(block_sums);
}

void radix_sort_1bit(uint64_t* unsorted_keys, uint64_t* sorted_keys, uint32_t* unsorted_values, uint32_t* sorted_values, int num_elements) {
    
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    uint32_t* bits;
    uint32_t* prefix_sums;
    uint64_t* temp_keys;
    uint32_t* temp_values;
    
    cudaMalloc(&bits, num_elements * sizeof(uint32_t));
    cudaMalloc(&prefix_sums, num_elements * sizeof(uint32_t));
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int bit = 0; bit < 64; bit++) {
        extract_bit_kernel<<<grid_size, block_size>>>(sorted_keys, bits, bit, num_elements);
        cudaDeviceSynchronize();
        
        exclusive_scan_wrapper(bits, prefix_sums, num_elements);
        cudaDeviceSynchronize();
        
        uint32_t total_ones_host;
        cudaMemcpy(&total_ones_host, &prefix_sums[num_elements - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        uint32_t last_bit_host;
        cudaMemcpy(&last_bit_host, &bits[num_elements - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        total_ones_host += last_bit_host; 
        uint32_t total_zeros = num_elements - total_ones_host;
        
        scatter_radix_kernel<<<grid_size, block_size>>>(sorted_keys, sorted_values, bits, prefix_sums, temp_keys, temp_values, total_zeros, num_elements);
        cudaDeviceSynchronize();
        
        uint64_t* temp_k = sorted_keys;
        sorted_keys = temp_keys;
        temp_keys = temp_k;
        
        uint32_t* temp_v = sorted_values;
        sorted_values = temp_values;
        temp_values = temp_v;
    }
    
    cudaFree(bits);
    cudaFree(prefix_sums);
    cudaFree(temp_keys);
    cudaFree(temp_values);
}