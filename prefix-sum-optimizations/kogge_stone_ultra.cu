#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kogge_stone_inclusive_scan_ultra_warp_optimized_kernel(const uint32_t* __restrict__ input_array, uint32_t* __restrict__ output_array, uint32_t* __restrict__ block_sums, int num_of_elements) {
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ volatile uint32_t shared_output_array[];
    grid_group grid = this_grid();
    
    const uint32_t input_val = __ldg(&input_array[thread_id]);
    uint32_t val = input_val;
    
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
        block_sums[blockIdx.x] = val;
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