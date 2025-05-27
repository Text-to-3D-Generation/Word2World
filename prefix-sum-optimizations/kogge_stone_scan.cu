#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kogge_stone_inclusive_scan_no_optimizations_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    output_array[thread_id] = input_array[thread_id];
    __syncthreads();
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = output_array[thread_id - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            output_array[thread_id] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = output_array[thread_id];
}

__global__ void kogge_stone_inclusive_scan_shared_memory_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    extern __shared__ uint32_t shared_output_array[];
    shared_output_array[threadIdx.x] = input_array[thread_id];
    __syncthreads();
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = shared_output_array[threadIdx.x - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            shared_output_array[threadIdx.x] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = shared_output_array[threadIdx.x];

    output_array[thread_id] = shared_output_array[threadIdx.x];
}

__global__ void kogge_stone_inclusive_scan_global_sync_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ uint32_t shared_output_array[];
    grid_group grid = this_grid();
    
    shared_output_array[threadIdx.x] = input_array[thread_id];
    __syncthreads();
    
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = shared_output_array[threadIdx.x - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            shared_output_array[threadIdx.x] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = shared_output_array[threadIdx.x];

    grid.sync();

    if (thread_id == 0) 
        for (int i = 1; i < gridDim.x; i++) 
            block_sums[i] += block_sums[i - 1];
        
    grid.sync();

    uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = shared_output_array[threadIdx.x] + block_sum_to_add;
}

__global__ void kogge_stone_inclusive_scan_global_sync_double_buffering_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ uint32_t shared_output_array[];
    uint32_t* input_buffer = shared_output_array;
    uint32_t* output_buffer = shared_output_array + blockDim.x;
    grid_group grid = this_grid();
    
    input_buffer[threadIdx.x] = input_array[thread_id];
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            output_buffer[threadIdx.x] = input_buffer[threadIdx.x] + input_buffer[threadIdx.x - stride];
        else
            output_buffer[threadIdx.x] = input_buffer[threadIdx.x];
        __syncthreads(); 
        uint32_t* temp_ptr = input_buffer;
        input_buffer = output_buffer;
        output_buffer = temp_ptr;
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = input_buffer[threadIdx.x];

    grid.sync();

    if (thread_id == 0) 
        for (int i = 1; i < gridDim.x; i++) 
            block_sums[i] += block_sums[i - 1];
        
    grid.sync();

    uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = input_buffer[threadIdx.x] + block_sum_to_add;
}

__global__ void kogge_stone_inclusive_scan_warp_optimized_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ uint32_t shared_output_array[];
    grid_group grid = this_grid();
    
    uint32_t val = input_array[thread_id];
    
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t temp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += temp;
    }
    
    if (lane == 31) shared_output_array[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        uint32_t warp_sum = (threadIdx.x < (blockDim.x >> 5)) ? shared_output_array[threadIdx.x] : 0;
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            uint32_t temp = __shfl_up_sync(0xffffffff, warp_sum, offset);
            if (threadIdx.x >= offset) warp_sum += temp;
        }
        
        shared_output_array[threadIdx.x] = warp_sum;
    }
    __syncthreads();
    
    if (warp_id > 0) val += shared_output_array[warp_id - 1];
    
    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = val;

    grid.sync();

    if (thread_id == 0) {
        #pragma unroll 8
        for (int i = 1; i < gridDim.x; i++) 
            block_sums[i] += block_sums[i - 1];
    }
        
    grid.sync();

    uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = val + block_sum_to_add;
}

__global__ void kogge_stone_exclusive_scan_no_optimizations_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    if (threadIdx.x == 0)
        output_array[thread_id] = 0;
    else
        output_array[thread_id] = input_array[thread_id - 1];
    __syncthreads();
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = output_array[thread_id - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            output_array[thread_id] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = output_array[thread_id] + input_array[thread_id];
}

__global__ void kogge_stone_exclusive_scan_shared_memory_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    extern __shared__ uint32_t shared_output_array[];
    if (threadIdx.x == 0)
        shared_output_array[threadIdx.x] = 0;
    else
        shared_output_array[threadIdx.x] = input_array[thread_id - 1];
    __syncthreads();
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = shared_output_array[threadIdx.x - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            shared_output_array[threadIdx.x] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = shared_output_array[threadIdx.x] + input_array[thread_id];

    output_array[thread_id] = shared_output_array[threadIdx.x];
}

__global__ void kogge_stone_exclusive_scan_global_sync_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ uint32_t shared_output_array[];
    grid_group grid = this_grid();
    
    if (threadIdx.x == 0)
        shared_output_array[threadIdx.x] = 0;
    else
        shared_output_array[threadIdx.x] = input_array[thread_id - 1];
    __syncthreads();
    
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = shared_output_array[threadIdx.x - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            shared_output_array[threadIdx.x] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = shared_output_array[threadIdx.x] + input_array[thread_id];

    grid.sync();

    if (thread_id == 0)
        for (int i = 1; i < gridDim.x; i++) 
            block_sums[i] += block_sums[i - 1];
        
    grid.sync();

    uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = shared_output_array[threadIdx.x] + block_sum_to_add;
}

__global__ void kogge_stone_exclusive_scan_global_sync_double_buffering_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ uint32_t shared_output_array[];
    uint32_t* input_buffer = shared_output_array;
    uint32_t* output_buffer = shared_output_array + blockDim.x;
    grid_group grid = this_grid();
    
    if (threadIdx.x == 0)
        input_buffer[threadIdx.x] = 0;
    else
        input_buffer[threadIdx.x] = input_array[thread_id - 1];
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            output_buffer[threadIdx.x] = input_buffer[threadIdx.x] + input_buffer[threadIdx.x - stride];
        else
            output_buffer[threadIdx.x] = input_buffer[threadIdx.x];
        __syncthreads(); 
        uint32_t* temp_ptr = input_buffer;
        input_buffer = output_buffer;
        output_buffer = temp_ptr;
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = input_buffer[threadIdx.x] + input_array[thread_id];

    grid.sync();

    if (thread_id == 0)
        for (int i = 1; i < gridDim.x; i++) 
            block_sums[i] += block_sums[i - 1];
        
    grid.sync();

    uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = input_buffer[threadIdx.x] + block_sum_to_add;
}

__global__ void kogge_stone_exclusive_scan_warp_optimized_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    
    extern __shared__ uint32_t shared_output_array[];
    grid_group grid = this_grid();
    
    uint32_t val;
    if (threadIdx.x == 0) {
        val = 0;  
    } else {
        val = input_array[thread_id - 1];  
    }
    
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t temp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += temp;
    }
    
    if (lane == 31) shared_output_array[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        uint32_t warp_sum = (threadIdx.x < (blockDim.x >> 5)) ? shared_output_array[threadIdx.x] : 0;
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            uint32_t temp = __shfl_up_sync(0xffffffff, warp_sum, offset);
            if (threadIdx.x >= offset) warp_sum += temp;
        }
        
        shared_output_array[threadIdx.x] = warp_sum;
    }
    __syncthreads();
    
    if (warp_id > 0) val += shared_output_array[warp_id - 1];
    
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = val + input_array[thread_id];
    }

    grid.sync();

    if (thread_id == 0) {
        #pragma unroll 8
        for (int i = 1; i < gridDim.x; i++) 
            block_sums[i] += block_sums[i - 1];
    }
        
    grid.sync();

    uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
    output_array[thread_id] = val + block_sum_to_add;
}

__global__ void add_block_sums_to_inclusive_output_kernel(uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements || blockIdx.x == 0) return;
    output_array[thread_id] += block_sums[blockIdx.x - 1];
}

__global__ void add_block_sums_to_exclusive_output_kernel(uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements || blockIdx.x == 0) return;
    output_array[thread_id] += block_sums[blockIdx.x];
}