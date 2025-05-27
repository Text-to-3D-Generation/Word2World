#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define COARSENING_FACTOR 8

__global__ void brent_kung_inclusive_scan_shared_memory_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int segment_start_index = 2 * blockIdx.x * blockDim.x;
    if (segment_start_index >= num_of_elements) return;
    int thread_id = segment_start_index + threadIdx.x;
    extern __shared__ uint32_t shared_output_array[];
    shared_output_array[threadIdx.x] = (thread_id < num_of_elements) ? input_array[thread_id] : 0;
    shared_output_array[threadIdx.x + blockDim.x] = (thread_id + blockDim.x < num_of_elements) ? input_array[thread_id + blockDim.x] : 0;
    __syncthreads();
    
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        int equivalent_index = 2 * stride * (threadIdx.x + 1) - 1;
        if (equivalent_index < 2 * blockDim.x) 
            shared_output_array[equivalent_index] += shared_output_array[equivalent_index - stride];
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int equivalent_index = 2 * stride * (threadIdx.x + 1) - 1;
        if (equivalent_index + stride < 2 * blockDim.x) 
            shared_output_array[equivalent_index + stride] += shared_output_array[equivalent_index];
        __syncthreads();
    }

    if (threadIdx.x == 0) 
        block_sums[blockIdx.x] = shared_output_array[2 * blockDim.x - 1];

    if (thread_id < num_of_elements)
        output_array[thread_id] = shared_output_array[threadIdx.x];
    
    if (thread_id + blockDim.x < num_of_elements)
        output_array[thread_id + blockDim.x] = shared_output_array[threadIdx.x + blockDim.x];
}

__global__ void add_block_sums_to_inclusive_output_kernel(uint32_t* output_array, const uint32_t* block_sums, int n)
{
    uint32_t offset = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];

    int first  = 2 * blockIdx.x * blockDim.x + threadIdx.x;  
    int second = first + blockDim.x; 

    if (first  < n) output_array[first]  += offset;
    if (second < n) output_array[second] += offset;
}
