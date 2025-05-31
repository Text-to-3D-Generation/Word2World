#include "prepareSort.h"
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;



__global__ void block_radix_sort_2bit_kernel(const uint64_t* __restrict__ input_keys, 
                                            const uint32_t* __restrict__ input_values,
                                            uint64_t* __restrict__ output_keys,
                                            uint32_t* __restrict__ output_values,
                                            uint32_t* __restrict__ block_summationLocation,
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
        
        block_summationLocation[block_id * 4 + 0] = bucket_counts[0];
        block_summationLocation[block_id * 4 + 1] = bucket_counts[1];
        block_summationLocation[block_id * 4 + 2] = bucket_counts[2];
        block_summationLocation[block_id * 4 + 3] = bucket_counts[3];
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
                                        const uint32_t* __restrict__ block_summationLocation,
                                        const uint32_t* __restrict__ global_bucket_summationLocation,
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
    
    uint32_t count_00 = block_summationLocation[block_id * 4 + 0];
    uint32_t count_01 = block_summationLocation[block_id * 4 + 1];
    uint32_t count_10 = block_summationLocation[block_id * 4 + 2];
    uint32_t count_11 = block_summationLocation[block_id * 4 + 3];
    
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
    
    uint32_t global_bucket_start = global_bucket_summationLocation[block_id * 4 + bucket];
    uint32_t global_pos = global_bucket_start + local_bucket_pos;
    
    output_keys[global_pos] = key;
    output_values[global_pos] = value;
}
__device__ void emitTileKeyValuePairs(
    int THREADID,
    int2 tileRangeMin,       // x: minimumTileX, y: minimumTileY
    int2 tileRangeMax,       // x: maximumTileX, y: maximumTileY
    uint32_t depthBits,
    uint2* keys32bits,       // reinterpret_cast<uint2*>(unsortedKeys)
    uint32_t* valuesOut,     // unsortedGVals
    uint32_t outPtrStart     // starting write offset
) {
    int width = tileRangeMax.x - tileRangeMin.x;
    int height = tileRangeMax.y - tileRangeMin.y;
    int tileCount = width * height;

    for (int i = 0; i < tileCount; ++i) 
    {
        int dx = i % width;
        int dy = i / width;

        int tx = tileRangeMin.x + dx;
        int ty = tileRangeMin.y + dy;

        uint32_t tileId = ty * 50 + tx;

        uint32_t outPtr = outPtrStart + i;
        keys32bits[outPtr] = make_uint2(depthBits, tileId);  // {low, high}
        valuesOut[outPtr] = THREADID;
    }
}

void compute_global_bucket_summationLocation_2bit(uint32_t* block_summationLocation, uint32_t* global_bucket_summationLocation, 
                                       int num_blocks) {
    
    uint32_t total_counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < num_blocks; i++) {
        total_counts[0] += block_summationLocation[i * 4 + 0];
        total_counts[1] += block_summationLocation[i * 4 + 1];
        total_counts[2] += block_summationLocation[i * 4 + 2];
        total_counts[3] += block_summationLocation[i * 4 + 3];
    }
    
    uint32_t global_starts[4];
    global_starts[0] = 0;
    global_starts[1] = total_counts[0];
    global_starts[2] = total_counts[0] + total_counts[1];
    global_starts[3] = total_counts[0] + total_counts[1] + total_counts[2];
    
    uint32_t running_sums[4] = {0, 0, 0, 0};
    
    for (int i = 0; i < num_blocks; i++) {
        global_bucket_summationLocation[i * 4 + 0] = global_starts[0] + running_sums[0];
        global_bucket_summationLocation[i * 4 + 1] = global_starts[1] + running_sums[1];
        global_bucket_summationLocation[i * 4 + 2] = global_starts[2] + running_sums[2];
        global_bucket_summationLocation[i * 4 + 3] = global_starts[3] + running_sums[3];
        
        running_sums[0] += block_summationLocation[i * 4 + 0];
        running_sums[1] += block_summationLocation[i * 4 + 1];
        running_sums[2] += block_summationLocation[i * 4 + 2];
        running_sums[3] += block_summationLocation[i * 4 + 3];
    }
}

__global__ void fillMainArr(
    float2*     __restrict__ muuTwoD,            // [P] center in pixel space
    float*      __restrict__ zDistances,         // [P] depth (positive)
    int*        __restrict__ radss,              // [P] radius per Gaussian
    int         numGaussians,                    // total number of Gaussians
    uint32_t*   __restrict__ summationLocation,  // [P] prefix-sum of #tiles
    uint64_t*   __restrict__ unsortedGKeys,      // [N] output keys
    uint32_t*   __restrict__ unsortedGVals       // [N] output Gaussian IDs
)
                    
{
       int THREADID = blockIdx.x * blockDim.x + threadIdx.x;
    if (THREADID >= numGaussians)
    {
             return;          // out-of-range threads exit
    }
    if (radss[THREADID] <= 0) 
    {        
        return;       
       }   // invisible → nothing to duplicate

    uint32_t outPtr;
if (THREADID == 0)
{
    outPtr = 0;
}
else
{
    outPtr = summationLocation[THREADID - 1];
}



float2 center = muuTwoD[THREADID];
int rad = radss[THREADID];

auto clampTileCoord = [](int v) -> unsigned int {
    return static_cast<unsigned int>(std::min(50, std::max(0, v)));
};

int minimumTileX = clampTileCoord((int)floorf((center.x - rad) / 16.0f));
int maximumTileX = clampTileCoord((int)ceilf((center.x + rad) / 16.0f));
int minimumTileY = clampTileCoord((int)floorf((center.y - rad) / 16.0f));
int maximumTileY = clampTileCoord((int)ceilf((center.y + rad) / 16.0f));

uint2 topLeftGaussianCornerrad = { (unsigned int)minimumTileX, (unsigned int)minimumTileY };
uint2 bottomRightGaussianCornerrad = { (unsigned int)maximumTileX, (unsigned int)maximumTileY };
    auto* keys32bits = reinterpret_cast<uint2*>(unsortedGKeys);
       uint32_t depthBits = *reinterpret_cast<   uint32_t*>(&zDistances[THREADID]);
int2 tileMin = make_int2(topLeftGaussianCornerrad.x, topLeftGaussianCornerrad.y);
int2 tileMax = make_int2(bottomRightGaussianCornerrad.x, bottomRightGaussianCornerrad.y);

emitTileKeyValuePairs(
    THREADID,
    tileMin,
    tileMax,
    depthBits,
    keys32bits,
    unsortedGVals,
    outPtr
);

}

void radix_sort_2bit_coalesced(uint64_t* unsorted_keys, uint64_t* sorted_keys, 
                              uint32_t* unsorted_values, uint32_t* sorted_values, 
                              int num_elements) {
    
    const int block_size = 512; 
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    uint64_t* temp_keys;
    uint32_t* temp_values;
    uint32_t* block_summationLocation;           // 4 counts per block
    uint32_t* global_bucket_summationLocation;   // 4 global summationLocation per block
    
    cudaMalloc(&temp_keys, num_elements * sizeof(uint64_t));
    cudaMalloc(&temp_values, num_elements * sizeof(uint32_t));
    cudaMalloc(&block_summationLocation, num_blocks * 4 * sizeof(uint32_t));
    cudaMalloc(&global_bucket_summationLocation, num_blocks * 4 * sizeof(uint32_t));
    
    uint32_t* h_block_summationLocation = new uint32_t[num_blocks * 4];
    uint32_t* h_global_bucket_summationLocation = new uint32_t[num_blocks * 4];
    
    cudaMemcpy(sorted_keys, unsorted_keys, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sorted_values, unsorted_values, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    for (int bit = 0; bit < 64; bit += 2) {
        
        size_t shared_mem_size = block_size * (sizeof(uint64_t) + sizeof(uint32_t) + // keys + values
                                              sizeof(uint32_t) +                      // bits
                                              4 * sizeof(uint32_t));                  // 4 scan arrays
        
        block_radix_sort_2bit_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            sorted_keys, sorted_values, temp_keys, temp_values, 
            block_summationLocation, bit, num_elements);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_block_summationLocation, block_summationLocation, num_blocks * 4 * sizeof(uint32_t), 
                   cudaMemcpyDeviceToHost);
        
        compute_global_bucket_summationLocation_2bit(h_block_summationLocation, h_global_bucket_summationLocation, num_blocks);
        
        cudaMemcpy(global_bucket_summationLocation, h_global_bucket_summationLocation, 
                   num_blocks * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        global_merge_2bit_kernel<<<num_blocks, block_size>>>(
            temp_keys, temp_values, sorted_keys, sorted_values,
            block_summationLocation, global_bucket_summationLocation, num_elements, num_blocks);
        
        cudaDeviceSynchronize();
    }
    
    cudaFree(temp_keys);
    cudaFree(temp_values);
    cudaFree(block_summationLocation);
    cudaFree(global_bucket_summationLocation);
    
    delete[] h_block_summationLocation;
    delete[] h_global_bucket_summationLocation;
}

void prepareAndSort(
       std::function<char*(size_t)> BBuffer,
   
   
       int numGaussians,
    uint32_t* locationsOfPointsWRT,
   
   
          float2* means2D,
    float* zDistances,
    int* radss,
    
    
    int& totalGaussianInstances,
    uint32_t*& MainGaussianIdArray,uint32_t*& unsortedPoints,
   
    uint64_t*& keysArray,
    uint64_t*& unsortedPointsKeys)
{
    // 1) Read last offset to get #rendered
    cudaMemcpy(
         &totalGaussianInstances,
          locationsOfPointsWRT + (numGaussians - 1),
        
          sizeof(int), cudaMemcpyDeviceToHost
    );

    auto align128 = [](size_t x) { return (x + 127) & ~static_cast<size_t>(127); };

    size_t offset = 0;
            offset = align128(offset) + totalGaussianInstances * sizeof(uint32_t);  // MainGaussianIdArray
    offset = align128(offset) + totalGaussianInstances * sizeof(uint32_t);  // unsortedPoints

        
                offset = align128(offset) + totalGaussianInstances * sizeof(uint64_t);  // unsortedKeys
         offset = align128(offset) + totalGaussianInstances * sizeof(uint64_t);  // keys
    size_t temp_bytes = 0;
      offset = align128(offset) + temp_bytes;
    size_t chunk_size = offset + 128;
    
    
    char* chunk = BBuffer(chunk_size);

    std::uintptr_t addr;

    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
    
    
        
            MainGaussianIdArray = reinterpret_cast<uint32_t*>(addr);
    chunk = reinterpret_cast<char*>(MainGaussianIdArray + totalGaussianInstances);
    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
    unsortedPoints = reinterpret_cast<uint32_t*>(addr);
    chunk = reinterpret_cast<char*>(unsortedPoints + totalGaussianInstances);
    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
                unsortedPointsKeys = reinterpret_cast<uint64_t*>(addr);
    chunk = reinterpret_cast<char*>(unsortedPointsKeys + totalGaussianInstances);
                addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
            keysArray = reinterpret_cast<uint64_t*>(addr);
    chunk = reinterpret_cast<char*>(keysArray + totalGaussianInstances);

    // Launch kernel to fill unsorted arrays
   fillMainArr<<<(numGaussians + 255) / 256, 256>>>(
    means2D, zDistances,radss,numGaussians,locationsOfPointsWRT,    unsortedPointsKeys,      
    unsortedPoints          
);


    // Sort using your custom radix sort
    radix_sort_2bit_coalesced(unsortedPointsKeys, keysArray,
        unsortedPoints, MainGaussianIdArray,
        totalGaussianInstances
    );
}

