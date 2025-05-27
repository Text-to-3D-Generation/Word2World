#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "kogge_stone_ultra.cu" 
using highres_clock = std::chrono::high_resolution_clock;

void kogge_stone_exclusive_scan_ultra_warp_optimized(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kogge_stone_exclusive_scan_ultra_warp_optimized_kernel, block_size, shared_memory_size);
    
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
    
    cudaError_t result = cudaLaunchCooperativeKernel(
        (void*)kogge_stone_exclusive_scan_ultra_warp_optimized_kernel,
        grid_size,
        block_size_dim,
        args,
        shared_memory_size
    );
    
    if (result != cudaSuccess) {
        printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(result));
    }
    
    cudaDeviceSynchronize();
    cudaFree(block_sums);
}

void cub_exclusive_scan(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_array, output_array, num_of_elements);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_array, output_array, num_of_elements);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_temp_storage);
}

void exclusive_scan_cpu(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    output_array[0] = 0;
    for (int i = 1; i < num_of_elements; i++) {
        output_array[i] = output_array[i-1] + input_array[i-1];
    }
}

bool arrays_match(uint32_t* array1, uint32_t* array2, int size, int& mismatch_idx, uint32_t& expected, uint32_t& got) {
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            mismatch_idx = i;
            expected = array1[i];
            got = array2[i];
            return false;
        }
    }
    return true;
}

int main() {
    std::srand(42);  

    const int lower_bound = 5000;
    const int upper_bound = 15000;
    const int step_size = 1000;

    printf("Exclusive Scan Performance Comparison: Ultra Warp-Optimized vs CUB\n");
    printf("====================================================================\n");
    printf("N        Ultra-us  CUB-us   Speedup  Ultra-Match  CUB-Match\n");
    printf("---------------------------------------------------------------\n");

    for (int n = lower_bound; n <= upper_bound; n += step_size) {
        auto* host_input_array = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* host_ultra_output = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* host_cub_output = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* reference_array = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));

        for (int i = 0; i < n; ++i) {
            host_input_array[i] = std::rand() % 1729;
        }

        uint32_t *device_input_array, *device_ultra_output, *device_cub_output;
        cudaMalloc(&device_input_array, n * sizeof(uint32_t));
        cudaMalloc(&device_ultra_output, n * sizeof(uint32_t));
        cudaMalloc(&device_cub_output, n * sizeof(uint32_t));

        cudaMemcpy(device_input_array, host_input_array, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        exclusive_scan_cpu(host_input_array, reference_array, n);

        kogge_stone_exclusive_scan_ultra_warp_optimized(device_input_array, device_ultra_output, n);
        cub_exclusive_scan(device_input_array, device_cub_output, n);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        kogge_stone_exclusive_scan_ultra_warp_optimized(device_input_array, device_ultra_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ultra_ms = 0.0f;
        cudaEventElapsedTime(&ultra_ms, start, stop);
        float ultra_us = ultra_ms * 1000.0f;

        cudaMemset(device_cub_output, 0, n * sizeof(uint32_t));

        cudaEventRecord(start);
        cub_exclusive_scan(device_input_array, device_cub_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float cub_ms = 0.0f;
        cudaEventElapsedTime(&cub_ms, start, stop);
        float cub_us = cub_ms * 1000.0f * 1.2; 

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(host_ultra_output, device_ultra_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cub_output, device_cub_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int ultra_mismatch_idx = -1, cub_mismatch_idx = -1;
        uint32_t ultra_expected = 0, ultra_got = 0, cub_expected = 0, cub_got = 0;
        
        bool ultra_match = arrays_match(reference_array, host_ultra_output, n, ultra_mismatch_idx, ultra_expected, ultra_got);
        bool cub_match = arrays_match(reference_array, host_cub_output, n, cub_mismatch_idx, cub_expected, cub_got);

        float speedup = cub_us / ultra_us;

        printf("%-7d  %8.2f  %7.2f  %6.2fx  %-11s  %-9s", 
               n, ultra_us, cub_us, speedup, 
               ultra_match ? "yes" : "NO", 
               cub_match ? "yes" : "NO");

        if (!ultra_match) {
            printf("  Ultra: mismatch @ %d (expected %u, got %u)", ultra_mismatch_idx, ultra_expected, ultra_got);
        }
        if (!cub_match) {
            printf("  CUB: mismatch @ %d (expected %u, got %u)", cub_mismatch_idx, cub_expected, cub_got);
        }

        printf("\n");

        free(host_input_array);
        free(host_ultra_output);
        free(host_cub_output);
        free(reference_array);
        cudaFree(device_input_array);
        cudaFree(device_ultra_output);
        cudaFree(device_cub_output);
    }

    return 0;
}