#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

#include "kogge_stone_scan.cu"
using highres_clock = std::chrono::high_resolution_clock;

void kogge_stone_inclusive_scan_no_optimizations(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    int block_size = 128;
    int num_blocks = (num_of_elements + block_size - 1) / block_size;
    uint32_t* block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(uint32_t));
    kogge_stone_inclusive_scan_no_optimizations_kernel<<<num_blocks, block_size>>>(input_array, output_array, block_sums, num_of_elements);
    if (num_blocks > 1) {
        kogge_stone_inclusive_scan_no_optimizations(block_sums, block_sums, num_blocks);
        add_block_sums_to_inclusive_output_kernel<<<num_blocks, block_size>>>(output_array, block_sums, num_of_elements);
    }
    cudaDeviceSynchronize();
    cudaFree(block_sums);
}

void kogge_stone_inclusive_scan_global_sync(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kogge_stone_inclusive_scan_global_sync_kernel, block_size, shared_memory_size);
    
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
    
    cudaError_t result = cudaLaunchCooperativeKernel((void*)kogge_stone_inclusive_scan_global_sync_kernel, grid_size, block_size_dim, args, shared_memory_size);
    
    if (result != cudaSuccess) {
        printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(result));
    }
    
    cudaDeviceSynchronize();
    cudaFree(block_sums);
}

void kogge_stone_inclusive_scan_cpu(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    output_array[0] = input_array[0];
    for (int i = 1; i < num_of_elements; i++) 
        output_array[i] = output_array[i-1] + input_array[i];
}

int main()
{
    std::srand(42);

    const int lower_bound = 5000;
    const int upper_bound = 15000;
    const int step_size  = 1000;

    printf("N        CPU-ms   GPU-ms   Match?\n");
    printf("----------------------------------\n");

    for (int n = lower_bound; n <= upper_bound; n += step_size)
    {
        auto* host_input_array  = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* host_output_array = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* reference_array   = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));

        for (int i = 0; i < n; ++i)
            host_input_array[i] = std::rand() % 1729;

        uint32_t *device_input_array, *device_output_array;
        cudaMalloc(&device_input_array,  n * sizeof(uint32_t));
        cudaMalloc(&device_output_array, n * sizeof(uint32_t));
        cudaMemcpy(device_input_array, host_input_array, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        auto t0 = highres_clock::now();
        kogge_stone_inclusive_scan_cpu(host_input_array, reference_array, n);
        auto t1 = highres_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        kogge_stone_inclusive_scan_global_sync(device_input_array, device_output_array, n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(host_output_array, device_output_array, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        bool match = true;
        int mismatch_idx = -1;
        uint32_t expected = 0;
        uint32_t got = 0;
        for (int i = 0; i < n; ++i) {
            if (host_output_array[i] != reference_array[i]) {
                match = false;
                mismatch_idx = i;
                expected = reference_array[i];
                got = host_output_array[i];
                break;                
            }
        }

        printf("%-7d  %7.3f  %7.3f  %s", n, cpu_ms, gpu_ms, match ? "yes" : "NO");

        if (!match)
            printf("   (mismatch @ %d: expected %u, got %u)", mismatch_idx, expected, got);

        printf("\n");

        free(host_input_array);
        free(host_output_array);
        free(reference_array);
        cudaFree(device_input_array);
        cudaFree(device_output_array);
    }

    return 0;
}