#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

#include "brent_kung_scan.cu"
using highres_clock = std::chrono::high_resolution_clock;

void brent_kung_inclusive_scan_shared_memory(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    int block_size = 512;
    int elements_per_block = 2 * block_size;
    int shared_memory_size = elements_per_block * sizeof(uint32_t);
    int num_blocks = (num_of_elements + elements_per_block - 1) / elements_per_block;

    uint32_t* block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(uint32_t));
    brent_kung_inclusive_scan_shared_memory_kernel<<<num_blocks, block_size, shared_memory_size>>>(input_array, output_array, block_sums, num_of_elements);
    if (num_blocks > 1) {
        brent_kung_inclusive_scan_shared_memory(block_sums, block_sums, num_blocks);
        add_block_sums_to_inclusive_output_kernel<<<num_blocks, block_size>>>(output_array, block_sums, num_of_elements);
    }
    cudaDeviceSynchronize();
    cudaFree(block_sums);
}

void brent_kung_inclusive_scan_cpu(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
    output_array[0] = input_array[0];
    for (int i = 1; i < num_of_elements; i++) 
        output_array[i] = output_array[i-1] + input_array[i];
}

int main()
{
    std::srand(42);

    const int lower_bound = 5000;
    const int upper_bound = 150000;
    const int step_size  = 10000;

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
        brent_kung_inclusive_scan_cpu(host_input_array, reference_array, n);
        auto t1 = highres_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        brent_kung_inclusive_scan_shared_memory(device_input_array,
                                                    device_output_array, n);

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