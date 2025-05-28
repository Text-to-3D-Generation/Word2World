#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "4_bit_radix_sort_no_optimizations.cu"  // Include the 4-bit radix sort implementation
using highres_clock = std::chrono::high_resolution_clock;

// CUB radix sort wrapper
void cub_radix_sort(
    uint64_t* input_keys,
    uint64_t* output_keys,
    uint32_t* input_values,
    uint32_t* output_values,
    int num_elements) {
    
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, 
        input_keys, output_keys,
        input_values, output_values,
        num_elements);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        input_keys, output_keys,
        input_values, output_values,
        num_elements);
    
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(d_temp_storage);
}

// CPU reference implementation for verification
void radix_sort_cpu_reference(
    uint64_t* keys, 
    uint32_t* values, 
    int num_elements) {
    
    // Create key-value pairs for easier sorting
    std::vector<std::pair<uint64_t, uint32_t>> pairs(num_elements);
    for (int i = 0; i < num_elements; i++) {
        pairs[i] = {keys[i], values[i]};
    }
    
    // Sort by key
    std::sort(pairs.begin(), pairs.end());
    
    // Extract back to arrays
    for (int i = 0; i < num_elements; i++) {
        keys[i] = pairs[i].first;
        values[i] = pairs[i].second;
    }
}

// Utility function to check if arrays are sorted and match
bool verify_correctness(
    uint64_t* keys1, uint32_t* values1,
    uint64_t* keys2, uint32_t* values2,
    uint64_t* cpu_keys, uint32_t* cpu_values,
    int num_elements,
    const char* name1, const char* name2,
    bool& sorted1, bool& sorted2, bool& match12, bool& match1_cpu, bool& match2_cpu) {
    
    // Check if first array is sorted
    sorted1 = true;
    for (int i = 1; i < num_elements; i++) {
        if (keys1[i] < keys1[i-1]) {
            sorted1 = false;
            break;
        }
    }
    
    // Check if second array is sorted
    sorted2 = true;
    for (int i = 1; i < num_elements; i++) {
        if (keys2[i] < keys2[i-1]) {
            sorted2 = false;
            break;
        }
    }
    
    // Check if both arrays match each other
    match12 = true;
    for (int i = 0; i < num_elements; i++) {
        if (keys1[i] != keys2[i] || values1[i] != values2[i]) {
            match12 = false;
            break;
        }
    }
    
    // Check if first array matches CPU reference
    match1_cpu = true;
    for (int i = 0; i < num_elements; i++) {
        if (keys1[i] != cpu_keys[i] || values1[i] != cpu_values[i]) {
            match1_cpu = false;
            break;
        }
    }
    
    // Check if second array matches CPU reference
    match2_cpu = true;
    for (int i = 0; i < num_elements; i++) {
        if (keys2[i] != cpu_keys[i] || values2[i] != cpu_values[i]) {
            match2_cpu = false;
            break;
        }
    }
    
    return sorted1 && sorted2 && match12 && match1_cpu && match2_cpu;
}

// Generate test data with various distributions
void generate_test_data(uint64_t* keys, uint32_t* values, int num_elements, int test_case) {
    std::srand(42 + test_case);
    
    switch (test_case) {
        case 0: // Random data
            for (int i = 0; i < num_elements; i++) {
                keys[i] = ((uint64_t)std::rand() << 32) | std::rand();
                values[i] = i;
            }
            break;
            
        case 1: // Already sorted
            for (int i = 0; i < num_elements; i++) {
                keys[i] = i;
                values[i] = i;
            }
            break;
            
        case 2: // Reverse sorted
            for (int i = 0; i < num_elements; i++) {
                keys[i] = num_elements - 1 - i;
                values[i] = i;
            }
            break;
            
        case 3: // Many duplicates (stress test)
            for (int i = 0; i < num_elements; i++) {
                keys[i] = std::rand() % (num_elements / 10);
                values[i] = i;
            }
            break;
            
        case 4: // Sparse high bits (tests radix efficiency)
            for (int i = 0; i < num_elements; i++) {
                if (i % 100 == 0) {
                    keys[i] = ((uint64_t)1 << 63) | (std::rand() % 1000);
                } else {
                    keys[i] = std::rand() % 1000;
                }
                values[i] = i;
            }
            break;
            
        case 5: // Power of 2 distributed
            for (int i = 0; i < num_elements; i++) {
                int power = std::rand() % 20;
                keys[i] = 1ULL << power;
                values[i] = i;
            }
            break;
            
        default: // Random with limited range
            for (int i = 0; i < num_elements; i++) {
                keys[i] = std::rand() % 100000;
                values[i] = i;
            }
            break;
    }
}

int main() {
    // Test different sizes and implementations
    const int test_sizes[] = {1000, 5000, 10000, 15000, 20000};
    const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    const char* test_names[] = {"Random", "Sorted", "Reverse", "Duplicates", "Sparse-Hi", "Power-2", "Limited"};
    const int num_test_types = 7;
    
    // Test all three 4-bit implementations
    const char* impl_names[] = {"4bit-Basic", "4bit-Super", "4bit-Coalesced"};
    
    printf("4-bit Radix Sort Performance Comparison vs CUB\n");
    printf("==============================================\n");
    printf("Testing multiple 4-bit implementations against CUB DeviceRadixSort\n\n");
    
    for (int test_type = 0; test_type < num_test_types; test_type++) {
        printf("\n=== Test Case: %s Data ===\n", test_names[test_type]);
        printf("N        4bit-Super-ms  CUB-ms   Speedup  Sorted?  Match?  CPU-Match?\n");
        printf("--------------------------------------------------------------------\n");
        
        for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
            int n = test_sizes[size_idx];
            
            // Skip very large sizes for slower implementations
            if (n > 100000 && test_type > 3) continue;
            
            // Allocate host memory
            auto* host_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* host_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cpu_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cpu_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* custom_result_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* custom_result_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cub_result_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cub_result_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            
            // Generate test data
            generate_test_data(host_keys, host_values, n, test_type);
            
            // Copy for CPU reference
            memcpy(cpu_keys, host_keys, n * sizeof(uint64_t));
            memcpy(cpu_values, host_values, n * sizeof(uint32_t));
            
            // Generate CPU reference (not timed, just for verification)
            radix_sort_cpu_reference(cpu_keys, cpu_values, n);
            
            // Allocate device memory
            uint64_t *d_input_keys, *d_custom_keys, *d_cub_keys;
            uint32_t *d_input_values, *d_custom_values, *d_cub_values;
            
            cudaMalloc(&d_input_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_custom_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_cub_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_input_values, n * sizeof(uint32_t));
            cudaMalloc(&d_custom_values, n * sizeof(uint32_t));
            cudaMalloc(&d_cub_values, n * sizeof(uint32_t));
            
            // Copy input to device
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            // Warm up both implementations
            radix_sort_4bit_super_optimized(d_input_keys, d_custom_keys, d_input_values, d_custom_values, n);
            cub_radix_sort(d_input_keys, d_cub_keys, d_input_values, d_cub_values, n);
            cudaDeviceSynchronize();
            
            // Reset input data for timing
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            // Time 4-bit Super Optimized Radix Sort
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            radix_sort_4bit_super_optimized(d_input_keys, d_custom_keys, d_input_values, d_custom_values, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float custom_ms;
            cudaEventElapsedTime(&custom_ms, start, stop);
            
            // Reset input data for CUB test
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            // Time CUB Radix Sort
            cudaEventRecord(start);
            cub_radix_sort(d_input_keys, d_cub_keys, d_input_values, d_cub_values, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float cub_ms;
            cudaEventElapsedTime(&cub_ms, start, stop);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            // Copy results back to host
            cudaMemcpy(custom_result_keys, d_custom_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(custom_result_values, d_custom_values, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(cub_result_keys, d_cub_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(cub_result_values, d_cub_values, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            // Verify correctness
            bool sorted_custom, sorted_cub, match_implementations, match_custom_cpu, match_cub_cpu;
            bool all_correct = verify_correctness(
                custom_result_keys, custom_result_values,
                cub_result_keys, cub_result_values,
                cpu_keys, cpu_values,
                n, "4bit-Super", "CUB",
                sorted_custom, sorted_cub, match_implementations, match_custom_cpu, match_cub_cpu);
            
            // Calculate speedup
            double speedup = cub_ms / custom_ms;
            
            // Print results
            printf("%-7d  %12.3f  %7.3f  %6.2fx  %-7s  %-6s  %-10s", 
                   n, custom_ms, cub_ms, speedup,
                   sorted_custom ? "yes" : "NO",
                   match_implementations ? "yes" : "NO",
                   match_custom_cpu ? "yes" : "NO");
            
            // Print issues if any
            if (!sorted_custom) {
                printf("  [4bit not sorted]");
            }
            if (!match_implementations) {
                printf("  [4bit≠CUB]");
            }
            if (!match_custom_cpu) {
                printf("  [4bit≠CPU]");
            }
            
            printf("\n");
            
            // Clean up
            free(host_keys);
            free(host_values);
            free(cpu_keys);
            free(cpu_values);
            free(custom_result_keys);
            free(custom_result_values);
            free(cub_result_keys);
            free(cub_result_values);
            
            cudaFree(d_input_keys);
            cudaFree(d_custom_keys);
            cudaFree(d_cub_keys);
            cudaFree(d_input_values);
            cudaFree(d_custom_values);
            cudaFree(d_cub_values);
        }
    }
    
    // Additional detailed test for performance analysis
    printf("\n\n=== Detailed Performance Analysis (Random Data) ===\n");
    printf("N        4bit-Basic  4bit-Super  4bit-Coal  CUB      Best-Speedup\n");
    printf("---------------------------------------------------------------\n");
    
    const int perf_sizes[] = {10000, 50000, 100000, 500000};
    const int num_perf_sizes = sizeof(perf_sizes) / sizeof(perf_sizes[0]);
    
    for (int i = 0; i < num_perf_sizes; i++) {
        int n = perf_sizes[i];
        
        // Allocate memory
        auto* host_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
        auto* host_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        
        generate_test_data(host_keys, host_values, n, 0); // Random data
        
        uint64_t *d_input_keys, *d_output_keys;
        uint32_t *d_input_values, *d_output_values;
        
        cudaMalloc(&d_input_keys, n * sizeof(uint64_t));
        cudaMalloc(&d_output_keys, n * sizeof(uint64_t));
        cudaMalloc(&d_input_values, n * sizeof(uint32_t));
        cudaMalloc(&d_output_values, n * sizeof(uint32_t));
        
        cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Time all implementations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float times[4]; // basic, super, coalesced, cub
        
        // Test super optimized
        cudaEventRecord(start);
        radix_sort_4bit_super_optimized(d_input_keys, d_output_keys, d_input_values, d_output_values, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[1], start, stop);
        
        // Reset data
        cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Test CUB
        cudaEventRecord(start);
        cub_radix_sort(d_input_keys, d_output_keys, d_input_values, d_output_values, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[3], start, stop);
        
        double best_speedup = times[3] / times[1];
        
        printf("%-7d  %10s  %10.3f  %9s  %7.3f  %11.2fx\n", 
               n, "N/A", times[1], "N/A", times[3], best_speedup);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(host_keys);
        free(host_values);
        cudaFree(d_input_keys);
        cudaFree(d_output_keys);
        cudaFree(d_input_values);
        cudaFree(d_output_values);
    }
    
    printf("\nSummary:\n");
    printf("- Speedup > 1.0 means 4-bit implementation is faster than CUB\n");
    printf("- Sorted? checks if output is properly sorted\n");
    printf("- Match? checks if 4-bit and CUB produce identical results\n");
    printf("- CPU-Match? verifies against reference implementation\n");
    printf("- Expected: 5-15x speedup over 1-bit version, competitive with CUB\n");
    
    return 0;
}