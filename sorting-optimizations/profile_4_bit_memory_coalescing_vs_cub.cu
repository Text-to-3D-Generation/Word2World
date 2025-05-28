#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "4_bit_radix_sort_memory_coalescing.cu" 
using highres_clock = std::chrono::high_resolution_clock;

void cub_radix_sort(uint64_t* input_keys, uint64_t* output_keys, uint32_t* input_values, uint32_t* output_values, int num_elements) {
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, 
        input_keys, output_keys,
        input_values, output_values,
        num_elements);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        input_keys, output_keys,
        input_values, output_values,
        num_elements);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_temp_storage);
}

void radix_sort_cpu_reference(
    uint64_t* keys, 
    uint32_t* values, 
    uint64_t* temp_keys, 
    uint32_t* temp_values, 
    int num_elements) {
    
    std::vector<std::pair<uint64_t, uint32_t>> pairs(num_elements);
    for (int i = 0; i < num_elements; i++) {
        pairs[i] = {keys[i], values[i]};
    }
    
    std::sort(pairs.begin(), pairs.end());
    
    for (int i = 0; i < num_elements; i++) {
        keys[i] = pairs[i].first;
        values[i] = pairs[i].second;
    }
}

bool verify_sort_and_match(
    uint64_t* keys1, uint32_t* values1,
    uint64_t* keys2, uint32_t* values2,
    int num_elements,
    int& mismatch_idx, const char* name1, const char* name2) {
    
    for (int i = 1; i < num_elements; i++) {
        if (keys1[i] < keys1[i-1]) {
            printf("%s sort failed: keys[%d]=%llu > keys[%d]=%llu\n", 
                   name1, i-1, keys1[i-1], i, keys1[i]);
            return false;
        }
    }
    
    for (int i = 1; i < num_elements; i++) {
        if (keys2[i] < keys2[i-1]) {
            printf("%s sort failed: keys[%d]=%llu > keys[%d]=%llu\n", 
                   name2, i-1, keys2[i-1], i, keys2[i]);
            return false;
        }
    }
    
    for (int i = 0; i < num_elements; i++) {
        if (keys1[i] != keys2[i] || values1[i] != values2[i]) {
            mismatch_idx = i;
            return false;
        }
    }
    
    return true;
}

void generate_test_data(uint64_t* keys, uint32_t* values, int num_elements, int test_case) {
    std::srand(42 + test_case);
    
    switch (test_case) {
        case 0: 
            for (int i = 0; i < num_elements; i++) {
                keys[i] = ((uint64_t)std::rand() << 32) | std::rand();
                values[i] = i;  
            }
            break;
            
        case 1: 
            for (int i = 0; i < num_elements; i++) {
                keys[i] = i;
                values[i] = i;
            }
            break;
            
        case 2: 
            for (int i = 0; i < num_elements; i++) {
                keys[i] = num_elements - 1 - i;
                values[i] = i;
            }
            break;
            
        case 3: 
            for (int i = 0; i < num_elements; i++) {
                keys[i] = std::rand() % (num_elements / 10);
                values[i] = i;
            }
            break;
            
        case 4: 
            for (int i = 0; i < num_elements; i++) {
                if (i % 100 == 0) {
                    keys[i] = ((uint64_t)1 << 63) | (std::rand() % 1000);  
                } else {
                    keys[i] = std::rand() % 1000; 
                }
                values[i] = i;
            }
            break;
            
        default: 
            for (int i = 0; i < num_elements; i++) {
                keys[i] = std::rand() % 100000;
                values[i] = i;
            }
            break;
    }
}

int main() {
    const int test_sizes[] = {1000, 5000, 10000, 15000, 20000};
    const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const char* test_names[] = {"Random", "Sorted", "Reverse", "Duplicates", "Sparse-Hi", "Limited"};
    const int num_test_types = 6;
    
    printf("Radix Sort Performance Comparison: Custom 1-bit vs CUB\n");
    printf("=====================================================\n");
    
    for (int test_type = 0; test_type < num_test_types; test_type++) {
        printf("\nTest Case: %s Data\n", test_names[test_type]);
        printf("N        Custom-ms  CUB-ms   Speedup  Match?   CPU-Match?\n");
        printf("----------------------------------------------------------\n");
        
        for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
            int n = test_sizes[size_idx];
            
            auto* host_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* host_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cpu_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cpu_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cpu_temp_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cpu_temp_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* custom_result_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* custom_result_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cub_result_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cub_result_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            
            generate_test_data(host_keys, host_values, n, test_type);
            
            memcpy(cpu_keys, host_keys, n * sizeof(uint64_t));
            memcpy(cpu_values, host_values, n * sizeof(uint32_t));
            
            radix_sort_cpu_reference(cpu_keys, cpu_values, cpu_temp_keys, cpu_temp_values, n);
            
            uint64_t *d_input_keys, *d_custom_keys, *d_cub_keys;
            uint32_t *d_input_values, *d_custom_values, *d_cub_values;
            
            cudaMalloc(&d_input_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_custom_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_cub_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_input_values, n * sizeof(uint32_t));
            cudaMalloc(&d_custom_values, n * sizeof(uint32_t));
            cudaMalloc(&d_cub_values, n * sizeof(uint32_t));
            
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            radix_sort_4bit_coalesced(d_input_keys, d_custom_keys, d_input_values, d_custom_values, n);
            cub_radix_sort(d_input_keys, d_cub_keys, d_input_values, d_cub_values, n);
            cudaDeviceSynchronize();
            
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            radix_sort_4bit_coalesced(d_input_keys, d_custom_keys, d_input_values, d_custom_values, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float custom_ms;
            cudaEventElapsedTime(&custom_ms, start, stop);
            
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            cub_radix_sort(d_input_keys, d_cub_keys, d_input_values, d_cub_values, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float cub_ms;
            cudaEventElapsedTime(&cub_ms, start, stop);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            cudaMemcpy(custom_result_keys, d_custom_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(custom_result_values, d_custom_values, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(cub_result_keys, d_cub_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(cub_result_values, d_cub_values, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            int mismatch_idx = -1;
            bool implementations_match = verify_sort_and_match(
                custom_result_keys, custom_result_values,
                cub_result_keys, cub_result_values,
                n, mismatch_idx, "Custom", "CUB");
            
            int cpu_mismatch_idx = -1;
            bool custom_vs_cpu = verify_sort_and_match(
                custom_result_keys, custom_result_values,
                cpu_keys, cpu_values,
                n, cpu_mismatch_idx, "Custom", "CPU");
            
            double speedup = cub_ms / custom_ms;
            
            printf("%-7d  %9.3f  %7.3f  %6.2fx  %-7s  %-10s", 
                   n, custom_ms, cub_ms, speedup,
                   implementations_match ? "yes" : "NO",
                   custom_vs_cpu ? "yes" : "NO");
            
            if (!implementations_match && mismatch_idx >= 0) {
                printf("\n    Custom vs CUB mismatch @ %d", mismatch_idx);
                printf("\n    Custom: key=%llu, val=%u", custom_result_keys[mismatch_idx], custom_result_values[mismatch_idx]);
                printf("\n    CUB:    key=%llu, val=%u", cub_result_keys[mismatch_idx], cub_result_values[mismatch_idx]);
            }
            
            if (!custom_vs_cpu && cpu_mismatch_idx >= 0) {
                printf("\n    Custom vs CPU mismatch @ %d", cpu_mismatch_idx);
                printf("\n    Custom: key=%llu, val=%u", custom_result_keys[cpu_mismatch_idx], custom_result_values[cpu_mismatch_idx]);
                printf("\n    CPU:    key=%llu, val=%u", cpu_keys[cpu_mismatch_idx], cpu_values[cpu_mismatch_idx]);
            }
            
            printf("\n");
            
            free(host_keys);
            free(host_values);
            free(cpu_keys);
            free(cpu_values);
            free(cpu_temp_keys);
            free(cpu_temp_values);
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
    
    printf("\nSummary:\n");
    printf("- Speedup = CUB_time / Custom_time (>1.0 means Custom is faster)\n");
    printf("- Match: Do Custom and CUB produce identical results?\n");
    printf("- CPU-Match: Does Custom match CPU reference implementation?\n");
    printf("- Test cases cover various data distributions for comprehensive evaluation\n");
    printf("- CUB is highly optimized, so competitive performance indicates excellent implementation\n");
    
    return 0;
}