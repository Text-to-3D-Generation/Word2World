#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#include "2_bit_radix_sort_memory_coalescing.cu" 
using highres_clock = std::chrono::high_resolution_clock;

void radix_sort_cpu(uint64_t* keys, uint32_t* values, uint64_t* temp_keys, uint32_t* temp_values, int num_elements) {
    for (int bit = 0; bit < 64; bit++) {
        std::vector<std::pair<uint64_t, uint32_t>> zeros, ones;
        
        for (int i = 0; i < num_elements; i++) {
            if ((keys[i] >> bit) & 1) {
                ones.push_back({keys[i], values[i]});
            } else {
                zeros.push_back({keys[i], values[i]});
            }
        }
        
        int idx = 0;
        for (const auto& pair : zeros) {
            temp_keys[idx] = pair.first;
            temp_values[idx] = pair.second;
            idx++;
        }
        for (const auto& pair : ones) {
            temp_keys[idx] = pair.first;
            temp_values[idx] = pair.second;
            idx++;
        }
        
        std::swap(keys, temp_keys);
        std::swap(values, temp_values);
    }
}

bool verify_sort(uint64_t* gpu_keys, uint32_t* gpu_values, uint64_t* cpu_keys, uint32_t* cpu_values, int num_elements, int& mismatch_idx) {
    
    for (int i = 1; i < num_elements; i++) {
        if (gpu_keys[i] < gpu_keys[i-1]) {
            printf("GPU sort failed: keys[%d]=%llu > keys[%d]=%llu\n", 
                   i-1, gpu_keys[i-1], i, gpu_keys[i]);
            return false;
        }
    }
    
    for (int i = 0; i < num_elements; i++) {
        if (gpu_keys[i] != cpu_keys[i] || gpu_values[i] != cpu_values[i]) {
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
                keys[i] = std::rand() % 100;
                values[i] = i;
            }
            break;
            
        default:
            for (int i = 0; i < num_elements; i++) {
                keys[i] = std::rand() % 10000;
                values[i] = i;
            }
            break;
    }
}

int main() {
    const int test_sizes[] = {1000, 5000, 10000, 15000, 20000};
    const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const char* test_names[] = {"Random", "Sorted", "Reverse", "Duplicates", "Limited"};
    const int num_test_types = 5;
    
    printf("1-bit Radix Sort Performance Comparison\n");
    printf("=======================================\n");
    
    for (int test_type = 0; test_type < num_test_types; test_type++) {
        printf("\nTest Case: %s Data\n", test_names[test_type]);
        printf("N        CPU-ms   GPU-ms   Speedup  Match?\n");
        printf("---------------------------------------------\n");
        
        for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
            int n = test_sizes[size_idx];
            
            auto* host_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* host_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cpu_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cpu_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* cpu_temp_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* cpu_temp_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            auto* gpu_result_keys = static_cast<uint64_t*>(malloc(n * sizeof(uint64_t)));
            auto* gpu_result_values = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
            
            generate_test_data(host_keys, host_values, n, test_type);
            
            memcpy(cpu_keys, host_keys, n * sizeof(uint64_t));
            memcpy(cpu_values, host_values, n * sizeof(uint32_t));
            
            auto t0 = highres_clock::now();
            radix_sort_cpu(cpu_keys, cpu_values, cpu_temp_keys, cpu_temp_values, n);
            auto t1 = highres_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            
            uint64_t *d_input_keys, *d_output_keys;
            uint32_t *d_input_values, *d_output_values;
            
            cudaMalloc(&d_input_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_output_keys, n * sizeof(uint64_t));
            cudaMalloc(&d_input_values, n * sizeof(uint32_t));
            cudaMalloc(&d_output_values, n * sizeof(uint32_t));
            
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            radix_sort_2bit_coalesced(d_input_keys, d_output_keys, d_input_values, d_output_values, n);
            cudaDeviceSynchronize();
            
            cudaMemcpy(d_input_keys, host_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_values, host_values, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            radix_sort_2bit_coalesced(d_input_keys, d_output_keys, d_input_values, d_output_values, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float gpu_ms;
            cudaEventElapsedTime(&gpu_ms, start, stop);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            cudaMemcpy(gpu_result_keys, d_output_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(gpu_result_values, d_output_values, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            int mismatch_idx = -1;
            bool match = verify_sort(gpu_result_keys, gpu_result_values, cpu_keys, cpu_values, n, mismatch_idx);
            
            double speedup = cpu_ms / gpu_ms;
            
            printf("%-7d  %7.2f  %7.2f  %6.2fx  %s", 
                   n, cpu_ms, gpu_ms, speedup, match ? "yes" : "NO");
            
            if (!match && mismatch_idx >= 0) {
                printf("  (mismatch @ %d)", mismatch_idx);
                printf("\n  GPU: key=%llu, val=%u", gpu_result_keys[mismatch_idx], gpu_result_values[mismatch_idx]);
                printf("\n  CPU: key=%llu, val=%u", cpu_keys[mismatch_idx], cpu_values[mismatch_idx]);
            }
            
            printf("\n");
            
            free(host_keys);
            free(host_values);
            free(cpu_keys);
            free(cpu_values);
            free(cpu_temp_keys);
            free(cpu_temp_values);
            free(gpu_result_keys);
            free(gpu_result_values);
            
            cudaFree(d_input_keys);
            cudaFree(d_output_keys);
            cudaFree(d_input_values);
            cudaFree(d_output_values);
        }
    }
    return 0;
}