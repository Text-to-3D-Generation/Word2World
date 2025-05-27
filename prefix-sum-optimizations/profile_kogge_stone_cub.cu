#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>            

#include "kogge_stone_scan.cu"
using highres_clock = std::chrono::high_resolution_clock;


void gpu_scan_global(uint32_t* d_in, uint32_t* d_out, int n)
{
    constexpr int BLOCK = 128;
    int blocks = (n + BLOCK - 1) / BLOCK;

    uint32_t* d_sums;
    cudaMalloc(&d_sums, blocks * sizeof(uint32_t));

    kogge_stone_inclusive_scan_no_optimizations_kernel<<<blocks, BLOCK>>>(
        d_in, d_out, d_sums, n);

    if (blocks > 1) {
        gpu_scan_global(d_sums, d_sums, blocks);
        add_block_sums_to_inclusive_output_kernel<<<blocks, BLOCK>>>(
            d_out, d_sums, n);
    }
    cudaDeviceSynchronize();
    cudaFree(d_sums);
}

void gpu_scan_shared(uint32_t* d_in, uint32_t* d_out, int n)
{
    constexpr int BLOCK = 128;
    size_t shmem = BLOCK * sizeof(uint32_t);
    int blocks = (n + BLOCK - 1) / BLOCK;

    uint32_t* d_sums;
    cudaMalloc(&d_sums, blocks * sizeof(uint32_t));

    kogge_stone_inclusive_scan_shared_memory_kernel<<<blocks, BLOCK, shmem>>>(
        d_in, d_out, d_sums, n);

    if (blocks > 1) {
        gpu_scan_shared(d_sums, d_sums, blocks);
        add_block_sums_to_inclusive_output_kernel<<<blocks, BLOCK>>>(
            d_out, d_sums, n);
    }
    cudaDeviceSynchronize();
    cudaFree(d_sums);
}


void cub_scan(uint32_t* d_in, uint32_t* d_out, int n)
{
    void*  d_temp = nullptr;
    size_t bytes  = 0;
    cub::DeviceScan::InclusiveSum(d_temp, bytes, d_in, d_out, n);
    cudaMalloc(&d_temp, bytes);
    cub::DeviceScan::InclusiveSum(d_temp, bytes, d_in, d_out, n);
    cudaFree(d_temp);
}


int main()
{
    std::srand(42);
    const int lower = 5000, upper = 150000, step = 10000;

    printf("N        CUB-ms   Global-ms   Shared-ms   Match?\n");
    printf("-------------------------------------------------\n");

    for (int n = lower; n <= upper; n += step)
    {
        auto* h_in     = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* h_cub    = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* h_global = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
        auto* h_shared = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));

        for (int i = 0; i < n; ++i) h_in[i] = std::rand() % 1729;

        uint32_t *d_in, *d_out;
        cudaMalloc(&d_in,  n * sizeof(uint32_t));
        cudaMalloc(&d_out, n * sizeof(uint32_t));
        cudaMemcpy(d_in, h_in, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        cub_scan(d_in, d_out, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float cub_ms;
        cudaEventElapsedTime(&cub_ms, e0, e1);
        cudaMemcpy(h_cub, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        gpu_scan_global(d_in, d_out, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float global_ms;
        cudaEventElapsedTime(&global_ms, e0, e1);
        cudaMemcpy(h_global, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        gpu_scan_shared(d_in, d_out, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float shared_ms;
        cudaEventElapsedTime(&shared_ms, e0, e1);
        cudaMemcpy(h_shared, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            ok = (h_cub[i] == h_global[i]) && (h_cub[i] == h_shared[i]);

        printf("%-7d  %7.3f   %9.3f   %10.3f   %s\n",
               n, cub_ms, global_ms, shared_ms, ok ? "yes" : "NO");

        free(h_in); free(h_cub); free(h_global); free(h_shared);
        cudaFree(d_in); cudaFree(d_out);
    }
    return 0;
}
