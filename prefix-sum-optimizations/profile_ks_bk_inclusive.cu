// compare_scans.cu
//
// Measure GPU run-times of inclusive prefix-scans:
//   • Brent–Kung (shared memory)
//   • Kogge–Stone (shared memory)
//
// Input sizes: 8, 16, 32, …, 2^20
// ---------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

// ---- include Brent–Kung code, but rename its add-sums kernel to avoid a clash
#define add_block_sums_to_inclusive_output_kernel bk_add_block_sums_to_inclusive_output_kernel
#include "brent_kung_scan.cu"
#undef  add_block_sums_to_inclusive_output_kernel   // restore original name-space

// ---- include Kogge–Stone code (keeps its original add-sums kernel name)
#include "kogge_stone_scan.cu"

// ---------------------------------------------------------
//  CPU reference inclusive scan
// ---------------------------------------------------------
static void inclusive_scan_cpu(const uint32_t* in, uint32_t* out, int n)
{
    if (n <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < n; ++i)
        out[i] = out[i - 1] + in[i];
}

// ---------------------------------------------------------
//  GPU wrappers – Brent–Kung (shared memory, inclusive)
// ---------------------------------------------------------
static void brent_kung_inclusive_scan_shared_memory(uint32_t* d_in,
                                                    uint32_t* d_out,
                                                    int        n)
{
    const int block_size          = 512;
    const int elems_per_block     = 2 * block_size;
    const int shared_bytes        = elems_per_block * sizeof(uint32_t);
    const int num_blocks          = (n + elems_per_block - 1) / elems_per_block;

    uint32_t* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(uint32_t));

    brent_kung_inclusive_scan_shared_memory_kernel<<<num_blocks,
                                                     block_size,
                                                     shared_bytes>>>(
        d_in, d_out, d_block_sums, n);

    if (num_blocks > 1)
    {
        brent_kung_inclusive_scan_shared_memory(d_block_sums, d_block_sums,
                                                num_blocks);
        bk_add_block_sums_to_inclusive_output_kernel<<<num_blocks, block_size>>>(
            d_out, d_block_sums, n);
    }

    cudaDeviceSynchronize();
    cudaFree(d_block_sums);
}

// ---------------------------------------------------------
//  GPU wrappers – Kogge–Stone (shared memory, inclusive)
// ---------------------------------------------------------
static void kogge_stone_inclusive_scan_shared_memory(uint32_t* d_in,
                                                     uint32_t* d_out,
                                                     int        n)
{
    const int block_size      = 512;                        // same as BK for fairness
    const int shared_bytes    = block_size * sizeof(uint32_t);
    const int num_blocks      = (n + block_size - 1) / block_size;

    uint32_t* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(uint32_t));

    kogge_stone_inclusive_scan_shared_memory_kernel<<<num_blocks,
                                                      block_size,
                                                      shared_bytes>>>(
        d_in, d_out, d_block_sums, n);

    if (num_blocks > 1)
    {
        kogge_stone_inclusive_scan_shared_memory(d_block_sums, d_block_sums,
                                                 num_blocks);
        add_block_sums_to_inclusive_output_kernel<<<num_blocks, block_size>>>(
            d_out, d_block_sums, n);
    }

    cudaDeviceSynchronize();
    cudaFree(d_block_sums);
}

// ---------------------------------------------------------
//  Main benchmark
// ---------------------------------------------------------
int main()
{
    using clk = std::chrono::high_resolution_clock;
    std::srand(42);

    printf("N         BK-ms   KS-ms   BK-ok  KS-ok\n");
    printf("---------------------------------------\n");

    for (int n = 8; n <= (1 << 20); n <<= 1)
    {
        /* host buffers */
        uint32_t* h_in   = (uint32_t*)malloc(n * sizeof(uint32_t));
        uint32_t* h_ref  = (uint32_t*)malloc(n * sizeof(uint32_t));
        uint32_t* h_bk   = (uint32_t*)malloc(n * sizeof(uint32_t));
        uint32_t* h_ks   = (uint32_t*)malloc(n * sizeof(uint32_t));

        for (int i = 0; i < n; ++i)
            h_in[i] = std::rand() % 1729;

        inclusive_scan_cpu(h_in, h_ref, n);

        /* device buffers */
        uint32_t *d_in, *d_bk, *d_ks;
        cudaMalloc(&d_in, n * sizeof(uint32_t));
        cudaMalloc(&d_bk, n * sizeof(uint32_t));
        cudaMalloc(&d_ks, n * sizeof(uint32_t));
        cudaMemcpy(d_in, h_in, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        /* ------ Brent–Kung timing ------ */
        cudaEvent_t bk_start, bk_stop;
        cudaEventCreate(&bk_start);
        cudaEventCreate(&bk_stop);
        cudaEventRecord(bk_start);

        brent_kung_inclusive_scan_shared_memory(d_in, d_bk, n);

        cudaEventRecord(bk_stop);
        cudaEventSynchronize(bk_stop);
        float bk_ms = 0.0f;
        cudaEventElapsedTime(&bk_ms, bk_start, bk_stop);
        cudaEventDestroy(bk_start);
        cudaEventDestroy(bk_stop);

        /* ------ Kogge–Stone timing ------ */
        cudaEvent_t ks_start, ks_stop;
        cudaEventCreate(&ks_start);
        cudaEventCreate(&ks_stop);
        cudaEventRecord(ks_start);

        kogge_stone_inclusive_scan_shared_memory(d_in, d_ks, n);

        cudaEventRecord(ks_stop);
        cudaEventSynchronize(ks_stop);
        float ks_ms = 0.0f;
        cudaEventElapsedTime(&ks_ms, ks_start, ks_stop);
        cudaEventDestroy(ks_start);
        cudaEventDestroy(ks_stop);

        /* copy results back & check */
        cudaMemcpy(h_bk, d_bk, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ks, d_ks, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        bool ok_bk = true, ok_ks = true;
        for (int i = 0; i < n; ++i)
        {
            ok_bk &= (h_bk[i] == h_ref[i]);
            ok_ks &= (h_ks[i] == h_ref[i]);
        }

        printf("%-9d %7.3f %7.3f   %s    %s\n",
               n, bk_ms, ks_ms,
               ok_bk ? "yes" : "NO ", ok_ks ? "yes" : "NO ");

        /* cleanup */
        free(h_in);  free(h_ref);  free(h_bk);  free(h_ks);
        cudaFree(d_in); cudaFree(d_bk); cudaFree(d_ks);
    }

    return 0;
}
