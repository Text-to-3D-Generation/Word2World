#include "tile_based_densification.cu"
#include <chrono>

static inline int iterativeMin(float* d_points, int n)
{
    int curr = n;
    while (curr > BLOCK_DIM) {
        int blocks = (curr + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMin256 << <blocks, BLOCK_DIM >> > (d_points, curr, n); 
        CHECK_CUDA(cudaGetLastError());
        curr = blocks;
    }
    return curr;
}

static inline int iterativeMax(float* d_points, int n)
{
    int curr = n;
    while (curr > BLOCK_DIM) {
        int blocks = (curr + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMax256 << <blocks, BLOCK_DIM >> > (d_points, curr, n); 
        CHECK_CUDA(cudaGetLastError());
        curr = blocks;
    }
    return curr;
}


void gpuMinMax(const float* d_points_in, int n, float out[6])
{
    const size_t bytes = 3ULL * n * sizeof(float);

    float* d_min, * d_max;
    CHECK_CUDA(cudaMalloc(&d_min, bytes));
    CHECK_CUDA(cudaMalloc(&d_max, bytes));
    CHECK_CUDA(cudaMemcpy(d_min, d_points_in, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, d_points_in, bytes, cudaMemcpyDeviceToDevice));

    int tailMin = iterativeMin(d_min, n); 
    int tailMax = iterativeMax(d_max, n); 

    const int tail = std::max(tailMin, tailMax);
    float h_min[3 * BLOCK_DIM]; 
    float h_max[3 * BLOCK_DIM];
    CHECK_CUDA(cudaMemcpy(h_min,
        d_min,
        tailMin * sizeof(float),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_min + tailMin,
        d_min + n,
        tailMin * sizeof(float),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_min + 2 * tailMin,
        d_min + 2 * n,
        tailMin * sizeof(float),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_max,
        d_max,
        tailMax * sizeof(float),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_max + tailMax,
        d_max + n,
        tailMax * sizeof(float),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_max + 2 * tailMax,
        d_max + 2 * n,
        tailMax * sizeof(float),
        cudaMemcpyDeviceToHost));

    out[0] = FLT_MAX;   out[1] = -FLT_MAX; 
    out[2] = FLT_MAX;   out[3] = -FLT_MAX;  
    out[4] = FLT_MAX;   out[5] = -FLT_MAX;  

    for (int i = 0; i < tail; ++i) {
        out[0] = std::min(out[0], h_min[i]);            
        out[1] = std::max(out[1], h_max[i]);               
        out[2] = std::min(out[2], h_min[tail + i]);        
        out[3] = std::max(out[3], h_max[tail + i]);       
        out[4] = std::min(out[4], h_min[2 * tail + i]);   
        out[5] = std::max(out[5], h_max[2 * tail + i]);  
    }

    cudaFree(d_min);
    cudaFree(d_max);
}

void cpuMinMax(const float* h_points, int n, float out[6])
{
    out[0] = FLT_MAX;   out[1] = -FLT_MAX;
    out[2] = FLT_MAX;   out[3] = -FLT_MAX;
    out[4] = FLT_MAX;   out[5] = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        const float x = h_points[i];
        const float y = h_points[n + i];
        const float z = h_points[2 * n + i];
        out[0] = std::min(out[0], x);
        out[1] = std::max(out[1], x);
        out[2] = std::min(out[2], y);
        out[3] = std::max(out[3], y);
        out[4] = std::min(out[4], z);
        out[5] = std::max(out[5], z);
    }
}


int main(int argc, char** argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);  
    printf("Iterative min/max test with %d points\n", n);

    float* h_points = (float*)std::malloc(3ULL * n * sizeof(float));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
    for (long long i = 0; i < 3LL * n; ++i) h_points[i] = dist(gen);

    float* d_points;
    CHECK_CUDA(cudaMalloc(&d_points, 3ULL * n * sizeof(float)));

    cudaEvent_t h2d_start, h2d_stop;
    cudaEventCreate(&h2d_start); cudaEventCreate(&h2d_stop);
    cudaEventRecord(h2d_start, 0);
    CHECK_CUDA(cudaMemcpy(d_points, h_points, 3ULL * n * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(h2d_stop, 0);
    cudaEventSynchronize(h2d_stop);
    float ms_h2d = 0.f;
    cudaEventElapsedTime(&ms_h2d, h2d_start, h2d_stop);

    float gpuOut[6];
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start); cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);
    gpuMinMax(d_points, n, gpuOut);         
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    float ms_gpu = 0.f;
    cudaEventElapsedTime(&ms_gpu, gpu_start, gpu_stop);

    float cpuOut[6];
    const auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpuMinMax(h_points, n, cpuOut);
    const auto cpu_t1 = std::chrono::high_resolution_clock::now();
    const double ms_cpu = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    const float eps = 1e-5f;
    bool ok = true;
    for (int i = 0; i < 6; ++i) {
        const float diff = std::fabs(gpuOut[i] - cpuOut[i]);
        if (diff > eps) {
            printf("Mismatch idx %d  GPU %f  CPU %f  diff %f\n", i, gpuOut[i], cpuOut[i], diff);
                ok = false;
        }
    }

    printf(ok ? "PASS - results match.\n" : "FAIL - mismatch detected.\n");

    printf("CPU   reduction time : %8.3f ms\n", ms_cpu);
    printf("GPU   reduction time : %8.3f ms\n", ms_gpu);
    printf("Host -> Device copy   : %8.3f ms (3n floats)\n", ms_h2d);

    std::free(h_points);
    cudaFree(d_points);
    cudaEventDestroy(h2d_start);  cudaEventDestroy(h2d_stop);
    cudaEventDestroy(gpu_start);  cudaEventDestroy(gpu_stop);

    return ok ? 0 : 1;
}
