#include "tile_based_densification.cu"
#include <random>
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

static void cpuPointToTile(const float* pts, int n,
    const float ext[6], 
    int num_tiles,
    int* out)
{
    const float floor_x = std::floor(ext[0]);
    const float ceil_x = std::ceil(ext[1]);
    const float floor_y = std::floor(ext[2]);
    const float ceil_y = std::ceil(ext[3]);
    const float floor_z = std::floor(ext[4]);
    const float ceil_z = std::ceil(ext[5]);

    const float dx = (ceil_x - floor_x) / num_tiles + 1e-20f;
    const float dy = (ceil_y - floor_y) / num_tiles + 1e-20f;
    const float dz = (ceil_z - floor_z) / num_tiles + 1e-20f;

    for (int i = 0; i < n; ++i) {
        const float x = pts[i];
        const float y = pts[n + i];
        const float z = pts[2 * n + i];

        int ix = int((x - floor_x) / dx);
        int iy = int((y - floor_y) / dy);
        int iz = int((z - floor_z) / dz);

        ix = std::max(0, std::min(ix, num_tiles - 1));
        iy = std::max(0, std::min(iy, num_tiles - 1));
        iz = std::max(0, std::min(iz, num_tiles - 1));

        out[i] = ix;
        out[n + i] = iy;
        out[2 * n + i] = iz;
    }
}

int main(int argc, char** argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    const int num_tiles = (argc > 2) ? std::atoi(argv[2]) : 100;

    printf("Point-to-tile test with n=%d  tiles/axis=%d\n", n, num_tiles);

    std::vector<float> h_pts(3ULL * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1000.f, 1000.f);
    for (auto& v : h_pts) v = d(rng);

    float* d_pts{};
    CHECK_CUDA(cudaMalloc(&d_pts, h_pts.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pts, h_pts.data(),
        h_pts.size() * sizeof(float),
        cudaMemcpyHostToDevice));


    float extrema[6];
    gpuMinMax(d_pts, n, extrema);

    int* d_tile{};
    CHECK_CUDA(cudaMalloc(&d_tile, 3ULL * n * sizeof(int)));

    const int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    cudaEvent_t k_start, k_stop;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_stop);

    cudaEventRecord(k_start, 0);
    pointToTile << <blocks, BLOCK_DIM >> > (d_pts, n,
        extrema[0], extrema[1],
        extrema[2], extrema[3],
        extrema[4], extrema[5],
        num_tiles,
        d_tile);
    cudaEventRecord(k_stop, 0);
    CHECK_CUDA(cudaGetLastError());
    cudaEventSynchronize(k_stop);

    float ms_gpu = 0.f;
    cudaEventElapsedTime(&ms_gpu, k_start, k_stop);

    std::vector<int> h_tile(3ULL * n);
    CHECK_CUDA(cudaMemcpy(h_tile.data(), d_tile,
        3ULL * n * sizeof(int),
        cudaMemcpyDeviceToHost));


    std::vector<int> ref_tile(3ULL * n);

    const auto t0 = std::chrono::high_resolution_clock::now();
    cpuPointToTile(h_pts.data(), n, extrema, num_tiles, ref_tile.data());
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double ms_cpu =
        std::chrono::duration<double, std::milli>(t1 - t0).count();


    bool ok = true;
    for (size_t i = 0; i < h_tile.size(); ++i) {
        if (h_tile[i] != ref_tile[i]) {
            printf("Mismatch at idx %zu  GPU=%d  CPU=%d\n",
                i, h_tile[i], ref_tile[i]);
            ok = false;
            break;
        }
    }

    puts(ok ? "PASS - output identical to CPU reference."
        : "FAIL - discrepancy detected.");

    printf("CPU  point -> tile time : %8.3f ms\n", ms_cpu);
    printf("GPU  point -> tile time : %8.3f ms\n", ms_gpu);


    cudaFree(d_pts);
    cudaFree(d_tile);
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_stop);

    return ok ? 0 : 1;
}

