#include "tile_based_densification.cu" 
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cassert>

static inline int iterativeMin(float* d_points, int n)
{
    while (n > BLOCK_DIM) {
        int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMin256 << <blocks, BLOCK_DIM >> > (d_points, n, n);
        CHECK_CUDA(cudaGetLastError());
        n = blocks;
    }
    return n;
}
static inline int iterativeMax(float* d_points, int n)
{
    while (n > BLOCK_DIM) {
        int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMax256 << <blocks, BLOCK_DIM >> > (d_points, n, n);
        CHECK_CUDA(cudaGetLastError());
        n = blocks;
    }
    return n;
}

void gpuMinMax(const float* d_points, int n, float out[6])
{
    const size_t bytes = 3ULL * n * sizeof(float);
    float* d_min, * d_max;
    CHECK_CUDA(cudaMalloc(&d_min, bytes));
    CHECK_CUDA(cudaMalloc(&d_max, bytes));
    CHECK_CUDA(cudaMemcpy(d_min, d_points, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, d_points, bytes, cudaMemcpyDeviceToDevice));

    int tailMin = iterativeMin(d_min, n);
    int tailMax = iterativeMax(d_max, n);
    const int tail = std::max(tailMin, tailMax);

    float h_min[3 * BLOCK_DIM], h_max[3 * BLOCK_DIM];
    CHECK_CUDA(cudaMemcpy(h_min, d_min, tailMin * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_min + tailMin, d_min + n, tailMin * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_min + 2 * tailMin, d_min + 2 * n, tailMin * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(h_max, d_max, tailMax * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_max + tailMax, d_max + n, tailMax * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_max + 2 * tailMax, d_max + 2 * n, tailMax * sizeof(float), cudaMemcpyDeviceToHost));

    out[0] = FLT_MAX; out[1] = -FLT_MAX; out[2] = FLT_MAX; out[3] = -FLT_MAX; out[4] = FLT_MAX; out[5] = -FLT_MAX;
    for (int i = 0; i < tail; ++i) {
        out[0] = std::min(out[0], h_min[i]);
        out[1] = std::max(out[1], h_max[i]);
        out[2] = std::min(out[2], h_min[tail + i]);
        out[3] = std::max(out[3], h_max[tail + i]);
        out[4] = std::min(out[4], h_min[2 * tail + i]);
        out[5] = std::max(out[5], h_max[2 * tail + i]);
    }
    cudaFree(d_min); cudaFree(d_max);
}

static void cpuCovariance(const float* pts,
    const int* idx,
    int         n,
    int         T,
    std::vector<float>& cov_out)
{
    const int T3 = T * T * T;
    std::vector<int>    cnt(T3, 0);
    std::vector<double> sum(T3 * 3, 0.0);
    std::vector<double> s2(T3 * 6, 0.0);

    for (int i = 0; i < n; ++i) {
        int tile = ((idx[2 * n + i] * T) + idx[n + i]) * T + idx[i];
        double x = pts[i], y = pts[n + i], z = pts[2 * n + i];
        ++cnt[tile];
        sum[3 * tile + 0] += x; sum[3 * tile + 1] += y; sum[3 * tile + 2] += z;
        s2[6 * tile + 0] += x * x; s2[6 * tile + 1] += x * y; s2[6 * tile + 2] += x * z;
        s2[6 * tile + 3] += y * y; s2[6 * tile + 4] += y * z; s2[6 * tile + 5] += z * z;
    }

    cov_out.assign(9 * T3, 0.f);
    for (int t = 0; t < T3; ++t) {
        int m = cnt[t]; if (m < 2) continue;
        double invm = 1.0 / m;
        double mx = sum[3 * t + 0] * invm, my = sum[3 * t + 1] * invm, mz = sum[3 * t + 2] * invm;
        double d = 1.0 / (m - 1);
        double cxx = (s2[6 * t + 0] - m * mx * mx) * d, cxy = (s2[6 * t + 1] - m * mx * my) * d, cxz = (s2[6 * t + 2] - m * mx * mz) * d;
        double cyy = (s2[6 * t + 3] - m * my * my) * d, cyz = (s2[6 * t + 4] - m * my * mz) * d, czz = (s2[6 * t + 5] - m * mz * mz) * d;
        float* C = &cov_out[9 * t];
        C[0] = cxx; C[1] = cxy; C[2] = cxz; C[3] = cxy; C[4] = cyy; C[5] = cyz; C[6] = cxz; C[7] = cyz; C[8] = czz;
    }
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int tiles = (argc > 2) ? std::atoi(argv[2]) : 64;
    printf("Covariance-per-tile  (n=%d  tiles/axis=%d)\n", n, tiles);

    std::vector<float> h_pts(3ULL * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-1000.f, 1000.f);
    for (float& v : h_pts) v = U(rng);

    float* d_pts; CHECK_CUDA(cudaMalloc(&d_pts, h_pts.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pts, h_pts.data(),
        h_pts.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    float bb[6]; gpuMinMax(d_pts, n, bb);

    int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, 3ULL * n * sizeof(int)));
    dim3 blk(BLOCK_DIM); dim3 grd((n + blk.x - 1) / blk.x);
    pointToTile << <grd, blk >> > (d_pts, n,
        bb[0], bb[1], bb[2], bb[3], bb[4], bb[5],
        tiles, d_idx);
    CHECK_CUDA(cudaGetLastError());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    int   T3 = tiles * tiles * tiles;
    float* d_cov; CHECK_CUDA(cudaMalloc(&d_cov, 9ULL * T3 * sizeof(float)));

    cudaEventRecord(t0);
    gpuCovariancePerTile(d_pts, d_idx, n, tiles, d_cov);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_gpu = 0.f; cudaEventElapsedTime(&ms_gpu, t0, t1);

    std::vector<float> h_cov(9ULL * T3);
    CHECK_CUDA(cudaMemcpy(h_cov.data(), d_cov,
        h_cov.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    std::vector<int> h_idx(3ULL * n);
    CHECK_CUDA(cudaMemcpy(h_idx.data(), d_idx,
        h_idx.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> ref;
    cpuCovariance(h_pts.data(), h_idx.data(), n, tiles, ref);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    const float eps = 6e-1f;
    bool ok = true;
    for (size_t i = 0; i < h_cov.size(); ++i) {
        float diff = fabsf(h_cov[i] - ref[i]);
        if (diff > eps && fabsf(ref[i]) > 0.f) {
            printf("Mismatch at %zu  gpu=%g  cpu=%g\n", i, h_cov[i], ref[i]);
            ok = false; break;
        }
    }
    puts(ok ? "PASS - results within tolerance." : "FAIL - discrepancy detected.");

    printf("CPU  time : %8.3f ms\n", ms_cpu);
    printf("GPU  time : %8.3f ms\n", ms_gpu);

    cudaFree(d_pts); cudaFree(d_idx); cudaFree(d_cov);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ok ? 0 : 1;
}
