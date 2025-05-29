#include "tile_based_densification.cu"
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

static inline int redMin(float* d, int n) {
    while (n > BLOCK_DIM) {
        int b = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMin256 << <b, BLOCK_DIM >> > (d, n, n); CHECK_CUDA(cudaGetLastError()); n = b;
    } return n;
}
static inline int redMax(float* d, int n) {
    while (n > BLOCK_DIM) {
        int b = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMax256 << <b, BLOCK_DIM >> > (d, n, n); CHECK_CUDA(cudaGetLastError()); n = b;
    } return n;
}
static void gpuMinMax(const float* d_pts, int n, float bb[6]) {
    size_t bytes = 3ULL * n * sizeof(float); float* dmin, * dmax;
    CHECK_CUDA(cudaMalloc(&dmin, bytes)); CHECK_CUDA(cudaMalloc(&dmax, bytes));
    CHECK_CUDA(cudaMemcpy(dmin, d_pts, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dmax, d_pts, bytes, cudaMemcpyDeviceToDevice));
    int tm = redMin(dmin, n), tx = redMax(dmax, n), tail = std::max(tm, tx);
    float hmin[3 * BLOCK_DIM], hmax[3 * BLOCK_DIM];
    CHECK_CUDA(cudaMemcpy(hmin, dmin, tm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmin + tm, dmin + n, tm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmin + 2 * tm, dmin + 2 * n, tm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmax, dmax, tx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmax + tx, dmax + n, tx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmax + 2 * tx, dmax + 2 * n, tx * sizeof(float), cudaMemcpyDeviceToHost));
    bb[0] = FLT_MAX; bb[1] = -FLT_MAX; bb[2] = FLT_MAX; bb[3] = -FLT_MAX; bb[4] = FLT_MAX; bb[5] = -FLT_MAX;
    for (int i = 0; i < tail; ++i) {
        bb[0] = std::min(bb[0], hmin[i]); bb[1] = std::max(bb[1], hmax[i]);
        bb[2] = std::min(bb[2], hmin[tail + i]); bb[3] = std::max(bb[3], hmax[tail + i]);
        bb[4] = std::min(bb[4], hmin[2 * tail + i]); bb[5] = std::max(bb[5], hmax[2 * tail + i]);
    }
    cudaFree(dmin); cudaFree(dmax);
}

static void cpuEigen3(double a, double b, double c, double d, double e, double f,
    double& l0, double& l1, double& l2)
{
    double m = (a + b + c) / 3.0, a0 = a - m, b0 = b - m, c0 = c - m;
    double p = sqrt((a0 * a0 + b0 * b0 + c0 * c0 + 2 * (d * d + e * e + f * f)) / 6.0);
    if (p == 0) { l0 = l1 = l2 = 0; return; }
    a0 /= p; b0 /= p; c0 /= p; d /= p; e /= p; f /= p;
    double det = a0 * b0 * c0 + 2 * d * e * f - a0 * f * f - b0 * e * e - c0 * d * d;
    double r = std::max(-1.0, std::min(1.0, det * 0.5));
    double t = acos(r) / 3.0;
    auto C = [](double s) {return 2.0 * std::cos(s); };
    l0 = p * C(t) + m; l1 = p * C(t + 2 * M_PI / 3.0) + m; l2 = p * C(t + 4 * M_PI / 3.0) + m;
    if (l0 < l1)std::swap(l0, l1); if (l1 < l2)std::swap(l1, l2); if (l0 < l1)std::swap(l0, l1);
}

static void cpuFlagPerTile(const float* pts, const int* idx,
    int n, int T,
    float th1, float th2,
    std::vector<unsigned char>& flag)
{
    const float wL = 0.60f, wP = 0.30f, wS = 0.10f;
    int T3 = T * T * T;
    std::vector<int>    cnt(T3, 0);
    std::vector<double> sum(T3 * 3, 0.0), s2(T3 * 6, 0.0);

    for (int i = 0; i < n; ++i) {
        int tile = ((idx[2 * n + i] * T) + idx[n + i]) * T + idx[i];
        double x = pts[i], y = pts[n + i], z = pts[2 * n + i];
        ++cnt[tile];
        sum[3 * tile] += x; sum[3 * tile + 1] += y; sum[3 * tile + 2] += z;
        s2[6 * tile] += x * x; s2[6 * tile + 1] += x * y; s2[6 * tile + 2] += x * z;
        s2[6 * tile + 3] += y * y; s2[6 * tile + 4] += y * z; s2[6 * tile + 5] += z * z;
    }

    flag.assign(T3, 0);
    for (int t = 0; t < T3; ++t) {
        int m = cnt[t]; if (m < 2) continue;
        double inv = 1.0 / m,
            mx = sum[3 * t] * inv, my = sum[3 * t + 1] * inv, mz = sum[3 * t + 2] * inv;
        double fac = 1.0 / (m - 1);
        double cxx = (s2[6 * t] - m * mx * mx) * fac,
            cxy = (s2[6 * t + 1] - m * mx * my) * fac,
            cxz = (s2[6 * t + 2] - m * mx * mz) * fac,
            cyy = (s2[6 * t + 3] - m * my * my) * fac,
            cyz = (s2[6 * t + 4] - m * my * mz) * fac,
            czz = (s2[6 * t + 5] - m * mz * mz) * fac;
        double l0, l1, l2; cpuEigen3(cxx, cyy, czz, cxy, cxz, cyz, l0, l1, l2);
        if (l0 <= 0) continue;
        float V = l0 + l1 + l2; if (V < th1) continue;
        float L = (l0 - l1) / l0, P = (l1 - l2) / l0, S = 1.f - L - P;
        L = fminf(fmaxf(L, 0.f), 1.f); P = fminf(fmaxf(P, 0.f), 1.f); S = fminf(fmaxf(S, 0.f), 1.f);
        float score = wL * L + wP * P + wS * S;
        if (score >= th2) flag[t] = 1;
    }
}

int main(int argc, char** argv)
{
    int   n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int   tiles = (argc > 2) ? std::atoi(argv[2]) : 64;
    float th1 = (argc > 3) ? std::atof(argv[3]) : 50.f;
    float th2 = (argc > 4) ? std::atof(argv[4]) : 0.5f;

    printf("E2E test  n=%d  tiles=%d  th1=%g  th2=%g\n\n", n, tiles, th1, th2);

    std::vector<float> h_pts(3ULL * n);
    std::mt19937 rng(42); std::uniform_real_distribution<float>U(-1000.f, 1000.f);
    for (float& v : h_pts) v = U(rng);

    float* d_pts; CHECK_CUDA(cudaMalloc(&d_pts, h_pts.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pts, h_pts.data(), h_pts.size() * sizeof(float), cudaMemcpyHostToDevice));

    float bb[6]; gpuMinMax(d_pts, n, bb);

    int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, 3ULL * n * sizeof(int)));
    dim3 blk(BLOCK_DIM), grd((n + blk.x - 1) / blk.x);
    pointToTile << <grd, blk >> > (d_pts, n, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], tiles, d_idx);
    CHECK_CUDA(cudaGetLastError());

    int T3 = tiles * tiles * tiles;
    unsigned char* d_tflag; CHECK_CUDA(cudaMalloc(&d_tflag, T3));
    gpuDensifyFlagPerTile(d_pts, d_idx, n, tiles, th1, th2, d_tflag);

    unsigned char* d_pflag; CHECK_CUDA(cudaMalloc(&d_pflag, n));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    gpuPointFlagPerPoint(d_idx, n, tiles, d_tflag, d_pflag);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_gpu_map = 0.f; cudaEventElapsedTime(&ms_gpu_map, e0, e1);

    std::vector<unsigned char> h_pflag(n);
    CHECK_CUDA(cudaMemcpy(h_pflag.data(), d_pflag, n, cudaMemcpyDeviceToHost));

    std::vector<int> h_idx(3ULL * n);
    CHECK_CUDA(cudaMemcpy(h_idx.data(), d_idx, h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> ref_tflag(T3);
    {   
        cpuFlagPerTile(h_pts.data(), h_idx.data(), n, tiles, th1, th2, ref_tflag);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<unsigned char> ref_pflag(n, 0);
    for (int i = 0; i < n; ++i) {
        int ix = h_idx[i], iy = h_idx[n + i], iz = h_idx[2 * n + i];
        int tile = ((iz * tiles) + iy) * tiles + ix;
        ref_pflag[i] = ref_tflag[tile];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu_map = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t mism = 0, gpu_true = 0, gpu_false = 0;
    for (int i = 0; i < n; ++i) {
        if (h_pflag[i] != ref_pflag[i]) ++mism;
        if (h_pflag[i]) ++gpu_true; else ++gpu_false;
    }
    printf("Per-point mismatches : %zu / %d  (%.5f %%)\n",
        mism, n, 100.0 * mism / n);
    printf("GPU :  true=%zu  false=%zu\n\n", gpu_true, gpu_false);

    printf("CPU map time : %8.3f ms\n", ms_cpu_map);
    printf("GPU map time : %8.3f ms\n", ms_gpu_map);

    cudaFree(d_pts); cudaFree(d_idx); cudaFree(d_tflag); cudaFree(d_pflag);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
