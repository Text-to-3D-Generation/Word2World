#include "tile_based_densification.cu"
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static inline int iterativeMin(float* d_points, int n) {
    while (n > BLOCK_DIM) {
        int b = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMin256 << <b, BLOCK_DIM >> > (d_points, n, n); CHECK_CUDA(cudaGetLastError()); n = b;
    }return n;
}
static inline int iterativeMax(float* d_points, int n) {
    while (n > BLOCK_DIM) {
        int b = (n + BLOCK_DIM - 1) / BLOCK_DIM;
        reduceMax256 << <b, BLOCK_DIM >> > (d_points, n, n); CHECK_CUDA(cudaGetLastError()); n = b;
    }return n;
}
void gpuMinMax(const float* d_points, int n, float out[6]) {
    const size_t bytes = 3ULL * n * sizeof(float); float* dmin, * dmax;
    CHECK_CUDA(cudaMalloc(&dmin, bytes)); CHECK_CUDA(cudaMalloc(&dmax, bytes));
    CHECK_CUDA(cudaMemcpy(dmin, d_points, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dmax, d_points, bytes, cudaMemcpyDeviceToDevice));
    int tm = iterativeMin(dmin, n), tx = iterativeMax(dmax, n), tail = std::max(tm, tx);
    float hmin[3 * BLOCK_DIM], hmax[3 * BLOCK_DIM];
    CHECK_CUDA(cudaMemcpy(hmin, dmin, tm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmin + tm, dmin + n, tm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmin + 2 * tm, dmin + 2 * n, tm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmax, dmax, tx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmax + tx, dmax + n, tx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hmax + 2 * tx, dmax + 2 * n, tx * sizeof(float), cudaMemcpyDeviceToHost));
    out[0] = FLT_MAX; out[1] = -FLT_MAX; out[2] = FLT_MAX; out[3] = -FLT_MAX; out[4] = FLT_MAX; out[5] = -FLT_MAX;
    for (int i = 0; i < tail; ++i) {
        out[0] = std::min(out[0], hmin[i]);            out[1] = std::max(out[1], hmax[i]);
        out[2] = std::min(out[2], hmin[tail + i]);       out[3] = std::max(out[3], hmax[tail + i]);
        out[4] = std::min(out[4], hmin[2 * tail + i]);     out[5] = std::max(out[5], hmax[2 * tail + i]);
    }
    cudaFree(dmin); cudaFree(dmax);
}

static void cpuEigen3x3(double a, double b, double c, double d, double e, double f,
    double& l0, double& l1, double& l2)
{
    const double m = (a + b + c) / 3.0;
    double a0 = a - m, b0 = b - m, c0 = c - m;
    double p = sqrt((a0 * a0 + b0 * b0 + c0 * c0 + 2.0 * (d * d + e * e + f * f)) / 6.0);
    if (p == 0.0) { l0 = l1 = l2 = 0.0; return; }
    a0 /= p; b0 /= p; c0 /= p; d /= p; e /= p; f /= p;
    double detB = a0 * b0 * c0 + 2 * d * e * f - a0 * f * f - b0 * e * e - c0 * d * d;
    double r = std::max(-1.0, std::min(1.0, detB * 0.5));
    double t = acos(r) / 3.0;
    auto C = [&](double s) {return 2.0 * cos(s); };
    l0 = p * C(t) + m; l1 = p * C(t + 2.0 * M_PI / 3.0) + m; l2 = p * C(t + 4.0 * M_PI / 3.0) + m;
    if (l0 < l1)std::swap(l0, l1); if (l1 < l2)std::swap(l1, l2); if (l0 < l1)std::swap(l0, l1);
}
static void cpuEigenPerTile(const float* pts, const int* idx, int n, int T, std::vector<float>& eig)
{
    int T3 = T * T * T; std::vector<int>cnt(T3, 0); std::vector<double>sum(T3 * 3, 0.0), s2(T3 * 6, 0.0);
    for (int i = 0; i < n; ++i) {
        int tile = ((idx[2 * n + i] * T) + idx[n + i]) * T + idx[i];
        double x = pts[i], y = pts[n + i], z = pts[2 * n + i]; ++cnt[tile];
        sum[3 * tile + 0] += x; sum[3 * tile + 1] += y; sum[3 * tile + 2] += z;
        s2[6 * tile + 0] += x * x; s2[6 * tile + 1] += x * y; s2[6 * tile + 2] += x * z;
        s2[6 * tile + 3] += y * y; s2[6 * tile + 4] += y * z; s2[6 * tile + 5] += z * z;
    }
    eig.assign(3 * T3, 0.f);
    for (int t = 0; t < T3; ++t) {
        int m = cnt[t]; if (m < 2)continue;
        double inv = 1.0 / m, mx = sum[3 * t + 0] * inv, my = sum[3 * t + 1] * inv, mz = sum[3 * t + 2] * inv;
        double d = 1.0 / (m - 1);
        double cxx = (s2[6 * t + 0] - m * mx * mx) * d, cxy = (s2[6 * t + 1] - m * mx * my) * d, cxz = (s2[6 * t + 2] - m * mx * mz) * d;
        double cyy = (s2[6 * t + 3] - m * my * my) * d, cyz = (s2[6 * t + 4] - m * my * mz) * d, czz = (s2[6 * t + 5] - m * mz * mz) * d;
        double l0, l1, l2; cpuEigen3x3(cxx, cyy, czz, cxy, cxz, cyz, l0, l1, l2);
        eig[3 * t] = l0; eig[3 * t + 1] = l1; eig[3 * t + 2] = l2;
    }
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int T = (argc > 2) ? std::atoi(argv[2]) : 64;
    printf("Eigen-per-tile  (n=%d  tiles/axis=%d)\n", n, T);

    std::vector<float> h_pts(3ULL * n);
    std::mt19937 gen(1234); std::uniform_real_distribution<float>U(-1000.f, 1000.f);
    for (float& v : h_pts) v = U(gen);

    float* d_pts; CHECK_CUDA(cudaMalloc(&d_pts, h_pts.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pts, h_pts.data(), h_pts.size() * sizeof(float), cudaMemcpyHostToDevice));

    float bb[6]; gpuMinMax(d_pts, n, bb);

    int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, 3ULL * n * sizeof(int)));
    dim3 blk(BLOCK_DIM); dim3 grd((n + blk.x - 1) / blk.x);
    pointToTile << <grd, blk >> > (d_pts, n, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], T, d_idx);
    CHECK_CUDA(cudaGetLastError());

    int T3 = T * T * T; float* d_eig; CHECK_CUDA(cudaMalloc(&d_eig, 3ULL * T3 * sizeof(float)));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    gpuEigenvaluesPerTile(d_pts, d_idx, n, T, d_eig);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_gpu = 0.f; cudaEventElapsedTime(&ms_gpu, e0, e1);

    std::vector<float> h_eig(3ULL * T3);
    CHECK_CUDA(cudaMemcpy(h_eig.data(), d_eig, h_eig.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> h_idx(3ULL * n);
    CHECK_CUDA(cudaMemcpy(h_idx.data(), d_idx, h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    auto cpu0 = std::chrono::high_resolution_clock::now();
    std::vector<float> ref; cpuEigenPerTile(h_pts.data(), h_idx.data(), n, T, ref);
    auto cpu1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(cpu1 - cpu0).count();

    bool ok = true; const float eps = 8e-1f;
    for (size_t i = 0; i < h_eig.size(); ++i) {
        float diff = fabsf(h_eig[i] - ref[i]);
        if (diff > eps && fabsf(ref[i]) > 0.f) {
            printf("Mismatch at %zu  gpu=%g  cpu=%g (diff=%g)\n", i, h_eig[i], ref[i], diff);
            ok = false; break;
        }
    }
    puts(ok ? "PASS: results within tolerance." : "FAIL: discrepancy detected.");

    printf("CPU  time : %8.3f ms\n", ms_cpu);
    printf("GPU  time : %8.3f ms\n", ms_gpu);

    cudaFree(d_pts); cudaFree(d_idx); cudaFree(d_eig);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ok ? 0 : 1;
}
