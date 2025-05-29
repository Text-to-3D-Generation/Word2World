#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#if defined(_WIN32) || defined(_WIN64)
  #define DLL_EXPORT extern "C" __declspec(dllexport)
#else
  #define DLL_EXPORT extern "C"
#endif


#define BLOCK_DIM 256
#define M_PI 3.1415926535
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr,"CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

__global__ void reduceMin256(float* points, int len, int stride)
{
    __shared__ float s[BLOCK_DIM];
    int idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    float* base = points;
    s[threadIdx.x] = (idx < len) ? base[idx] : FLT_MAX; __syncthreads();
    for (int st = BLOCK_DIM >> 1; st; st >>= 1) {
        if (threadIdx.x < st)
            s[threadIdx.x] = fminf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads();
    }
    if (threadIdx.x == 0) base[blockIdx.x] = s[0];

    base = points + stride;
    s[threadIdx.x] = (idx < len) ? base[idx] : FLT_MAX; __syncthreads();
    for (int st = BLOCK_DIM >> 1; st; st >>= 1) {
        if (threadIdx.x < st)
            s[threadIdx.x] = fminf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads();
    }
    if (threadIdx.x == 0) base[blockIdx.x] = s[0];

    base = points + 2 * stride;
    s[threadIdx.x] = (idx < len) ? base[idx] : FLT_MAX; __syncthreads();
    for (int st = BLOCK_DIM >> 1; st; st >>= 1) {
        if (threadIdx.x < st)
            s[threadIdx.x] = fminf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads();
    }
    if (threadIdx.x == 0) base[blockIdx.x] = s[0];
}
__global__ void reduceMax256(float* points, int len, int stride)
{
    __shared__ float s[BLOCK_DIM];
    int idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    float* base = points;
    s[threadIdx.x] = (idx < len) ? base[idx] : -FLT_MAX; __syncthreads();
    for (int st = BLOCK_DIM >> 1; st; st >>= 1) {
        if (threadIdx.x < st)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads();
    }
    if (threadIdx.x == 0) base[blockIdx.x] = s[0];

    base = points + stride;
    s[threadIdx.x] = (idx < len) ? base[idx] : -FLT_MAX; __syncthreads();
    for (int st = BLOCK_DIM >> 1; st; st >>= 1) {
        if (threadIdx.x < st)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads();
    }
    if (threadIdx.x == 0) base[blockIdx.x] = s[0];

    base = points + 2 * stride;
    s[threadIdx.x] = (idx < len) ? base[idx] : -FLT_MAX; __syncthreads();
    for (int st = BLOCK_DIM >> 1; st; st >>= 1) {
        if (threadIdx.x < st)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads();
    }
    if (threadIdx.x == 0) base[blockIdx.x] = s[0];
}

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

/* ------------------------------------------------------------------ */
/* -------------------- point → tile index (unchanged) -------------- */
/* ------------------------------------------------------------------ */
__global__ void pointToTile(const float* __restrict__ points, int n,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z,
    int num_tiles,
    int* __restrict__ out_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= n) return;
    float x = points[tid], y = points[n + tid], z = points[2 * n + tid];

    float fx = floorf(min_x), fy = floorf(min_y), fz = floorf(min_z);
    float cx = ceilf(max_x), cy = ceilf(max_y), cz = ceilf(max_z);

    float dx = (cx - fx) / num_tiles + 1e-20f;
    float dy = (cy - fy) / num_tiles + 1e-20f;
    float dz = (cz - fz) / num_tiles + 1e-20f;

    int ix = (int)((x - fx) / dx);
    int iy = (int)((y - fy) / dy);
    int iz = (int)((z - fz) / dz);
    ix = max(0, min(ix, num_tiles - 1));
    iy = max(0, min(iy, num_tiles - 1));
    iz = max(0, min(iz, num_tiles - 1));

    out_idx[tid] = ix;
    out_idx[n + tid] = iy;
    out_idx[2 * n + tid] = iz;
}

/* ------------------------------------------------------------------ */
/* -------- accumulate first & second moments (unchanged) ----------- */
/* ------------------------------------------------------------------ */
__global__ void accumulateTileMoments(const float* __restrict__ pts,
    const int* __restrict__ idx,
    int n, int T,
    int* __restrict__ dc,
    float* __restrict__ ds,
    float* __restrict__ dss)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= n) return;
    int ix = idx[tid], iy = idx[n + tid], iz = idx[2 * n + tid];
    int tile = ((iz * T) + iy) * T + ix;
    float x = pts[tid], y = pts[n + tid], z = pts[2 * n + tid];

    atomicAdd(dc + tile, 1);
    atomicAdd(ds + 3 * tile + 0, x); atomicAdd(ds + 3 * tile + 1, y); atomicAdd(ds + 3 * tile + 2, z);
    atomicAdd(dss + 6 * tile + 0, x * x); atomicAdd(dss + 6 * tile + 1, x * y); atomicAdd(dss + 6 * tile + 2, x * z);
    atomicAdd(dss + 6 * tile + 3, y * y); atomicAdd(dss + 6 * tile + 4, y * z); atomicAdd(dss + 6 * tile + 5, z * z);
}

__device__ __forceinline__
void eigSymm3x3(float a, float b, float c, float d, float e, float f, float& l0, float& l1, float& l2)
{
    float m = (a + b + c) / 3.f;
    float a0 = a - m, b0 = b - m, c0 = c - m;
    float p = sqrtf((a0 * a0 + b0 * b0 + c0 * c0 + 2.f * (d * d + e * e + f * f)) / 6.f);
    float invp = 1.f / (p + 1e-30f);
    a0 *= invp; b0 *= invp; c0 *= invp; d *= invp; e *= invp; f *= invp;
    float det = a0 * b0 * c0 + 2.f * d * e * f - a0 * f * f - b0 * e * e - c0 * d * d;
    float r = fmaxf(-1.f, fminf(1.f, det * 0.5f));
    float t = acosf(r) / 3.f;
    float c1 = 2.f * cosf(t);
    float c2 = 2.f * cosf(t + 4.f * M_PI / 3.f);
    float c3 = 2.f * cosf(t + 8.f * M_PI / 3.f);
    l0 = p * c1 + m; l1 = p * c2 + m; l2 = p * c3 + m;
    if (l0 < l1) { float tmp = l0; l0 = l1; l1 = tmp; }
    if (l1 < l2) { float tmp = l1; l1 = l2; l2 = tmp; }
    if (l0 < l1) { float tmp = l0; l0 = l1; l1 = tmp; }
}

__global__ void finaliseEigenvalues(int T3, const int* dc, const float* ds, const float* dss, float* deig)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x; if (tile >= T3) return;
    int n = dc[tile]; if (n < 2) { deig[3 * tile] = deig[3 * tile + 1] = deig[3 * tile + 2] = 0.f; return; }

    float invn = 1.f / n;
    float mx = ds[3 * tile] * invn, my = ds[3 * tile + 1] * invn, mz = ds[3 * tile + 2] * invn;
    float d = 1.f / (n - 1);
    float cxx = (dss[6 * tile] - n * mx * mx) * d;
    float cxy = (dss[6 * tile + 1] - n * mx * my) * d;
    float cxz = (dss[6 * tile + 2] - n * mx * mz) * d;
    float cyy = (dss[6 * tile + 3] - n * my * my) * d;
    float cyz = (dss[6 * tile + 4] - n * my * mz) * d;
    float czz = (dss[6 * tile + 5] - n * mz * mz) * d;

    float l0, l1, l2; eigSymm3x3(cxx, cyy, czz, cxy, cxz, cyz, l0, l1, l2);
    deig[3 * tile] = l0; deig[3 * tile + 1] = l1; deig[3 * tile + 2] = l2;
}

__global__ void finaliseStructureMetrics(int T3, const int* dc,
    const float* ds, const float* dss,
    float* dm)       
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= T3) return;

    int n = dc[tile];
    if (n < 2) {
        dm[4 * tile] = dm[4 * tile + 1] = dm[4 * tile + 2] = dm[4 * tile + 3] = 0.f;
        return;
    }

    float invn = 1.f / n;
    float mx = ds[3 * tile] * invn;
    float my = ds[3 * tile + 1] * invn;
    float mz = ds[3 * tile + 2] * invn;

    float d = 1.f / (n - 1);
    float cxx = (dss[6 * tile] - n * mx * mx) * d;
    float cxy = (dss[6 * tile + 1] - n * mx * my) * d;
    float cxz = (dss[6 * tile + 2] - n * mx * mz) * d;
    float cyy = (dss[6 * tile + 3] - n * my * my) * d;
    float cyz = (dss[6 * tile + 4] - n * my * mz) * d;
    float czz = (dss[6 * tile + 5] - n * mz * mz) * d;

    float l0, l1, l2;
    eigSymm3x3(cxx, cyy, czz, cxy, cxz, cyz, l0, l1, l2);

    if (l0 <= 0.f) {                  
        dm[4 * tile] = dm[4 * tile + 1] = dm[4 * tile + 2] = dm[4 * tile + 3] = 0.f;
        return;
    }

    float V = l0 + l1 + l2;
    float L = (l0 - l1) / l0;             /* linearity   */
    float P = (l1 - l2) / l0;             /* planarity   */
    float S = 1.f - L - P;                /* scattering  */

    L = fmaxf(0.f, fminf(1.f, L));
    P = fmaxf(0.f, fminf(1.f, P));
    S = fmaxf(0.f, fminf(1.f, S));

    dm[4 * tile] = V;
    dm[4 * tile + 1] = L;
    dm[4 * tile + 2] = P;
    dm[4 * tile + 3] = S;
}


void gpuEigenvaluesPerTile(const float* d_pts, const int* d_idx,
    int n, int T, float* d_out)
{
    int T3 = T * T * T;
    int* dc; float* ds; float* dss;
    CHECK_CUDA(cudaMalloc(&dc, T3 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ds, T3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dss, T3 * 6 * sizeof(float)));
    CHECK_CUDA(cudaMemset(dc, 0, T3 * sizeof(int)));
    CHECK_CUDA(cudaMemset(ds, 0, T3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(dss, 0, T3 * 6 * sizeof(float)));

    dim3 blk(BLOCK_DIM), grd((n + blk.x - 1) / blk.x);
    accumulateTileMoments << <grd, blk >> > (d_pts, d_idx, n, T, dc, ds, dss);
    CHECK_CUDA(cudaGetLastError());

    dim3 grd2((T3 + blk.x - 1) / blk.x);
    finaliseEigenvalues << <grd2, blk >> > (T3, dc, ds, dss, d_out);
    CHECK_CUDA(cudaGetLastError());

    cudaFree(dc); cudaFree(ds); cudaFree(dss);
}

void gpuStructureMetricsPerTile(const float* d_pts, const int* d_idx, int n, int T, float* d_out)
{
    int T3 = T * T * T;
    int* dc; float* ds; float* dss;
    CHECK_CUDA(cudaMalloc(&dc, T3 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ds, T3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dss, T3 * 6 * sizeof(float)));
    CHECK_CUDA(cudaMemset(dc, 0, T3 * sizeof(int)));
    CHECK_CUDA(cudaMemset(ds, 0, T3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(dss, 0, T3 * 6 * sizeof(float)));

    dim3 blk(BLOCK_DIM), grd((n + blk.x - 1) / blk.x);
    accumulateTileMoments << <grd, blk >> > (d_pts, d_idx, n, T, dc, ds, dss);
    CHECK_CUDA(cudaGetLastError());

    dim3 grd2((T3 + blk.x - 1) / blk.x);
    finaliseStructureMetrics << <grd2, blk >> > (T3, dc, ds, dss, d_out);
    CHECK_CUDA(cudaGetLastError());

    cudaFree(dc); cudaFree(ds); cudaFree(dss);
}

__constant__ float c_wL = 0.60f;   /* weight for linearity   */
__constant__ float c_wP = 0.30f;   /* weight for planarity   */
__constant__ float c_wS = 0.10f;   /* weight for scattering  */

__global__ void finaliseDensifyFlag(int   T3,
    const int* __restrict__ dc,
    const float* __restrict__ ds,
    const float* __restrict__ dss,
    float  threshold1,      /* V cut-off */
    float  threshold2,      /* score cut-off */
    unsigned char* __restrict__ d_flag)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= T3) return;

    int n = dc[tile];
    if (n < 2) { d_flag[tile] = 0;  return; }

    float invn = 1.f / n;
    float mx = ds[3 * tile] * invn;
    float my = ds[3 * tile + 1] * invn;
    float mz = ds[3 * tile + 2] * invn;

    float fac = 1.f / (n - 1);
    float cxx = (dss[6 * tile] - n * mx * mx) * fac;
    float cxy = (dss[6 * tile + 1] - n * mx * my) * fac;
    float cxz = (dss[6 * tile + 2] - n * mx * mz) * fac;
    float cyy = (dss[6 * tile + 3] - n * my * my) * fac;
    float cyz = (dss[6 * tile + 4] - n * my * mz) * fac;
    float czz = (dss[6 * tile + 5] - n * mz * mz) * fac;

    float l0, l1, l2; eigSymm3x3(cxx, cyy, czz, cxy, cxz, cyz, l0, l1, l2);
    if (l0 <= 0.f) { d_flag[tile] = 0; return; }

    float V = l0 + l1 + l2;
    if (V < threshold1) { d_flag[tile] = 0; return; }

    float L = (l0 - l1) / l0;
    float P = (l1 - l2) / l0;
    float S = 1.f - L - P;

    L = fminf(fmaxf(L, 0.f), 1.f);
    P = fminf(fmaxf(P, 0.f), 1.f);
    S = fminf(fmaxf(S, 0.f), 1.f);

    float score = c_wL * L + c_wP * P + c_wS * S;
    d_flag[tile] = (score >= threshold2) ? 1 : 0;
}

void gpuDensifyFlagPerTile(const float* d_pts,
    const int* d_idx,
    int          n,
    int          tiles,
    float        threshold1,
    float        threshold2,
    unsigned char* d_flag) 
{
    const int T3 = tiles * tiles * tiles;

    int* dc;   float* ds;   float* dss;
    CHECK_CUDA(cudaMalloc(&dc, T3 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ds, T3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dss, T3 * 6 * sizeof(float)));
    CHECK_CUDA(cudaMemset(dc, 0, T3 * sizeof(int)));
    CHECK_CUDA(cudaMemset(ds, 0, T3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(dss, 0, T3 * 6 * sizeof(float)));

    dim3 blk(BLOCK_DIM), grd((n + blk.x - 1) / blk.x);
    accumulateTileMoments << <grd, blk >> > (d_pts, d_idx, n, tiles, dc, ds, dss);
    CHECK_CUDA(cudaGetLastError());

    dim3 grd2((T3 + blk.x - 1) / blk.x);
    finaliseDensifyFlag << <grd2, blk >> > (T3, dc, ds, dss,
        threshold1, threshold2,
        d_flag);
    CHECK_CUDA(cudaGetLastError());

    cudaFree(dc); cudaFree(ds); cudaFree(dss);
}

__global__ void mapPointFlag(const int* __restrict__ tile_idx, int n, int T, const unsigned char* tile_flag, unsigned char* point_flag)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int ix = tile_idx[tid];
    int iy = tile_idx[n + tid];
    int iz = tile_idx[2 * n + tid];
    int tile = ((iz * T) + iy) * T + ix;

    point_flag[tid] = tile_flag[tile];
}

void gpuPointFlagPerPoint(const int* d_tile_idx,   
    int                 n,
    int                 tiles,
    const unsigned char* d_tile_flag, 
    unsigned char* d_point_flag) 
{
    dim3 blk(BLOCK_DIM), grd((n + blk.x - 1) / blk.x);
    mapPointFlag << <grd, blk >> > (d_tile_idx, n, tiles,
        d_tile_flag, d_point_flag);
    CHECK_CUDA(cudaGetLastError());
}

DLL_EXPORT
void densify_point_flags(const float* h_pts,
    int          n,
    int          tiles,
    float        th1,
    float        th2,
    unsigned char* h_flag_out)
{
    float* d_pts;
    CHECK_CUDA(cudaMalloc(&d_pts, 3ULL * n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pts, h_pts,
        3ULL * n * sizeof(float),
        cudaMemcpyHostToDevice));

    float bb[6];
    {
        gpuMinMax(d_pts, n, bb);
    }

    int* d_idx;  
    CHECK_CUDA(cudaMalloc(&d_idx, 3ULL * n * sizeof(int)));

    dim3 blk(BLOCK_DIM), grd((n + blk.x - 1) / blk.x);
    pointToTile << <grd, blk >> > (
        d_pts, n,
        bb[0], bb[1], bb[2], bb[3], bb[4], bb[5],
        tiles,
        d_idx);
    CHECK_CUDA(cudaGetLastError());

    const int T3 = tiles * tiles * tiles;
    unsigned char* d_tile_flag;
    CHECK_CUDA(cudaMalloc(&d_tile_flag, T3));
    gpuDensifyFlagPerTile(d_pts, d_idx, n, tiles,
        th1, th2, d_tile_flag);

    unsigned char* d_point_flag;
    CHECK_CUDA(cudaMalloc(&d_point_flag, n));
    gpuPointFlagPerPoint(d_idx, n, tiles,
        d_tile_flag, d_point_flag);

    CHECK_CUDA(cudaMemcpy(h_flag_out, d_point_flag,
        n, cudaMemcpyDeviceToHost));

    cudaFree(d_pts);
    cudaFree(d_idx);
    cudaFree(d_tile_flag);
    cudaFree(d_point_flag);
}
