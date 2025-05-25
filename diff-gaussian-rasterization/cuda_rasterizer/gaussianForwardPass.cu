#include "gaussianForwardPass.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 256




__device__ void evalSHDeg1(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    out += -SH_C1 * d.y * sh[1];
    out +=  SH_C1 * d.z * sh[2];
    out += -SH_C1 * d.x * sh[3];
}

__device__ void evalSHDeg2(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
    float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;

    out += SH_C2[0] * xy                * sh[4];
    out += SH_C2[1] * yz                * sh[5];
    out += SH_C2[2] * (2 * zz - xx - yy) * sh[6];
    out += SH_C2[3] * xz                * sh[7];
    out += SH_C2[4] * (xx - yy)         * sh[8];
}

__device__ void evalSHDeg3(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;

    out += SH_C3[0] * d.y * (3 * xx - yy)          * sh[9];
    out += SH_C3[1] * d.x * d.y * d.z              * sh[10];
    out += SH_C3[2] * d.y * (4 * zz - xx - yy)     * sh[11];
    out += SH_C3[3] * d.z * (2 * zz - 3 * xx - 3 * yy) * sh[12];
    out += SH_C3[4] * d.x * (4 * zz - xx - yy)     * sh[13];
    out += SH_C3[5] * d.z * (xx - yy)              * sh[14];
    out += SH_C3[6] * d.x * (xx - 3 * yy)          * sh[15];
}















// tile_culling.cu

// template<int 3>
// __global__ void  tileCullingCUDAAABB(
//     int               numGaussians,
//        float*      gaussianAlphas,
//        float3*     covOP,
//        float*      detOP,
//        float2*     muu2dPixelCoord,
//     dim3              grid,           // number of 16×16 tiles in x/y
//     int*              radss,
//     uint32_t*         intersectedTiles
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numGaussians) return;

//     float3 covarianceMat = covOP[idx];
//     float determinent = detOP[idx];

//     float mid = 0.5f * (covarianceMat.x + covarianceMat.z);
//     float Y1 = mid + sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float Y2 = mid - sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float YMAX = fmaxf(Y1, Y2);

//     // 99% confidence radius
//     float allProbRadius = 3.f * sqrtf(YMAX);

//     // Early exit on low opacity
//     float opacity = gaussianAlphas[idx];
//     if (opacity <= 0.0039f) {
//         radss[idx] = 0;
//         intersectedTiles[idx] = 0;
//         return;
//     }

//     // Axis-aligned bounding box half extents (Eq. 10 & 11)
//     float lnRatio = logf(opacity * 255.f);
//     float r_x = sqrtf(2.f * covarianceMat.z * lnRatio); // covarianceMat.y ≡ Σ'_Y → x dir
//     float r_y = sqrtf(2.f * covarianceMat.x * lnRatio); // covarianceMat.x ≡ Σ'_X → y dir
//     r_x = fminf(r_x, allProbRadius);
//     r_y = fminf(r_y, allProbRadius);

//     // Keep old interface: write max(r_x, r_y) to radss
//     radss[idx] = ceilf(fmaxf(r_x, r_y));

//     float2 p = muu2dPixelCoord[idx];

//     // Tile rectangle [min, max)
//     int2 tmin = make_int2(
//         int((p.x - r_x) / 16.0f),
//         int((p.y - r_y) / 16.0f)
//     );
//     int2 tmax = make_int2(
//         int((p.x + r_x + 15.0f) / 16.0f),
//         int((p.y + r_y + 15.0f) / 16.0f)
//     );

//     uint2 rect_min = make_uint2(
//         max(0, tmin.x), max(0, tmin.y)
//     );
//     uint2 rect_max = make_uint2(
//         min(grid.x, tmax.x), min(grid.y, tmax.y)
//     );

//     intersectedTiles[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
// }


// template<int 3>
// __global__ void tileCullingAABBCUDA(
//     int               numGaussians,
//        float*      gaussianAlphas,
//        float3*     covOP,
//        float*      detOP,
//        float2*     muu2dPixelCoord,
//     dim3              grid,               // number of 16×16 tiles in x/y
//     int*              radss,              // circle-based fallback (max(r_x, r_y))
//     int*              rx,            // AABB width radius
//     int*              ry,            // AABB height radius
//     uint32_t*         intersectedTiles
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numGaussians) return;

//     float3 covarianceMat = covOP[idx];
//     float determinent = detOP[idx];

//     float mid = 0.5f * (covarianceMat.x + covarianceMat.z);
//     float Y1 = mid + sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float Y2 = mid - sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float YMAX = fmaxf(Y1, Y2);

//     float allProbRadius = 3.f * sqrtf(YMAX);
//     float opacity = gaussianAlphas[idx];

//     if (opacity <= 0.0039f) {
//         radss[idx]        = 0;
//         rx[idx]      = 0;
//         ry[idx]      = 0;
//         intersectedTiles[idx]= 0;
//         return;
//     }

//     float lnRatio = logf(opacity * 255.f);
//     float r_x = sqrtf(2.f * covarianceMat.z * lnRatio);
//     float r_y = sqrtf(2.f * covarianceMat.x * lnRatio);
//     r_x = fminf(r_x, allProbRadius);
//     r_y = fminf(r_y, allProbRadius);

//     rx[idx] = ceilf(r_x);
//     ry[idx] = ceilf(r_y);
//     radss[idx]   = ceilf(fmaxf(r_x, r_y));  // preserve old interface behavior

//     float2 p = muu2dPixelCoord[idx];
//     int2 tmin = make_int2(
//         int((p.x - r_x) / 16.0f),
//         int((p.y - r_y) / 16.0f)
//     );
//     int2 tmax = make_int2(
//         int((p.x + r_x + 15.0f) / 16.0f),
//         int((p.y + r_y + 15.0f) / 16.0f)
//     );

//     uint2 rect_min = make_uint2(max(0, tmin.x), max(0, tmin.y));
//     uint2 rect_max = make_uint2(min(grid.x, tmax.x), min(grid.y, tmax.y));

//     intersectedTiles[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
// }

// sh_coloring.cu

// assume 3==3 for RGB; if you ever have other channel counts, you can template it
__global__ void computeSHColorCUDA(
    int               numGaussians,
    int               shNum,
    int               M,
       float*      DefaultPoints,  // reinterpret as glm::vec3*
       glm::vec3*  positionOfCamera,      // pointer to a single camera pos
       float*      sphericalHarmonics,          // spherical harmonics coefficients
     bool*       clamped,
          float*      rgb           // length numGaussians*3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numGaussians) return;

    // computeColorFromSH signature:
    // glm::vec3 computeColorFromSH(int idx, int shNum, int M,
    //                                 glm::vec3* pts,
    //                              glm::vec3 cam,
    //                                 float* sphericalHarmonics,
    //                                 bool* clamped);
    glm::vec3 dir = glm::normalize(glm::make_vec3(&DefaultPoints[3 * idx]) - *positionOfCamera);

    glm::vec3* sh = ((glm::vec3*)sphericalHarmonics) + idx * M;
    
    glm::vec3 result = SH_C0 * sh[0];
    
    if (shNum >= 1) evalSHDeg1(dir, sh, result);
    if (shNum >= 2) evalSHDeg2(dir, sh, result);
    if (shNum >= 3) evalSHDeg3(dir, sh, result);
    
    result += 0.5f;
    
    clamped[3 * idx + 0] = (result.x < 0.0f);
    clamped[3 * idx + 1] = (result.y < 0.0f);
    clamped[3 * idx + 2] = (result.z < 0.0f);
    
    glm::vec3 final_color = glm::max(result, 0.0f);
    

    float* base = &rgb[idx * 3];
    glm::vec3* rgb_out = reinterpret_cast<glm::vec3*>(base);
    *rgb_out = result;
    
}
__device__ inline void carry(uint32_t N, uint32_t dsize, float *sm,    float *gm, int *offset, int *gaussian_ids)
{
    int local_id = threadIdx.y * blockDim.x + threadIdx.x;
    int n_turns = (dsize * N) / blockDim.x;
    int n_left  = (dsize * N) % blockDim.x;

    for (int i = 0; i < n_turns; i++) {
        sm[local_id + i * blockDim.x] =
            gm[dsize * gaussian_ids[(local_id + i * blockDim.x) / dsize] +
               (local_id + i * blockDim.x) % dsize];
    }

    if (local_id < n_left) {
        sm[local_id + n_turns * blockDim.x] =
            gm[dsize * gaussian_ids[(local_id + n_turns * blockDim.x) / dsize] +
               (local_id + n_turns * blockDim.x) % dsize];
    }
}











 
 


 
 








// void FORWARD::preprocessTileCullingAABB(
//     int numGaussians,
//        float* gaussianAlphas,
//        float3* covOP,
//        float* detOP,
//        float2* means2D,
//        dim3   grid,
//     int*         radss,
//     int*         rx,
//     int*         ry,
//     uint32_t*    intersectedTiles
// ) {
//     tileCullingAABBCUDA<NUM_CHANNELS><<<(numGaussians + 255) / 256, 256>>>(
//         numGaussians,
//         gaussianAlphas,
//         covOP,
//         detOP,
//         means2D,
//         grid,
//         radss,
//         rx,
//         ry,
//         intersectedTiles
//     );
//     CHECK_CUDA(cudaGetLastError(), "tileCullingCUDAAABB");
// }


void FORWARD::calcColour(
    int numGaussians, int shNum, int M,
       float* means3D,
       glm::vec3* positionOfCamera,
       float* sphericalHarmonics,
    bool* clamped,       // ✅ MUST BE   
    float* rgb
) {


    computeSHColorCUDA<<<(numGaussians + 255) / 256, 256>>>(
        numGaussians,
        shNum, M,
        means3D,
        positionOfCamera,
        sphericalHarmonics,
        clamped,
        rgb
    );
    CHECK_CUDA(cudaGetLastError(), "computeSHColorCUDA");
}