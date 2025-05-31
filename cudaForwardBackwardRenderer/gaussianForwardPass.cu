#include "gaussianForwardPass.h"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>





__device__ void evalSHDeg1(   glm::vec3& d,    glm::vec3* sphericalH, glm::vec3& out) {
    out += -0.4886025119029199f * d.y * sphericalH[1];
    out +=  0.4886025119029199f * d.z * sphericalH[2];
    out += -0.4886025119029199f * d.x * sphericalH[3];
}

__device__ void evalSHDeg2(   glm::vec3& d,    glm::vec3* sphericalH, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
    float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;

    out += 1.0925484305920792f * xy                * sphericalH[4];
    out += -1.0925484305920792f * yz                * sphericalH[5];
out += glm::vec3(0.3f * (2 * zz - xx - yy)) * sphericalH[6];

    out += -1.0925484305920792f * xz                * sphericalH[7];
    out += 0.5462742152960396f * (xx - yy)         * sphericalH[8];
}

__device__ void evalSHDeg3(   glm::vec3& d,    glm::vec3* sphericalH, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;

    out += -0.5900435899266435f * d.y * (3 * xx - yy)          * sphericalH[9];
    out += 2.890611442640554f * d.x * d.y * d.z              * sphericalH[10];
    out += -0.4570457994644658f * d.y * (4 * zz - xx - yy)     * sphericalH[11];
    out += 0.3731763325901154f * d.z * (2 * zz - 3 * xx - 3 * yy) * sphericalH[12];
    out += -0.4570457994644658f * d.x * (4 * zz - xx - yy)     * sphericalH[13];
    out += 1.445305721320277f * d.z * (xx - yy)              * sphericalH[14];
    out += -0.5900435899266435f * d.x * (xx - 3 * yy)          * sphericalH[15];
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
//     int THREADID = blockIdx.x * blockDim.x + threadIdx.x;
//     if (THREADID >= numGaussians) return;

//     float3 covarianceMat = covOP[THREADID];
//     float determinent = detOP[THREADID];

//     float mid = 0.5f * (covarianceMat.x + covarianceMat.z);
//     float Y1 = mid + sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float Y2 = mid - sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float YMAX = fmaxf(Y1, Y2);

//     // 99% confidence radius
//     float allProbRadius = 3.f * sqrtf(YMAX);

//     // Early exit on low opacity
//     float opacity = gaussianAlphas[THREADID];
//     if (opacity <= 0.0039f) {
//         radss[THREADID] = 0;
//         intersectedTiles[THREADID] = 0;
//         return;
//     }

//     // Axis-aligned bounding box half extents (Eq. 10 & 11)
//     float lnRatio = logf(opacity * 255.f);
//     float r_x = sqrtf(2.f * covarianceMat.z * lnRatio); // covarianceMat.y ≡ Σ'_Y → x dir
//     float r_y = sqrtf(2.f * covarianceMat.x * lnRatio); // covarianceMat.x ≡ Σ'_X → y dir
//     r_x = fminf(r_x, allProbRadius);
//     r_y = fminf(r_y, allProbRadius);

//     // Keep old interface: write max(r_x, r_y) to radss
//     radss[THREADID] = ceilf(fmaxf(r_x, r_y));

//     float2 p = muu2dPixelCoord[THREADID];

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

//     intersectedTiles[THREADID] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
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
//     int THREADID = blockIdx.x * blockDim.x + threadIdx.x;
//     if (THREADID >= numGaussians) return;

//     float3 covarianceMat = covOP[THREADID];
//     float determinent = detOP[THREADID];

//     float mid = 0.5f * (covarianceMat.x + covarianceMat.z);
//     float Y1 = mid + sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float Y2 = mid - sqrtf(fmaxf(0.1f, mid * mid - determinent));
//     float YMAX = fmaxf(Y1, Y2);

//     float allProbRadius = 3.f * sqrtf(YMAX);
//     float opacity = gaussianAlphas[THREADID];

//     if (opacity <= 0.0039f) {
//         radss[THREADID]        = 0;
//         rx[THREADID]      = 0;
//         ry[THREADID]      = 0;
//         intersectedTiles[THREADID]= 0;
//         return;
//     }

//     float lnRatio = logf(opacity * 255.f);
//     float r_x = sqrtf(2.f * covarianceMat.z * lnRatio);
//     float r_y = sqrtf(2.f * covarianceMat.x * lnRatio);
//     r_x = fminf(r_x, allProbRadius);
//     r_y = fminf(r_y, allProbRadius);

//     rx[THREADID] = ceilf(r_x);
//     ry[THREADID] = ceilf(r_y);
//     radss[THREADID]   = ceilf(fmaxf(r_x, r_y));  // preserve old interface behavior

//     float2 p = muu2dPixelCoord[THREADID];
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

//     intersectedTiles[THREADID] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
// }

//we want to compute rgb for each Gaussian
//will do so using spherical harmonics
__global__ void computeSHColorCUDA(
    int               numGaussians,
    int               shNum,
    int               M,
       float*      DefaultPoints,  // reinterpret as glm::vec3*
                glm::vec3*  positionOfCamera,      // pointer to a single camera pos
        float*      sphericalHarmonics,          // spherical harmonics coefficients
    bool*       backwardBoolean,
          
    
float*      rgb           // length numGaussians*3
) 


{
    int THREADID = blockIdx.x * blockDim.x + threadIdx.x;
    if (THREADID >= numGaussians)
    { 
        //printf("no sphericalH !");
        return;
    }


    glm::vec3 meanToCam = glm::normalize(glm::make_vec3(&DefaultPoints[3 * THREADID]) - *positionOfCamera);

    glm::vec3* sphericalH = ((glm::vec3*)sphericalHarmonics) + THREADID * M;
    
    glm::vec3 resFina = 0.28209479177387814f * sphericalH[0];
    

    //depending on deg we evalute
                if (shNum >= 1)
                { 
                    
                    evalSHDeg1(meanToCam, sphericalH, resFina);

                }
    
                if (shNum >= 2)
                {
                     evalSHDeg2(meanToCam, sphericalH, resFina);
                }


    if (shNum >= 3)
    { evalSHDeg3(meanToCam, sphericalH, resFina);
    }
    
    resFina += 0.5f;
    


int baseIndex = 3 * THREADID;

if (resFina.x < 0.0f) {
    backwardBoolean[baseIndex + 0] = true;
} 

else 
{
    
    
            backwardBoolean[baseIndex + 0] = false;
}

if (resFina.y < 0.0f) 
{
    backwardBoolean[baseIndex + 1] = true;
} 

else {
    backwardBoolean[baseIndex + 1] = false;
}

                if (resFina.z < 0.0f) 

{
    
    backwardBoolean[baseIndex + 2] = true;
} else {
    backwardBoolean[baseIndex + 2] = false;
}


    
    glm::vec3 finaleCo = glm::max(resFina, 0.0f);
    

    float* base = &rgb[THREADID * 3];
    glm::vec3* rgb_out = reinterpret_cast<glm::vec3*>(base);
    //final col
*rgb_out = resFina;
    
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
    bool* backwardBoolean,       
    float* rgb
) {


    computeSHColorCUDA<<<(numGaussians + 255) / 256, 256>>>(
        numGaussians,
        shNum, M,
        means3D,
        positionOfCamera,
        sphericalHarmonics,
        backwardBoolean,
        rgb
    );
}