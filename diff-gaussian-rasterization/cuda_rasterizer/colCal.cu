#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define NUM_CHANNELS 3
#include "colCal.h"
#include "auxiliary.h"



__device__ __forceinline__ void evalSHDeg1(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    out += -SH_C1 * d.y * sh[1];
    out +=  SH_C1 * d.z * sh[2];
    out += -SH_C1 * d.x * sh[3];
}

__device__ __forceinline__ void evalSHDeg2(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
    float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;

    out += SH_C2[0] * xy                 * sh[4];
    out += SH_C2[1] * yz                 * sh[5];
    out += SH_C2[2] * (2 * zz - xx - yy) * sh[6];
    out += SH_C2[3] * xz                 * sh[7];
    out += SH_C2[4] * (xx - yy)          * sh[8];
}

__device__ __forceinline__ void evalSHDeg3(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;

    out += SH_C3[0] * d.y * (3 * xx - yy)              * sh[9];
    out += SH_C3[1] * d.x * d.y * d.z                  * sh[10];
    out += SH_C3[2] * d.y * (4 * zz - xx - yy)         * sh[11];
    out += SH_C3[3] * d.z * (2 * zz - 3 * xx - 3 * yy) * sh[12];
    out += SH_C3[4] * d.x * (4 * zz - xx - yy)         * sh[13];
    out += SH_C3[5] * d.z * (xx - yy)                  * sh[14];
    out += SH_C3[6] * d.x * (xx - 3 * yy)              * sh[15];
}



// assume C==3 for RGB; if you ever have other channel counts, you can template it

__global__ void computeSHColorCUDA(
    int               numGaussians,
    int               shNum,
    int               M,
       float*      DefaultPoints,  // reinterpret as glm::vec3*
       glm::vec3*  positionOfCamera,      // pointer to a single camera pos
       float*      sphericalHarmonics,          // spherical harmonics coefficients
     bool*       clamped,
          float*      rgb           // length numGaussians*C
) {
    int C = 3;
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
    

    float* base = &rgb[idx * C];
    glm::vec3* rgb_out = reinterpret_cast<glm::vec3*>(base);
    *rgb_out = result;
    
}


namespace COLCAL
{
void calcColour(
    int numGaussians, int shNum, int M,
       float* means3D,
       glm::vec3* positionOfCamera,
       float* sphericalHarmonics,
    bool* clamped,       // ✅ MUST BE   
    float* rgb
) {


    computeSHColorCUDA<NUM_CHANNELS><<<(numGaussians + 255) / 256, 256>>>(
        numGaussians,
        shNum, M,
        means3D,
        positionOfCamera,
        sphericalHarmonics,
        clamped,
        rgb
    );
}

}