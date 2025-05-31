#ifndef PIXEL_GAUSSIANS_FULL_H
#define PIXEL_GAUSSIANS_FULL_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

void pixelGaussiansFull(
 dim3 block,
       uint2* tileSliceStartEnd,
       uint32_t* MainGaussianIdArray,
       float2* means2D,
       float* colors,
       float* zDistances,
       float4* alphaConicOfgauss,
    float* opacityOP,
    uint32_t* NContributedGaussians,
    float* colOP,
    float* ZOP,
       float3* covOP
);

#endif 
