#ifndef PROJECT_G_H_INCLUDED
#define PROJECT_G_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace PROJECTION 
{
void ProjectG(
    int numGaussians,
       float* means3D,
       glm::vec3* scales,
       glm::vec4* orientationss,
       float* matViewCam,
       float cameraFLenHorz,
    float* covvarance3D,
    int* radss,
    uint32_t* intersectedTiles,
    float3* covOP,
    float* detOP,
    float3* defOP,
       float* gaussianAlphas,
    float4* alphaConicOfgauss,
    float2* means2D,     
       float* mattProjCam,
    float* zDistances
);

}

#endif
