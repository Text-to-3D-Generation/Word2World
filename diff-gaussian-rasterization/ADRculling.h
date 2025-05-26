#ifndef ADR_H_INCLUDED
#define ADR_H_INCLUDED
#include <torch/extension.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



namespace ADR {

/// Preprocess step for tile-based culling of 2D Gaussians
void tileCulling(
    int numGaussians,
       float* __restrict__ gaussianAlphas,
       float3* __restrict__ covOP,
       float* __restrict__ detOP,
       float2* __restrict__ means2D,
    dim3 grid,
    int* __restrict__ radss,
    uint32_t* __restrict__ intersectedTiles
);

} 

#endif 
