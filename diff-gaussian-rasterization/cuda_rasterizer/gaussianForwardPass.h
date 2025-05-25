
 #ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
// 	// Perform initial steps for each Gaussian prior to rasterization.
	 void calcColour(
    int numGaussians, int shNum, int M,
       float* means3D,
       glm::vec3* positionOfCamera,
       float* sphericalHarmonics,
     bool* clamped,       // ✅ MUST BE   
    float* rgb
);


// void tileCulling(
//     int numGaussians,
//        float* gaussianAlphas,
//        float3* covOP,
//        float* detOP,
//        float2* means2D,
//        dim3   grid,
//     int*         radss,
//     uint32_t*    intersectedTiles
// );
// void preprocessTileCullingAABB(
//   int numGaussians,
//      float* gaussianAlphas,
//      float3* covOP,
//      float* detOP,
//      float2* means2D,
//      dim3   grid,
//   int*         radss,
//   int*         rx,
//   int*         ry,
//   uint32_t*    intersectedTiles
// );

	// Main rasterization method.
	// void render(
	// 	   dim3 grid, dim3 block,
	// 	   uint2* tileSliceStartEnd,
	// 	   uint32_t* MainGaussianIdArray,
	// 	int W, int H,
	// 	   float2* muu2dPixelCoord,
	// 	   float* features,
	// 	   float* zDistances,
	// 	   float4* alphaConicOfgauss,
	// 	float* opacityOP,
	// 	uint32_t* NContributedGaussians,
	// 	   float* bg_color,
	// 	float* colOP,
	// 	float* ZOP,
    //    float3* covOP);



}


#endif