
#include "ADRculling.h"

//now that we have our 2d gaussians and startingg our tile based rasterization,
//we need to cut of the gaussian tails to be more efficient
//we will take min of AdpativeRadius and 3 std div radius for such case
__global__ void tileCullingCUDA(
    int               numGaussians, // number of Gaussians!
      float*      gaussianAlphas, // alpha values of Gaussians!
      float3*     covOP, // covariance matrix of Gaussians!
      float*      detOP, //deteerminants
      float2*     muu2dPixelCoord, //2d means
    dim3              grid,           // number of 16×16 tiles in x/y
          int*         radss, //radiuses,, will put to zero if want to cull
    uint32_t*         intersectedTiles //finale
) 
{
       //linear index for the current Gaussian
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // If the index exceeds the number of Gaussians, exit early
      if (idx >= numGaussians)
    {   
        //printf("OOOOOOUUUUTTTTTTYYYYYYY\n");

        return;
    }

    
  
    //simply obtain the cov and dep 
    
    
    
    float3 covarianceMat = covOP[idx];
    
    
    float determinant = detOP[idx];

    // Compute eigenvalues Y! and Y2 of the covariance matrix
    float trace = covarianceMat.x + covarianceMat.z;
                float halfTrace = trace * 0.5f;

            float deltaSquared = halfTrace * halfTrace - determinant;
float rootPart = sqrtf(fmaxf(0.1f, deltaSquared));
//printf("rootPart: %f\n", rootPart);
        float Y1 = halfTrace + rootPart;
float Y2 = halfTrace - rootPart;

float YMMAX = fmaxf(Y1, Y2);  // Always take the max

    //the rad for 3std div, either picked or used
    float allProbRadius = 3.f * sqrtf(YMMAX);

    //so if the gaussian has alpha low just ignore it
    
    
    if (
        
           gaussianAlphas[idx] <= 0.0039f) {
        radss[idx] = 0;
            intersectedTiles[idx] = 0;
        return;
    }

    //adaptiveRadius equation
    
    
    float lnRatio = logf(gaussianAlphas[idx] * 255.0f);  // α_low = 1/255
         float adaptiveRadius = sqrtf(2.f * YMMAX * lnRatio);

    //pick min, math thingy
    float minRad = fminf(adaptiveRadius, allProbRadius);

    //radius will definetly be integer since pixel and such 
    radss[idx] = ceilf(minRad);

   //boundings in tile space
        int2 tmin = make_int2(
         int((muu2dPixelCoord[idx].x - minRad) / 16.0f),
int((muu2dPixelCoord[idx].y - minRad) / 16.0f)
    );
    int2 tmax = make_int2(
        
            int((muu2dPixelCoord[idx].x + minRad + 15) / 16.0f),
int((muu2dPixelCoord[idx].y + minRad + 15) / 16.0f)
    );

   //do not go out of screen, else the ghost error
    uint2 rect_min = make_uint2(
             max(0, tmin.x),
             
             
             max(0, tmin.y)
    );
    uint2 rect_max = make_uint2(
        min(grid.x, 
            
            tmax.x),  min(grid.y,
                
                tmax.y)
    );

    //total tiles = total vert * total hor
    //later used 
    uint32_t totHorz  = rect_max.x - rect_min.x;

        uint32_t totvert= rect_max.y - rect_min.y;
intersectedTiles[idx] = totHorz * totvert;

}

//simplle  wrapper for the kernel
namespace ADR {

    void tileCulling(
        int numGaussians,
                        float* gaussianAlphas,
          float3* covOP,
          float* detOP,
                    float2* means2D,
          dim3   grid,
                    int*         radss,
uint32_t*    intersectedTiles
    ) {
    
    
tileCullingCUDA<<<(numGaussians + 255) / 256, 256>>>(
            numGaussians,
                 gaussianAlphas,
            covOP,
    detOP,
    means2D ,
            
         grid,
    radss,
            intersectedTiles
        );
        
    }

} 
