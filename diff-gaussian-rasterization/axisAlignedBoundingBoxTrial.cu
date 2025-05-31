#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>  
#include <algorithm> 

extern "C" __global__ void CullUsingAxisAlignedBoundingBox(
    int     numGaussians,
    float2* means2D,          // [numGaussians]
    
     int*    radss,            // [numGaussians]  


    int     tilesX,
    int     tilesY,
                int*    aabbTL,           // [numGaussians*2] out
    int*    aabbBR            // [numGaussians*2] out
) 
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numGaussians)
    {
         return;
    }

float2 c = means2D[gid];
    int    r = radss[gid];

    // pixel bounds (inclusive)
    int pxMIN = static_cast<int>(floorf(c.x - r));
                int pyMIN = static_cast<int>(floorf(c.y - r));
    int pxMAX = static_cast<int>(ceilf (c.x + r));


    int pyMAX = static_cast<int>(ceilf (c.y + r));

    // convert to tile indices, clamp to grid
    int txMIN = max(0,          pxMIN / 16);
int tyMIN = max(0,          pyMIN / 16);

            int txMAX = min(tilesX - 1, pxMAX / 16);
    int tyMAX = min(tilesY - 1, pyMAX / 16);

aabbTL[gid * 2 + 0] = txMIN;
    
 aabbTL[gid * 2 + 1] = tyMIN;
        aabbBR[gid * 2 + 0] = txMAX;
    aabbBR[gid * 2 + 1] = tyMAX;
}
