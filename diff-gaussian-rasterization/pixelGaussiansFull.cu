#include "pixelGaussiansFull.h"
#include "pixelGaussiansDev.cuh"
__global__ void __launch_bounds__(256)
runPixelGaussiansBatches
(
   uint2*     tileSliceStartEnd, uint32_t*    sortedIds,  float2*     coords2D,
                               float*      featArr,
                               float*      depthArr,
                               float4*     conicArr,
                            float*            alphaOut,
                            uint32_t*         contribOut,
                            float*            colorOut,
                            float*            depthOut,
                               float3*     covArr)
{
    int colsss = 3;

                unsigned bx = blockIdx.x, by = blockIdx.y;             // tile coords
    unsigned tx = threadIdx.x, ty = threadIdx.y;           // pixel-in-tile
                unsigned px = bx * 16u + tx;                           // absolute pixel x
    unsigned py = by * 16u + ty;                           // absolute pixel y
            unsigned pid = py * 800 + px;                          // 1-shNum pixel index
       float2 pixelF = { float(px), float(py) };        // float coords

    unsigned gridW = 50u;                    // tiles in X


        
    bool active   = (px < 800u && py < 800u);
            
            bool finished = !active;                               // skip dead pixels


    uint2 range = tileSliceStartEnd[by * gridW + bx];           
            
    
    
        int   totalG = int(range.y - range.x);              


    __shared__ float2 fastMemMuu [256];
    
    
            __shared__ float3 fastMCovs  [256];
    __shared__ float4 fastMemConi [256];
   
   
      __shared__ float  fastMemDepth [256];
    __shared__ float  fastMFeat [256 * 3];
    float trans  = 1.0f;                   
            float accum  [3] = {0.0f};      
    float accumW = 0.0f;                  
            
    
            float accumD = 0.0f;               
    uint32_t lastCid = 0, cid = 0;         
    for (int offset = 0; offset < totalG; offset += 256)
    {
        int batchSize = min(256, totalG - offset);
        int flatId    = ty * blockDim.x + tx;           
                        int THREADID = range.x + offset + flatId;
        if (flatId < batchSize) {
            uint32_t gid = sortedIds[THREADID];
                        fastMemMuu [flatId] = coords2D[gid];
            fastMCovs  [flatId] = covArr  [gid];
                    
            
            fastMemConi [flatId] = conicArr[gid];  
                        fastMemDepth [flatId] = depthArr[gid];
            for (int   colChannel = 0;   colChannel < 3; ++  colChannel)
            {
                fastMFeat[flatId * 3 +   colChannel] =
                    featArr[gid * 3 +   colChannel];
            }
        }
        __syncthreads();   // ensure shared buffers are filled
        if (active && !finished) 
        {
            PixelGaussians(
                batchSize,fastMemMuu, fastMCovs, fastMemConi, fastMFeat, fastMemDepth,pixelF,accum,  accumW, accumD,trans,
                lastCid, cid,
                finished);
        }
        __syncthreads();   // all pixels done with shared data
    }
    if (active) 
    {
        contribOut[pid] = lastCid;         
                        
                alphaOut  [pid] = accumW;       
        depthOut  [pid] = accumD;   
        for (int   colChannel = 0;   colChannel < 3; ++  colChannel) 
        {
            unsigned idxC =   colChannel * 640000 + pid;       
            colorOut[idxC] = accum[  colChannel] + trans * 1.0f; 
        }
    }
}
void pixelGaussiansFull
(dim3 block,uint2* tileSliceStartEnd,uint32_t* MainGaussianIdArray,float2* means2D,float* colors,float* zDistances,
       float4* alphaConicOfgauss,
    float* opacityOP,
    uint32_t* NContributedGaussians,
    float* colOP,
    float* ZOP,
       float3* covOP
)
 {
    dim3 grid(50, 50); 
    runPixelGaussiansBatches<<<grid, block>>>(
        tileSliceStartEnd,
        MainGaussianIdArray,
        means2D,
        colors,
        zDistances,
        alphaConicOfgauss,
        opacityOP,
        NContributedGaussians,
        colOP,
        ZOP,
        covOP
    );
}
