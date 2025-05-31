#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "gaussEval.h"
__device__ void PixelGaussians
(
    int numGaussians,   
       float2*    meanPtr,  float3*    covPtr,  float4*    copPtr,   float*     colPointerr,float*     depthsPtr,
                        float2                pix,        
    float*                outC,   
            float&outW,        
    float&outD,    
             float&                outT,       
    uint32_t& lastContributor,
                 uint32_t& contributor,   
             bool&                 done           
)
 {

    float minAlpha = 1.0f / 255.0f;   // ignore alpha below 0.004
             float minT = 0.0001f;       
    float maxAlpha = 0.99f;           



    float accumC[3];
    for (uint32_t iii = 0; iii < 3; ++iii)
        accumC[iii] = outC[iii];               // colour

                float accumW = outW;                     // weight
                float accumD = outD;                     // depth
    float accumT = outT;                     // transmittance
                     bool  localDone = done;                
    uint32_t localId = contributor;          


    float2* mP = meanPtr;
                float3* cP = covPtr;
    float4* oP = copPtr;
            
    
    
        float*    colChannel = colPointerr;
    float*  dP = depthsPtr;


    int i = 0;
    while (i < numGaussians && !localDone)
    {
        ++localId;  


        float g = functionEXP(*mP, *cP, pix);  

                    float a = fminf(maxAlpha, oP->w * g);         

        if (a >= minAlpha)           
        {

            float newT = accumT * (1.0f - a);

            if (newT >= minT)     
            {
                float aT = a * accumT;

                #pragma unroll
                for (uint32_t iii = 0; iii < 3; ++iii)
                    accumC[iii] = __fmaf_rn(  colChannel[iii], aT, accumC[iii]); 

                        accumW += aT;           
                accumD += (*dP) * aT;   
                        accumT  = newT;         
                lastContributor = localId;
            }
            else                       
            {
                localDone = true;
            }
        }


                    ++i;
        ++mP; ++cP; ++oP;   colChannel += 3; ++dP;
    }

    for (int iii = 0; iii < 3; ++iii)
    {
        outC[iii] = accumC[iii];
    }

    outW         = accumW;
    outD         = accumD;
    outT         = accumT;
    contributor  = localId;
    done         = localDone;
}
