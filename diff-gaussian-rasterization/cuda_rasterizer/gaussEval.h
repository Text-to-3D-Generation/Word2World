#ifndef GAUSS_EVAL_H_INCLUDED
#define GAUSS_EVAL_H_INCLUDED

#include <cuda_runtime.h>
#include <device_functions.h>  // For expf

__device__ __forceinline__ float EvaluateGaussFunction
(
       float2 mu, 
       float3 covarianceMat, 
       float2 queryPoint)
{
    float cov_xx = covarianceMat.x;
    float cov_xy = covarianceMat.y;
    float cov_yy = covarianceMat.z;
    float determinent = cov_xx * cov_yy - cov_xy * cov_xy;

    float dx = queryPoint.x - mu.x;
    float dy = queryPoint.y - mu.y;

    float termX = dx * cov_yy - dy * cov_xy;
    float termY = -dx * cov_xy + dy * cov_xx;
    float mahalDist = (termX * dx + termY * dy) / determinent;

    if (mahalDist < 0.0f)
        mahalDist = 1000.0f;

    return expf(-0.5f * mahalDist);
}

#endif // GAUSS_EVAL_H_INCLUDED
