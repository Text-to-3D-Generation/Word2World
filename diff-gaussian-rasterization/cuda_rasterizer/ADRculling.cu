#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "ADRculling.h"

/**
 * CUDA Kernel: tileCullingCUDA
 * --------------------------------------
 * This kernel computes the screen-space tile coverage for each 2D projected Gaussian.
 * It estimates a bounding radius for each Gaussian based on its shape and opacity,
 * then determines which 16×16 screen tiles the Gaussian intersects.
 * 
 * Inputs:
 *  - numGaussians: total number of Gaussians to process
 *  - gaussianAlphas: per-Gaussian opacity values in [0, 1]
 *  - covOP: per-Gaussian 2D covariance matrix stored as float3 (xx, xy, yy)
 *  - detOP: determinant of the 2D covariance matrix (used to avoid recalculating)
 *  - muu2dPixelCoord: projected 2D screen coordinates of Gaussian centers (in pixels)
 *  - grid: dimensions of screen in tiles (i.e., number of 16×16 tile blocks in x/y)
 * 
 * Outputs:
 *  - radss: output buffer of per-Gaussian radius in tile units (used in peparing arrays)
 *  - intersectedTiles: output buffer of tile count per Gaussian (used in scan/buffer packing)
 */


__global__ void tileCullingCUDA(
    int               numGaussians,
      float*      gaussianAlphas,
      float3*     covOP,
      float*      detOP,
      float2*     muu2dPixelCoord,
    dim3              grid,           // number of 16×16 tiles in x/y
          int*         radss,
    uint32_t*         intersectedTiles
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numGaussians) return;

    // === [1] Estimate the Gaussian's 2D footprint ===
    // Recover 2×2 covariance matrix (symmetric) from packed float3:
    // cov = [ [xx, xy], [xy, yy] ]
    float3 covarianceMat = covOP[idx];
    float determinant = detOP[idx];

    // Compute eigenvalues λ₁ and λ₂ of the covariance matrix
    float mid = 0.5f * (covarianceMat.x + covarianceMat.z);  // average of diagonal
    float Y1 = mid + sqrtf(fmaxf(0.1f, mid * mid - determinant));
    float Y2 = mid - sqrtf(fmaxf(0.1f, mid * mid - determinant));
    float YMAX = fmaxf(Y1, Y2); // dominant spread direction

    // === [2] Compute original radius (3σ rule) ===
    // The 3σ bound covers ~99% of the Gaussian's influence
    float allProbRadius = 3.f * sqrtf(YMAX);

    // === [3] Early culling: discard completely invisible Gaussians ===
    if (gaussianAlphas[idx] <= 0.0039f) {
        radss[idx] = 0;
        intersectedTiles[idx] = 0;
        return;
    }

    // === [4] Compute adaptive radius based on opacity (Eq. 7) ===
    // This uses the thresholded Mahalanobis distance formulation:
    // adaptiveRadius = sqrt(2 · λ_max · ln(σᵢ / α_low))
    float lnRatio = logf(gaussianAlphas[idx] * 255.0f);  // α_low = 1/255
    float adaptiveRadius = sqrtf(2.f * YMAX * lnRatio);

    // Final effective radius: smaller of 3σ and adaptive bound
    float minRad = fminf(adaptiveRadius, allProbRadius);

    // Save the computed radius (rounded up)
    radss[idx] = ceilf(minRad);

    // === [5] Compute bounding box in tile space ===
    // Expand a square around the Gaussian center by radius and convert to tile units
    int2 tmin = make_int2(
        int((muu2dPixelCoord[idx].x - minRad) / 16.0f),
        int((muu2dPixelCoord[idx].y - minRad) / 16.0f)
    );
    int2 tmax = make_int2(
        int((muu2dPixelCoord[idx].x + minRad + 15) / 16.0f),
        int((muu2dPixelCoord[idx].y + minRad + 15) / 16.0f)
    );

    // Clamp to valid tile grid (screen bounds)
    uint2 rect_min = make_uint2(
        max(0, tmin.x), max(0, tmin.y)
    );
    uint2 rect_max = make_uint2(
        min(grid.x, tmax.x), min(grid.y, tmax.y)
    );

    // === [6] Compute tile coverage area ===
    // If a Gaussian touches zero tiles, it can be culled
    intersectedTiles[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
}


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
            means2D,
            grid,
            radss,
            intersectedTiles
        );
        
    }

} 
