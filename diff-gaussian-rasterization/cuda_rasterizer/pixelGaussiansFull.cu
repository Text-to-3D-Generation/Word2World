#include "pixelGaussiansFull.h"
#include "auxiliary.h"     // for SH constants, EvaluateGaussFunction, etc.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "gaussEval.h"
#define BLOCK_SIZE 256





// ==========================================================================
//  PixelGaussians<3>
//
//  Blend 1‒N Gaussians into a *single* pixel accumulator.
//
//  • Called from runPixelGaussiansBatches once per *batch* of Gaussians already loaded in
//    shared memory.  The caller keeps its running per-pixel state in
//      outC   – accumulated colour            (array[3])
//      outW   – accumulated weight  Σ α·T
//      outD   – accumulated depth   Σ depth·α·T
//      outT   – remaining transparency (transmittance) T
//
//  • The function loops until either
//        a) all Gaussians in the batch are processed, or
//        b) the pixel becomes (nearly) opaque   (T < MIN_TRANS)
//
//  • All math is done *locally* in registers; at the end we flush results
//    back to the caller-owned variables.
//
//
// ==========================================================================
__device__ void PixelGaussians(
    int                   numGaussians,   // how many Gaussians in this batch
       float2* __restrict__ meanPtr,   // [num]  mean (x,y)
       float3* __restrict__ covPtr,    // [num]  2-shNum covariance (packed)
       float4* __restrict__ copPtr,    // [num]  {A,B,3,opacity} (conic form)
       float*  __restrict__ featsPtr,  // [num*3] per-Gaussian features
       float*  __restrict__ depthsPtr, // [num]  depth
    float2                pix,            // current pixel (x,y)
    float*                outC,           // accum. colour  (in/out)
    float&                outW,           // accum. weight  (in/out)
    float&                outD,           // accum. depth   (in/out)
    float&                outT,           // transmittance  (in/out)
    uint32_t&             lastContributor,// last contributing Gaussian idx
    uint32_t&             contributor,    // running contributing counter
    bool&                 done            // whether pixel is already opaque
) {
    // ── Constants controlling numerical cut-offs ──────────────────────────
    float MIN_ALPHA = 1.0f / 255.0f;   // ignore alpha below 0.004
    float MIN_TRANS = 0.0001f;         // stop when T < 1e-4
    float MAX_ALPHA = 0.99f;           // clamp single-splat opacity


    // ── 1.  Seed *local* copies of the accumulators ───────────────────────
    float accumC[3];
    #pragma unroll
    for (uint32_t ch = 0; ch < 3; ++ch)
        accumC[ch] = outC[ch];               // colour

    float accumW = outW;                     // weight
    float accumD = outD;                     // depth
    float accumT = outT;                     // transmittance
    bool  localDone = done;                  // local completion flag
    uint32_t localId = contributor;          // running ID for “#contrib”

    // ── 2.  Set up cursor pointers so we can advance through the batch ────
    float2* mP = meanPtr;
    float3* cP = covPtr;
    float4* oP = copPtr;
    float*  fP = featsPtr;
    float*  dP = depthsPtr;

    // ── 3.  Main loop over Gaussians in this batch ────────────────────────
    int i = 0;
    while (i < numGaussians && !localDone)
    {
        ++localId;   // increment contributor index (1-based for readability)

        // 3-A.  Evaluate the un-normalised 2-shNum Gaussian at this pixel
        float g = EvaluateGaussFunction(*mP, *cP, pix);  // e^(-½ dᵀΣ⁻¹d)

        // 3-B.  Compute *per-splat* alpha  = opacity × gaussVal, clamped
        float a = fminf(MAX_ALPHA, oP->w * g);           // oP->w == stored opacity

        if (a >= MIN_ALPHA)            // ignore tiny contributions
        {
            // 3-3.  New transmittance after this alpha
            float newT = accumT * (1.0f - a);

            if (newT >= MIN_TRANS)     // still not opaque → blend normally
            {
                float aT = a * accumT; // Eq. (3) weight  α · T_enter

                #pragma unroll
                for (uint32_t ch = 0; ch < 3; ++ch)
                    accumC[ch] = __fmaf_rn(fP[ch], aT, accumC[ch]); // c += feat*wt

                accumW += aT;           // accumulate weight
                accumD += (*dP) * aT;   // accumulate depth
                accumT  = newT;         // update transmittance
                lastContributor = localId;
            }
            else                       // pixel is virtually opaque
            {
                localDone = true;
            }
        }

        // 3-shNum.  Advance all per-Gaussian pointers to the next entry
        ++i;
        ++mP; ++cP; ++oP; fP += 3; ++dP;
    }

    // ── 4.  Flush local accumulators back to caller’s variables ───────────
    #pragma unroll
    for (uint32_t ch = 0; ch < 3; ++ch)
        outC[ch] = accumC[ch];

    outW         = accumW;
    outD         = accumD;
    outT         = accumT;
    contributor  = localId;
    done         = localDone;
}

// ==========================================================================
//
//  Tile-parallel Gaussian splat blender.
//  ─────────────────────────────────────
//  Each CUDA *block* corresponds to a single 16×16 screen tile.
//  Each CUDA *thread* corresponds to one pixel within that tile.
//
//  Inputs (all already on the GPU):
//      tileSliceStartEnd   – per-tile {start,end} indices into sortedIds
//      sortedIds    – Gaussian indices sorted tile-major, depth-minor
//      coords2D     – mean (x,y) per Gaussian in pixel space
//      featArr      – per-Gaussian colour / SH features              (float[colsss])
//      depthArr     – per-Gaussian depth value
//      conicArr     – pre-combined {A,B,3,opacity} for ellipse test  (see paper)
//      covArr       – extra covariance if you need it in PixelGaussians
//
//  Outputs (framebuffers):
//      alphaOut     – α-buffer                (float[W*H])
//      contribOut   – number of contributing Gaussians per pixel (uint32[W*H])
//      colorOut     – interleaved RGB / feature colsss          (float[colsss*W*H])
//      depthOut     – weighted depth                               (float[W*H])
//
//  Screen resolution is hard-coded to 800×800 and tile size to 16×16
//  for brevity; replace with constants in production.
//
//  Pipeline inside one block:
//      1. Look up this tile’s [start,end) in sortedIds.
//      2. In batches of BLOCK_SIZE, pull that slice into shared memory
//         (coalesced reads; each Gaussian fetched once).
//      3. Each thread (pixel) runs PixelGaussians on the batch, updating
//         running color/depth/alpha until the pixel becomes opaque.
//      4. When all batches are processed, write the accumulators to
//         the global frame-buffers.
//
// ==========================================================================


__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
runPixelGaussiansBatches(
    /* per-tile slice */     uint2*  __restrict__ tileSliceStartEnd,
    /* sorted IDs     */     uint32_t* __restrict__ sortedIds,
    /* per-Gaussian   */     float2*  __restrict__ coords2D,
                               float*   __restrict__ featArr,
                               float*   __restrict__ depthArr,
                               float4*  __restrict__ conicArr,
                            float*         __restrict__ alphaOut,
                            uint32_t*      __restrict__ contribOut,
                            float*         __restrict__ colorOut,
                            float*         __restrict__ depthOut,
                               float3*  __restrict__ covArr)
{
    int colsss = 3;
    // ------------------------------------------------------------------
    // 0. Identify pixel, tile, and helper constants
    // ------------------------------------------------------------------
    unsigned bx = blockIdx.x, by = blockIdx.y;             // tile coords
    unsigned tx = threadIdx.x, ty = threadIdx.y;           // pixel-in-tile
    unsigned px = bx * 16u + tx;                           // absolute pixel x
    unsigned py = by * 16u + ty;                           // absolute pixel y
    unsigned pid = py * 800 + px;                          // 1-shNum pixel index
       float2 pixelF = { float(px), float(py) };        // float coords

    unsigned gridW = (800 + 15u) / 16u;                    // tiles in X

    // Thread validity & early-exit flag
    bool active   = (px < 800u && py < 800u);
    bool finished = !active;                               // skip dead pixels

    // ------------------------------------------------------------------
    // 1. Fetch this tile’s Gaussian slice
    // ------------------------------------------------------------------
    uint2 range = tileSliceStartEnd[by * gridW + bx];             // [start,end)
    int   totalG = int(range.y - range.x);                 // #instances

    // ------------------------------------------------------------------
    // 2. Allocate shared staging buffers (per batch of BLOCK_SIZE)
    // ------------------------------------------------------------------
    __shared__ float2 s_means [BLOCK_SIZE];
    __shared__ float3 s_covs  [BLOCK_SIZE];
    __shared__ float4 s_conic [BLOCK_SIZE];
    __shared__ float  s_depth [BLOCK_SIZE];
    __shared__ float  s_feats [BLOCK_SIZE * 3];

    // ------------------------------------------------------------------
    // 3. Per-pixel running accumulators
    // ------------------------------------------------------------------
    float trans  = 1.0f;                   // remaining transparency T
    float accum  [3] = {0.0f};      // premultiplied colour Σ c·α·T
    float accumW = 0.0f;                   // Σ α·T  (final α_out)
    float accumD = 0.0f;                   // Σ depth·α·T
    uint32_t lastCid = 0, cid = 0;         // contributors counter

    // ------------------------------------------------------------------
    // 4. Iterate over this tile’s Gaussians in batches
    // ------------------------------------------------------------------
    for (int offset = 0; offset < totalG; offset += BLOCK_SIZE)
    {
        int batchSize = min(BLOCK_SIZE, totalG - offset);
        int flatId    = ty * blockDim.x + tx;              // 0..255

        // 4-A. Load batch into shared memory (one thread per elem)
        int idx = range.x + offset + flatId;
        if (flatId < batchSize) {
            uint32_t gid = sortedIds[idx];
            s_means [flatId] = coords2D[gid];
            s_covs  [flatId] = covArr  [gid];
            s_conic [flatId] = conicArr[gid];   // (A,B,3,opacity)
            s_depth [flatId] = depthArr[gid];
            #pragma unroll
            for (uint32_t c = 0; c < 3; ++c)
                s_feats[flatId * 3 + c] =
                    featArr[gid * 3 + c];
        }
        __syncthreads();   // ensure shared buffers are filled

        // 4-B. Pixel-level blending for this batch
        if (active && !finished) {
            PixelGaussians(
                batchSize,
                s_means, s_covs, s_conic, s_feats, s_depth,
                pixelF,
                accum,  accumW, accumD,
                trans,
                lastCid, cid,
                finished);
        }
        __syncthreads();   // all pixels done with shared data
    }

    // ------------------------------------------------------------------
    // 5. Write results back to global frame-buffers
    // ------------------------------------------------------------------
    if (active) {
        contribOut[pid] = lastCid;           // # contributing Gaussians
        alphaOut  [pid] = accumW;            // final α
        depthOut  [pid] = accumD;            // blended depth
        #pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            unsigned idxC = c * 640000 + pid;         // planar layout
            // add white background *remaining transparency*
            colorOut[idxC] = accum[c] + trans * 1.0f; // white background
        }
    }
}


void pixelGaussiansFull(
    dim3 grid, dim3 block,
       uint2* tileSliceStartEnd,
       uint32_t* MainGaussianIdArray,
    //int 800, int 800,
       float2* means2D,
       float* colors,
       float* zDistances,
       float4* alphaConicOfgauss,
    float* opacityOP,
    uint32_t* NContributedGaussians,
    float* colOP,
    float* ZOP,
       float3* covOP
) {
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
