#include "gaussianRasterizer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>
#include <cstddef>
#include <cstdint>  // for std::uintptr_t
#include "projection.h"
#include "ADRculling.h"

#ifndef CHECK_DC_INT                    // ② safe fall-backs
#   define CHECK_DC_INT(x)   (void)0
#   define CHECK_DC_FLOAT(x) (void)0
#   define CHECK_DC_BOOL(x)  (void)0
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>



#include "gaussianForwardPass.h"
#include "backward.h"
#include "pixelGaussiansFull.h"
#include "gaussianTiles.h"
#include "prepareSort.h"









#include <cuda.h>
#include <cuda_runtime.h>

// Phase 1: Per‐block exclusive scan (Blelloch)
__global__ void blelloch_block_scan(
       uint32_t* __restrict__  input,
    uint32_t*       __restrict__  output,
    uint32_t*       __restrict__  block_sums,
    int                             N
) {
    extern __shared__ uint32_t sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load input (or 0 if out‐of‐bounds)
    sdata[tid] = (gid < N) ? input[gid] : 0u;
    __syncthreads();

    // --- Upsweep (reduce) ---
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x)
            sdata[idx] += sdata[idx - offset];
        __syncthreads();
    }

    // Save total sum of this block
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = sdata[tid];
        sdata[tid] = 0;  // prepare for downsweep
    }
    __syncthreads();

    // --- Downsweep ---
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            uint32_t t        = sdata[idx - offset];
            sdata[idx - offset] = sdata[idx];
            sdata[idx]       += t;
        }
        __syncthreads();
    }

    // Write exclusive‐scan result back
    if (gid < N)
        output[gid] = sdata[tid];
}

// Phase 2: Add block sums to each element
__global__ void add_block_sums(
    uint32_t* __restrict__ data,
       uint32_t* __restrict__ block_sums,
    int                              N
) {
    int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < N) {
        // accumulate sum of all prior blocks
        uint32_t carry = 0;
        for (int b = 0; b < blockIdx.x; ++b)
            carry += block_sums[b];
        data[gid] += carry;
    }
}

// Host‐side launch
void prefix_sum_blelloch(
       uint32_t* d_input,
    uint32_t*       d_output,
    int                   N
) {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // allocate block sums
    uint32_t* d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(uint32_t));

    // 1) block‐level scan
    size_t sharedMem = threads * sizeof(uint32_t);
    blelloch_block_scan<<<blocks, threads, sharedMem>>>(d_input, d_output, d_block_sums, N);

    // 2) propagate sums across blocks
    add_block_sums<<<blocks, threads>>>(d_output, d_block_sums, N);

    cudaFree(d_block_sums);
}




// __global__ void computeAABBTiles(
//        int     numGaussians,
//        float2* means2D,          // [numGaussians]
//        int*    radss,            // [numGaussians]  (pixels)
//        int     blockW,           // 16
//        int     blockH,           // 16
//        int     tilesX,
//        int     tilesY,
//     int*          aabbTL,           // [numGaussians*2] out
//     int*          aabbBR)           // [numGaussians*2] out
// {
//        int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (gid >= numGaussians) return;

//        float2 c = means2D[gid];
//        int    r = radss[gid];

//     // pixel bounds (inclusive)
//        int px_min = static_cast<int>(floorf(c.x - r));
//        int py_min = static_cast<int>(floorf(c.y - r));
//        int px_max = static_cast<int>(ceilf (c.x + r));
//        int py_max = static_cast<int>(ceilf (c.y + r));

//     // convert to tile indices, clamp to grid
//        int tx_min = max(0,            px_min / blockW);
//        int ty_min = max(0,            py_min / blockH);
//        int tx_max = min(tilesX - 1,   px_max / blockW);
//        int ty_max = min(tilesY - 1,   py_max / blockH);

//     aabbTL[gid * 2 + 0] = tx_min;
//     aabbTL[gid * 2 + 1] = ty_min;
//     aabbBR[gid * 2 + 0] = tx_max;
//     aabbBR[gid * 2 + 1] = ty_max;
// }








int gaussianForwardPass(
    float*       opacityOP,                // Output buffer: per-pixel accumulated opacity (alpha)
     float  cameraFLenVert,           // Vertical focal length for projection
    std::function<char*(size_t)> imageBuffer, // Allocator for screen-space buffers
    int*         radss,                    // Optional: per-Gaussian radius buffer (computed if null)
     float* orientationss,            // Per-Gaussian quaternion orientations (float4)
    int    numGaussians,             // Total number of Gaussians to render
     float* scales,                   // Per-Gaussian scale (float3)
     int    M,                        // Number of SH bands
     float* positionOfCamera,         // World-space camera position
    std::function<char*(size_t)> BBuffer,  // Allocator for peparing arrays buffers
     float* matViewCam,               // View matrix (4x4)
    float*       ZOP,                      // Output buffer: per-pixel depth
     float* mattProjCam,              // Projection matrix (4x4)
    std::function<char*(size_t)> gaussianInfoMemSpace, // Allocator for geometry-stage scratch memory
    int          shNum,                        // Total SH coefficients (shNum = 3 × M for RGB)
     float* sphericalHarmonics,       // SH coefficients per Gaussian
     float* means3D,                  // Gaussian centers in world space
     float* gaussianAlphas,           // Opacity (alpha) per Gaussian
     float  cameraFLenHorz,           // Horizontal focal length for projection
    float*       colOP                     // Output buffer: final RGB color image
) {
// ============================================================================
// SECTION: stage 1
// ============================================================================
// This  stage is is responsible
// for transforming each 3D Gaussian from world space into a format suitable
// for rasterization (i.e., drawing on screen).
//
// This stage includes:
//   1. Projecting the 3D mean of each Gaussian to 2D screen space.
//   2. Estimating the 2D footprint (covariance matrix) of the ellipsoid.
//   3. Clipping and clamping values to the screen bounds.
//   4. Evaluating opacity and shading contributions (e.g., SH -> RGB).
//
// The memory carved below is temporary "scratch space" to hold these results
// before passing them on to the next stages ( rendering).
//
// We allocate all the memory in **one big aligned chunk** to:
//   - Improve memory coalescing and cache usage on GPU.
//   - Avoid repeated `cudaMalloc` overhead.
//   - Simplify memory management by using pointer arithmetic.
// ============================================================================

// Ensures 128-byte alignment for fast global memory access.
// GPUs are optimized for loads/stores that are aligned to 128 bytes.
// Misalignment leads to performance penalties due to additional transactions.

//quick explaination
// 1st buffer: 1000 floats
// align128(0) = 0 + 4000 = 4000
// 2nd buffer: 1000 booleans (3 per Gaussian)
// bytes = 4096 + 3000 = 7096
auto align128 = [](size_t x) { return (x + 127) & ~size_t(127); };

// We will accumulate the total required memory size in `bytes`.
size_t bytes = 0;

// Allocate space for per-Gaussian depth in camera space.
// Used to sort and cull Gaussians based on visibility.
bytes = align128(bytes) + numGaussians * sizeof(float);      

// Allocate space to store whether the projected 2D point was clamped
// to the view frustum bounds (for x, y, and z separately).
bytes = align128(bytes) + numGaussians * 3 * sizeof(bool);   

// Stores per-Gaussian radius (computed internally if not supplied).
// Required for determining screen-space footprint.
bytes = align128(bytes) + numGaussians * sizeof(int);        

// 2D screen-space coordinates after projection of the 3D Gaussian means.
bytes = align128(bytes) + numGaussians * sizeof(float2);     

// 2D covariance matrices in image space (elliptical Gaussian shape).
// Only 6 values per matrix are needed due to symmetry (3x3 reduced).
bytes = align128(bytes) + numGaussians * 6 * sizeof(float);  

// Stores combined alpha and conic parameters (opacity, shape).
// Used later for blending in the rendering pass.
bytes = align128(bytes) + numGaussians * sizeof(float4);     

// RGB colors of the Gaussians after evaluating spherical harmonics
// (shading model to encode/view-dependent color).
bytes = align128(bytes) + numGaussians * 3 * sizeof(float);  

// For each Gaussian, we store the number of **screen tiles** it overlaps.
// Tiles are rectangular screen partitions to accelerate localized rendering.
bytes = align128(bytes) + numGaussians * sizeof(uint32_t);   

// ============================================================================
// SECTION: stage2 
// ============================================================================
// 
//
// In peparing arrays, we assign each Gaussian to one or more *tiles* of the image.
//
// Why?
//   - To improve rendering efficiency by only rasterizing Gaussians that
//     affect a particular screen region.
//   - This reduces overdraw and avoids checking all Gaussians for every pixel.
//
// We perform an Accumlative summing on `intersectedTiles[]` to prepare
// for parallel compaction and sorting of Gaussians per tile.
//
// We ask CUB (CUDA UnBound) how much temporary memory is needed for this.
// ============================================================================
// STEP: Query how much temporary device memory is needed for an inclusive scan operation.

// We will soon perform a CUB::InclusiveSum (prefix sum) on an array of `numGaussians` elements.
// But before we can do that, we need to know how much temporary workspace (scratch space)
// CUB will need for this operation — it varies depending on input size and GPU architecture.

// Set this to 0; CUB will fill it with the required size (in bytes).
size_t scannedAmount = 0;

// Perform a "dry run" — no actual computation happens here.
// We pass nullptrs as input/output buffers, and CUB returns how much temporary memory is needed.
//
// Parameters:
// - nullptr: No temp buffer provided yet (we're just querying).
// - scannedAmount: CUB fills this with the number of bytes it *would need*.
// - nullptr: Dummy input pointer (we don't care about actual data yet).
// - nullptr: Dummy output pointer (same).
// - numGaussians: Number of elements we will eventually scan.
cub::DeviceScan::InclusiveSum(
    nullptr,
    scannedAmount,
    (uint32_t*)nullptr,
    (uint32_t*)nullptr,
    numGaussians
);

// After this line, `scannedAmount` will contain the number of bytes
// we need to allocate for CUB's internal use during the real scan.
//
// This value is later used to carve memory from the geometry buffer:
//     carve(scannedSppacee, scannedAmount);
// so that we can pass it to the *actual* scan call during peparing arrays prep.

// Add memory for the temporary scan workspace (needed for CUB).
bytes = align128(bytes) + scannedAmount;

// Add space for the output of the scan:
// Each entry will contain the "index in compacted buffer" for the Gaussian.
bytes = align128(bytes) + numGaussians * sizeof(uint32_t);

// Final small padding for safety in case next block needs alignment.
bytes += 128;

// ============================================================================
// ALLOCATE FINAL BUFFER
// ============================================================================
// At this point, we have calculated the full amount of memory we need.
// We now ask the caller to allocate this buffer via `gaussianInfoMemSpace()`.
// This can be backed by a pre-allocated CUDA memory pool or device malloc.
char* Gpointer = gaussianInfoMemSpace(bytes);
char* GChunk   = Gpointer;

// ============================================================================
// HELPER: Carve aligned blocks from geometry scratch buffer
// ============================================================================
// This lambda is used to extract (aligned) typed pointers out of `GChunk`.
// Each buffer is guaranteed to begin at a 128-byte aligned address.
auto carve = [&](auto*& out, size_t count) {
    uintptr_t a = ((uintptr_t)GChunk + 127) & ~uintptr_t(127);
    out = reinterpret_cast<decltype(out)>(a);
    GChunk = reinterpret_cast<char*>(out + count);
};

// ============================================================================
// CARVE OUT ALL GEOMETRY AND peparing arrays BUFFERS
// ============================================================================
float*     zDistances;             // Depth in camera space
bool*      clamped;                // Clamping flags (x, y, z)
int*       internalRs;             // Fallback radius values
float2*    means2D;                // Projected 2D coords
float*     cov3D;                  // Covariance matrices
float4*    alphaConicOfgauss;      // Opacity and conic encoding
float*     rgb;                    // Shaded colors
uint32_t*  intersectedTiles;       // Tile overlap flags
char*      scannedSppacee;         // Scratch memory for CUB scan
uint32_t*  locationsOfPointsWRT;  // Gaussian index mapping after scan

// Use `carve()` to assign each pointer and move forward in the chunk
carve(zDistances,         numGaussians);
carve(clamped,            numGaussians * 3);
carve(internalRs,         numGaussians);
carve(means2D,            numGaussians);
carve(cov3D,              numGaussians * 6);
carve(alphaConicOfgauss,  numGaussians);
carve(rgb,                numGaussians * 3);
carve(intersectedTiles,   numGaussians);
carve(scannedSppacee,     scannedAmount);
carve(locationsOfPointsWRT, numGaussians);

// At this point, all intermediate buffers are ready to be filled.
// Geometry stage kernels will write into these before peparing arrays and rendering.
// Memory layout is linear, tightly packed, and fully GPU-resident.


// ======================================================================================
// SECTION: SETUP DEFAULTS AND TILE GRID
// ======================================================================================

// If the caller did not provide a `radss` buffer (which holds per-Gaussian radii),
// we fall back to using `internalRs`, a temporary buffer we allocated earlier.
// This makes the function robust to optional inputs.
if (!radss) radss = internalRs;

// Clear the intersectedTiles buffer — this will be filled during projection
// with tile overlap counts or bitmasks, depending on the later processing.
cudaMemset(intersectedTiles, 0, numGaussians * sizeof(uint32_t));

// Define the screen tiling grid.
// We divide the screen into 16×16 pixel tiles over an 800×800 image.
// This creates a 50x50 tile grid that enables localized rendering and memory access.
dim3 gridOfTiles((800 + 15)/16, (800 + 15)/16, 1);
dim3 block(16, 16, 1); // Typical CUDA block size for 2D kernels

// ======================================================================================
// SECTION: ALLOCATE IMAGE-SPACE BUFFERS
// ======================================================================================

// Buffers here are screen-sized (640k pixels) and represent per-pixel data used
// for rendering. We align allocations to 128 bytes for coalesced access.

// Helper function for 128-byte alignment
auto alignUp128 = [](size_t x, size_t A){ return (x + A - 1) & ~(A - 1); };

// Estimate required memory:
// - 1 uint32_t per pixel to count how many Gaussians contributed
// - 1 uint2 per pixel to store range (start, end index) in compacted Gaussian array
size_t img_bytes = alignUp128(640000 * sizeof(uint32_t), 128)
                 + 640000 * sizeof(uint2)
                 + 128; // Safety padding

// Allocate all image-space memory in one go
char* img_ptr = imageBuffer(img_bytes);
char* ichunk = img_ptr;

uint32_t* NContributedGaussians;
{
    uintptr_t a = alignUp128((uintptr_t)ichunk, 128);
    NContributedGaussians = reinterpret_cast<uint32_t*>(a);
    ichunk = reinterpret_cast<char*>(NContributedGaussians + 640000);
}

uint2* tileSliceStartEnd;
{
    uintptr_t a = alignUp128((uintptr_t)ichunk, 128);
    tileSliceStartEnd = reinterpret_cast<uint2*>(a);
}

// Zero out the pixel-contribution counters and tile range arrays
cudaMemset(NContributedGaussians, 0, 640000 * sizeof(uint32_t));
cudaMemset(tileSliceStartEnd, 0, gridOfTiles.x * gridOfTiles.y * sizeof(uint2));

// ======================================================================================
// SECTION: TEMP DEVICE BUFFERS (PROJECTION INTERMEDIATES)
// ======================================================================================
// These are intermediate CUDA device buffers used during projection:
// - d_cov2D: per-Gaussian 2D covariance matrix (ellipse shape)
// - d_dets : determinant of 2D Gaussian (for culling thresholds)
// - d_orig : original 3D center for debugging or validation
float3* d_cov2D; float* d_dets; float3* d_orig;
cudaMalloc(&d_cov2D, numGaussians * sizeof(float3));
cudaMalloc(&d_dets , numGaussians * sizeof(float ));
cudaMalloc(&d_orig , numGaussians * sizeof(float3));

// ======================================================================================
// STEP 1: PROJECT 3D GAUSSIANS TO 2D IMAGE PLANE
// ======================================================================================
// This transforms the 3D mean of each Gaussian into 2D screen space.
// It also computes:
// - Clamping of points to camera frustum bounds
// - 2D elliptical footprint from orientation and scale
// - Alpha-conic product (opacity and shape for later blending)
PROJECTION::ProjectG(
    numGaussians, means3D,
    (glm::vec3*)scales,
    (glm::vec4*)orientationss,
    matViewCam, cameraFLenHorz, cameraFLenVert,
    cov3D, radss, intersectedTiles,   // Outputs: covariances, radii, and tiles
    d_cov2D, d_dets, d_orig,          // Temporary buffers
    gaussianAlphas, alphaConicOfgauss, // Alpha + conic product
    means2D, mattProjCam, zDistances   // 2D projection result
);

// ======================================================================================
// STEP 2: CULL GAUSSIANS OUTSIDE THE SCREEN
// ======================================================================================
// This function removes Gaussians that do not significantly contribute to any screen tile.
// It uses the determinant of the covariance matrix (d_dets) to determine whether
// the projected ellipse is large enough to matter.
// Also updates intersectedTiles[] accordingly.
ADR::tileCulling(
    numGaussians, gaussianAlphas,
    d_cov2D, d_dets, means2D,
    gridOfTiles, radss, intersectedTiles
);

// ======================================================================================
// STEP 3: COMPUTE SHADING (RGB) FROM SPHERICAL HARMONICS
// ======================================================================================
// Converts spherical harmonics coefficients to RGB color based on view direction.
// The view direction is computed using the camera position and each Gaussian center.
// Outputs per-Gaussian RGB color to the `rgb` buffer.
FORWARD::calcColour(
    numGaussians, shNum, M, means3D,
    (glm::vec3*)positionOfCamera, sphericalHarmonics,
    clamped, rgb
);

// ======================================================================================
// STEP 4: SCAN — PREPARE FOR peparing arrays BY COMPUTING INDEXES
// ======================================================================================
// This inclusive scan computes the exclusive prefix sum of how many tiles each Gaussian touches.
// The result (`locationsOfPointsWRT`) tells us where each Gaussian should be written
// in a compacted array organized by screen tiles.
cub::DeviceScan::InclusiveSum(
    scannedSppacee,
    scannedAmount,
    intersectedTiles,
    locationsOfPointsWRT,
    numGaussians
);

// ======================================================================================
// STEP 5: ASSIGN GAUSSIANS TO TILES
// ======================================================================================
// Based on the tile coverage and scan results, we now compact Gaussians into tile bins.
// This enables tile-based rendering where each thread block processes only its local Gaussians.
int NGaussiansRendered;
auto carvedBin = prepareAndSort(
    BBuffer,
    numGaussians,
    locationsOfPointsWRT,
    means2D,
    zDistances,
    radss,
    gridOfTiles,
    NGaussiansRendered
);

// ======================================================================================
// STEP 6: UPDATE TILE slices in mainArr
// ======================================================================================
// This kernel computes the start and end index in the compacted Gaussian buffer
// for each tile. It fills the `tileSliceStartEnd` array which is indexed by tile ID.
if (NGaussiansRendered > 0) 
{
    gaussianTiles<<<(NGaussiansRendered + 255) / 256, 256>>>(
        NGaussiansRendered,
        carvedBin.keysArray, // Sorted tile ID for each Gaussian
        tileSliceStartEnd               // Output: [start, end) index range per tile
    );
}

// ======================================================================================
// STEP 7: RENDERING — FINAL DRAWING OF GAUSSIANS TO IMAGE
// ======================================================================================
// This kernel does the heavy lifting: it draws each Gaussian into its screen tile
// using additive alpha blending (weighted by opacity, conic, and SH-derived RGB).
// The result is written to:
// - colOP: RGB output image
// - opacityOP: accumulated alpha
// - ZOP: depth buffer (for z-compositing)
   float* pointerForFeatures = rgb;
pixelGaussiansFull(
    gridOfTiles, block,
    tileSliceStartEnd,
    carvedBin.MainGaussianIdArray,
    means2D,
    pointerForFeatures,
    zDistances,
    alphaConicOfgauss,
    opacityOP,
    NContributedGaussians,
    colOP,
    ZOP,
    d_cov2D
);

// ======================================================================================
// STEP 8: CLEANUP
// ======================================================================================
// Free the temporary buffers allocated on the device for projection intermediates
cudaFree(d_cov2D);
cudaFree(d_dets);
cudaFree(d_orig);

// Return the number of Gaussians that actually contributed to rendering
return NGaussiansRendered;

}





void backward(
  const int   numGaussians,      int shNum,  int M,  int R,
  const float* background,
  const int   width,  int height,
  const float* means3D,
  const float* sphericalHarmonics,
  const float* colors_precomp,
  const float* alphas,
  const float* scales,
  const float  scale_modifier,
  const float* orientationss,
  const float* cov3D_precomp,
  const float* matViewCam,
  const float* mattProjCam,
  const float* campos,
  const float  camHorizLengthTan, float camVertLengthTan,
  const int*   radss,
  char*        geom_buffer,
  char*        binning_buffer,
  char*        img_buffer,
  const float* dL_dpix,
  const float* dL_dpix_depth,
  const float* dL_dalphas,
  float*       dL_dmean2D,
  float*       dL_dconic,
  float*       dL_dopacity,
  float*       dL_dcolor,
  float*       dL_ddepth,
  float*       dL_dmean3D,
  float*       dL_dcov3D,
  float*       dL_dsh,
  float*       dL_dscale,
  float*       dL_drot,
  bool         debug)
{
  // ------------------------------------------------
  // 1) Hand-carve out the geometry buffers from geom_buffer
  // ------------------------------------------------
  auto align128 = [](size_t x){ return (x + 127) & ~static_cast<size_t>(127); };
  char* chunk = geom_buffer;

  // carve helper
  auto carve = [&](auto*& ptr, size_t count){
    uintptr_t a = align128(reinterpret_cast<uintptr_t>(chunk));
    ptr   = reinterpret_cast<decltype(ptr)>(a);
    chunk = reinterpret_cast<char*>(ptr + count);
  };

  float*    zDistances;
  carve(zDistances,             numGaussians);

  bool*     clamped;
  carve(clamped,            numGaussians * 3);

  int*      internalRs;
  carve(internalRs,     numGaussians);

  float2*   means2D;
  carve(means2D,            numGaussians);

  float*    cov3D;
  carve(cov3D,              numGaussians * 6);

  float4*   alphaConicOfgauss;
  carve(alphaConicOfgauss,      numGaussians);

  float*    rgb;
  carve(rgb,                numGaussians * 3);

  uint32_t* intersectedTiles;
  carve(intersectedTiles,      numGaussians);

  // figure out CUB scan temp-size
  size_t scannedAmount = 0;
  cub::DeviceScan::InclusiveSum(
    /*temporaryDeviceStorgae=*/    nullptr,
    /*temp_storage_bytes=*/ scannedAmount,
    /*d_in=*/              (uint32_t*)nullptr,
    /*d_out=*/             (uint32_t*)nullptr,
    /*num_items=*/         numGaussians
  );
  char*     scannedSppacee;
  carve(scannedSppacee,     scannedAmount);

  uint32_t* locationsOfPointsWRT;
  carve(locationsOfPointsWRT,      numGaussians);

  // if the user didn’t pass in radss, use our carved buffer
  if (radss == nullptr) radss = internalRs;

  // ------------------------------------------------
  // 2) Common camera + tile-grid setup
  // ------------------------------------------------
  const float cameraFLenVert = height / (2.0f * camVertLengthTan);
  const float cameraFLenHorz = width  / (2.0f * camHorizLengthTan);

  const dim3 gridOfTiles(
    (width  + 16 - 1) / 16,
    (height + 16 - 1) / 16,
    1
  );
  const dim3 block(16, 16, 1);

  const float* color_ptr = colors_precomp ? colors_precomp : rgb;
  const float* depth_ptr = zDistances;

  // ------------------------------------------------
  // 3) Carve out binning_buffer → MainGaussianIdArray[R]
  // ------------------------------------------------
  auto alignUp128 = align128;
  uintptr_t a_bin = alignUp128(reinterpret_cast<uintptr_t>(binning_buffer));
  uint32_t* MainGaussianIdArray = reinterpret_cast<uint32_t*>(a_bin);

  // ------------------------------------------------
  // 4) Carve out img_buffer → NContributedGaussians[640000], tileSliceStartEnd[640000]
  // ------------------------------------------------
  char* ichunk = img_buffer;
  size_t N = width * height;

  uintptr_t a_img = alignUp128(reinterpret_cast<uintptr_t>(ichunk));
  uint32_t* NContributedGaussians = reinterpret_cast<uint32_t*>(a_img);
  ichunk = reinterpret_cast<char*>(NContributedGaussians + N);

  a_img = alignUp128(reinterpret_cast<uintptr_t>(ichunk));
  uint2* tileSliceStartEnd = reinterpret_cast<uint2*>(a_img);

  // ------------------------------------------------
  // 5) BACKWARD::render
  // ------------------------------------------------
  BACKWARD::render(
    gridOfTiles,
    block,
    tileSliceStartEnd,
    MainGaussianIdArray,
    width, height,
    background,
    means2D,
    alphaConicOfgauss,
    color_ptr,
    depth_ptr,
    alphas,
    NContributedGaussians,
    dL_dpix,
    dL_dpix_depth,
    dL_dalphas,
    (float3*)dL_dmean2D,
    (float4*)dL_dconic,
    dL_dopacity,
    dL_dcolor,
    dL_ddepth
  );

  // ------------------------------------------------
  // 6) BACKWARD::preprocess
  // ------------------------------------------------
  const float* cov3D_ptr = cov3D_precomp ? cov3D_precomp : cov3D;
  BACKWARD::preprocess(
    numGaussians, shNum, M,
    (float3*)means3D,
    radss,
    sphericalHarmonics,
    clamped,
    (glm::vec3*)scales,
    (glm::vec4*)orientationss,
    scale_modifier,
    cov3D_ptr,
    matViewCam,
    mattProjCam,
    cameraFLenHorz, cameraFLenVert,
    camHorizLengthTan, camVertLengthTan,
    (glm::vec3*)campos,
    (float3*)dL_dmean2D,
    dL_dconic,
    (glm::vec3*)dL_dmean3D,
    dL_dcolor,
    dL_ddepth,
    dL_dcov3D,
    dL_dsh,
    (glm::vec3*)dL_dscale,
    (glm::vec4*)dL_drot
  );
}


