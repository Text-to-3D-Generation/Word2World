#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>
#include <cstddef>
#include <cstdint>  
#include "projection.h"
#include "ADRculling.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using highres_clock = std::chrono::high_resolution_clock;
#include <cstdio>
#include <cstdlib>
#include <chrono>
using namespace cooperative_groups;
#include "gaussianForwardPass.h"
#include "backward.h"
#include <cuda.h>
#include <cuda_runtime.h>


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


