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
#include "pixelGaussiansFull.h"
#include "gaussianTiles.h"
#include "prepareSort.h"
#include <cuda.h>
#include <cuda_runtime.h>
// void kogge_stone_inclusive_scan_ultra_warp_optimized(uint32_t* input_array, uint32_t* output_array, int num_of_elements) {
//     int block_size = 1024;
//     int shared_memory_size = block_size * sizeof(uint32_t);
//     int num_blocks = (num_of_elements + block_size - 1) / block_size;
    
//     int device;
//     cudaGetDevice(&device);
    
//     cudaDeviceProp deviceProp;
//     cudaGetDeviceProperties(&deviceProp, device);
    
//     if (!deviceProp.cooperativeLaunch) {
//         printf("Error: Device does not support cooperative kernel launch\n");
//         return;
//     }
    
//     int max_blocks_per_sm;
//     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kogge_stone_inclusive_scan_ultra_warp_optimized_kernel, block_size, shared_memory_size);
    
//     int max_blocks = max_blocks_per_sm * deviceProp.multiProcessorCount;
    
//     if (num_blocks > max_blocks) {
//         printf("Error: Requested %d blocks, but maximum cooperative blocks is %d\n", num_blocks, max_blocks);
//         return;
//     }
    
//     uint32_t* block_sums;
//     cudaMalloc(&block_sums, num_blocks * sizeof(uint32_t));
    
//     void* args[] = {&input_array, &output_array, &block_sums, &num_of_elements};
    
//     dim3 grid_size(num_blocks);
//     dim3 block_size_dim(block_size);
    
//     cudaError_t result = cudaLaunchCooperativeKernel((void*)kogge_stone_inclusive_scan_ultra_warp_optimized_kernel, grid_size, block_size_dim, args, shared_memory_size);
    
//     if (result != cudaSuccess) {
//         printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(result));
//     }
    
//     cudaDeviceSynchronize();
//     cudaFree(block_sums);
// }







__global__ void add_block_sums_to_inclusive_output_kernel(uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements || blockIdx.x == 0) return;
    output_array[thread_id] += block_sums[blockIdx.x - 1];
}
__global__ void kogge_stone_inclusive_scan_shared_memory_kernel(uint32_t* input_array, uint32_t* output_array, uint32_t* block_sums, int num_of_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_of_elements) return;
    extern __shared__ uint32_t shared_output_array[];
    shared_output_array[threadIdx.x] = input_array[thread_id];
    __syncthreads();
    uint32_t temp_value = 0;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride)
            temp_value = shared_output_array[threadIdx.x - stride];
        __syncthreads(); 
        if (threadIdx.x >= stride)
            shared_output_array[threadIdx.x] += temp_value;
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = shared_output_array[threadIdx.x];

    output_array[thread_id] = shared_output_array[threadIdx.x];
}
void kogge_stone_inclusive_scan_shared_memory(uint32_t* input_array, uint32_t* output_array, int num_of_elements)
 {
    int block_size = 128;
    int shared_memory_size = block_size * sizeof(uint32_t);
    int num_blocks = (num_of_elements + block_size - 1) / block_size;
    uint32_t* block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(uint32_t));
    kogge_stone_inclusive_scan_shared_memory_kernel<<<num_blocks, block_size, shared_memory_size>>>(input_array, output_array, block_sums, num_of_elements);
    if (num_blocks > 1) {
        kogge_stone_inclusive_scan_shared_memory(block_sums, block_sums, num_blocks);
        add_block_sums_to_inclusive_output_kernel<<<num_blocks, block_size>>>(output_array, block_sums, num_of_elements);
    }
    cudaDeviceSynchronize();
    cudaFree(block_sums);
}
int gaussianForwardPass(
    float*       opacityOP,
    std::function<char*(size_t)> finalOutt,
    int*         radss,
     float* orientationss,
    int    numGaussians,
     float* scales,
     int    M,
     float* positionOfCamera,
    std::function<char*(size_t)> BBuffer,
     float* matViewCam,
    float*       ZOP,
     float* mattProjCam,
    std::function<char*(size_t)> gaussianInfoMemSpace,
    int          shNum,
     float* sphericalHarmonics,
     float* means3D,
     float* gaussianAlphas,
     float  cameraFLenHorz,          
    float*       colOP                     
) 
{
//quick explaination
// 1st buffer: 1000 floats
// align128(0) = 0 + 4000 = 4000
// 2nd buffer: 1000 booleans (3 per Gaussian)
// bytes = 4096 + 3000 = 7096
auto align128 = [](size_t x) { return (x + 127) & ~size_t(127); };
size_t bytes = 0;
bytes = align128(bytes) + numGaussians * sizeof(float);      
bytes = align128(bytes) + numGaussians * 3 * sizeof(bool);   
bytes = align128(bytes) + numGaussians * sizeof(int);        
bytes = align128(bytes) + numGaussians * sizeof(float2);     
bytes = align128(bytes) + numGaussians * 6 * sizeof(float);  
bytes = align128(bytes) + numGaussians * sizeof(float4);     
bytes = align128(bytes) + numGaussians * 3 * sizeof(float);  
bytes = align128(bytes) + numGaussians * sizeof(uint32_t);   
size_t scannedAmount = 0;
bytes = align128(bytes) + scannedAmount;
bytes = align128(bytes) + numGaussians * sizeof(uint32_t);
bytes += 128;
char* Gpointer = gaussianInfoMemSpace(bytes);
char* GChunk   = Gpointer;
auto carve = [&](auto*& out, size_t count) 
{
    uintptr_t a = ((uintptr_t)GChunk + 127) & ~uintptr_t(127);
    out = reinterpret_cast<decltype(out)>(a);
    GChunk = reinterpret_cast<char*>(out + count);
};
float*     zDistances;         
bool*      clamped;             
int*       internalRs;           
float2*    means2D;               
float*     cov3D;              
float4*    alphaConicOfgauss;      
float*     rgb;                   
uint32_t*  intersectedTiles;      
char*      scannedSppacee;         
uint32_t*  locationsOfPointsWRT; 
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
cudaMemset(intersectedTiles, 0, numGaussians * sizeof(uint32_t));
dim3 block(16, 16, 1); // Typical CUDA block size for 2D kernels
auto alignUp128 = [](size_t x, size_t A){ return (x + A - 1) & ~(A - 1); };
size_t img_bytes = alignUp128(640000 * sizeof(uint32_t), 128)
                 + 640000 * sizeof(uint2)
                 + 128; 
char* finalOutPointerr = finalOutt(img_bytes);
char* ichunk = finalOutPointerr;

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
cudaMemset(NContributedGaussians, 0, 640000 * sizeof(uint32_t));
cudaMemset(tileSliceStartEnd, 0,2500 * sizeof(uint2));
float3* projectedCovariance; float* twoDDet; float3* O;
cudaMalloc(&projectedCovariance, numGaussians * sizeof(float3));
cudaMalloc(&twoDDet , numGaussians * sizeof(float ));
cudaMalloc(&O , numGaussians * sizeof(float3));
PROJECTION::ProjectG(
    numGaussians, means3D,
    (glm::vec3*)scales,
    (glm::vec4*)orientationss,
    matViewCam, cameraFLenHorz, 
    cov3D, radss, intersectedTiles,  
    projectedCovariance, twoDDet, O,         
    gaussianAlphas, alphaConicOfgauss, 
    means2D, mattProjCam, zDistances  
);
ADR::tileCulling(
    numGaussians, gaussianAlphas,
    projectedCovariance, twoDDet, means2D,
     radss, intersectedTiles
);
FORWARD::calcColour(
    numGaussians, shNum, M, means3D,
    (glm::vec3*)positionOfCamera, sphericalHarmonics,
    clamped, rgb
);
kogge_stone_inclusive_scan_shared_memory(
    intersectedTiles,
    locationsOfPointsWRT,
    numGaussians
);
// kogge_stone_inclusive_scan_ultra_warp_optimized(
//     intersectedTiles,
//     locationsOfPointsWRT,
//     numGaussians
// );
int NGaussiansRendered;
uint32_t* MainGaussianIdArray;
uint32_t* unsortedPoints;
uint64_t* keysArray;
uint64_t* unsortedPointsKeys;
prepareAndSort(BBuffer, numGaussians, locationsOfPointsWRT, means2D, zDistances, radss,
               NGaussiansRendered,
               MainGaussianIdArray, unsortedPoints, keysArray, unsortedPointsKeys);
if (NGaussiansRendered > 0) 
{
    gaussianTiles<<<(NGaussiansRendered + 255) / 256, 256>>>(
       
        keysArray,       
        tileSliceStartEnd,
         NGaussiansRendered    
    );
}
   float* pointerForFeatures = rgb;
pixelGaussiansFull(
     block,
    tileSliceStartEnd,
    MainGaussianIdArray,  
    means2D,
    pointerForFeatures,
    zDistances,
    alphaConicOfgauss,
    opacityOP,
    NContributedGaussians,
    colOP,
    ZOP,
    projectedCovariance
);
//clean up
cudaFree(projectedCovariance);
cudaFree(twoDDet);
cudaFree(O);
return NGaussiansRendered;

}

// __global__ void kogge_stone_inclusive_scan_ultra_warp_optimized_kernel(const uint32_t* __restrict__ input_array, uint32_t* __restrict__ output_array, uint32_t* __restrict__ block_sums, int num_of_elements) {
    
//     const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (thread_id >= num_of_elements) return;
    
//     extern __shared__ volatile uint32_t shared_output_array[];
//     grid_group grid = this_grid();
    
//     const uint32_t input_val = __ldg(&input_array[thread_id]);
//     uint32_t val = input_val;
    
//     const int lane = threadIdx.x & 31;
//     const int warp_id = threadIdx.x >> 5;
//     const int num_warps = blockDim.x >> 5;
    
//     uint32_t temp;
//     temp = __shfl_up_sync(0xffffffff, val, 1);  if (lane >= 1)  val += temp;
//     temp = __shfl_up_sync(0xffffffff, val, 2);  if (lane >= 2)  val += temp;
//     temp = __shfl_up_sync(0xffffffff, val, 4);  if (lane >= 4)  val += temp;
//     temp = __shfl_up_sync(0xffffffff, val, 8);  if (lane >= 8)  val += temp;
//     temp = __shfl_up_sync(0xffffffff, val, 16); if (lane >= 16) val += temp;
    
//     if (lane == 31) {
//         shared_output_array[warp_id] = val;
//     }
//     __syncthreads();
    
//     if (warp_id == 0 && threadIdx.x < num_warps) {
//         uint32_t warp_sum = shared_output_array[threadIdx.x];
        
//         temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 1);  if (threadIdx.x >= 1)  warp_sum += temp;
//         temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 2);  if (threadIdx.x >= 2)  warp_sum += temp;
//         temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 4);  if (threadIdx.x >= 4)  warp_sum += temp;
//         temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 8);  if (threadIdx.x >= 8)  warp_sum += temp;
//         temp = __shfl_up_sync((1u << num_warps) - 1, warp_sum, 16); if (threadIdx.x >= 16) warp_sum += temp;
        
//         shared_output_array[threadIdx.x] = warp_sum;
//     }
//     __syncthreads();
    
//     if (warp_id > 0) {
//         val += shared_output_array[warp_id - 1];
//     }
    
//     if (threadIdx.x == blockDim.x - 1) {
//         block_sums[blockIdx.x] = val;
//     }

//     grid.sync();

//     if (thread_id == 0) {
//         const int num_blocks = gridDim.x;
//         uint32_t running_sum = block_sums[0];
        
//         if (num_blocks <= 8) {
//             if (num_blocks > 1) { block_sums[1] += running_sum; running_sum = block_sums[1]; }
//             if (num_blocks > 2) { block_sums[2] += running_sum; running_sum = block_sums[2]; }
//             if (num_blocks > 3) { block_sums[3] += running_sum; running_sum = block_sums[3]; }
//             if (num_blocks > 4) { block_sums[4] += running_sum; running_sum = block_sums[4]; }
//             if (num_blocks > 5) { block_sums[5] += running_sum; running_sum = block_sums[5]; }
//             if (num_blocks > 6) { block_sums[6] += running_sum; running_sum = block_sums[6]; }
//             if (num_blocks > 7) { block_sums[7] += running_sum; }
//         } else {
//             #pragma unroll 4
//             for (int i = 1; i < num_blocks; i++) {
//                 block_sums[i] += block_sums[i - 1];
//             }
//         }
//     }
        
//     grid.sync();

//     const uint32_t block_sum_to_add = (blockIdx.x == 0) ? 0 : block_sums[blockIdx.x - 1];
//     output_array[thread_id] = val + block_sum_to_add;
// }
