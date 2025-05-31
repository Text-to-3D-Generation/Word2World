#ifndef GAUSSIAN_TILES_H_INCLUDED
#define GAUSSIAN_TILES_H_INCLUDED

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void gaussianTiles( uint64_t* keysArray, uint2* tileSliceStartEnd,int totRender);

#endif 
