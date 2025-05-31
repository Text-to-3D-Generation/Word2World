#include "gaussianTiles.h"

//totRender : Total number of gaussian instances
//keysArray : 64-bit keys of the gaussian instances
//tileSliceStartEnd : Array of uint2, where each element stores the start and end index of the corresponding tile
__global__ void gaussianTiles(  uint64_t*  sortedKeys, uint2*  tileBounds,int totalCount) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCount)
    {
        //printf("Thread %d is out of bounds (%d >= %d)\n", i, totalCount, totalCount);
         return;
    }

    // Get current tile ID from key
    uint32_t currentTile = sortedKeys[i] >> 32;

    // If this is the first occurrence of a tile (either i == 0 or different from previous)
    if (i == 0 || (sortedKeys[i - 1] >> 32) != currentTile)
     {
        //printf("Tile %u starts at index %d\n", currentTile, i);
        
        
        tileBounds[currentTile].x = i;
    }

    // If this is the last occurrence of a tile (either i == last or different from next)
    if (i == totalCount - 1 || (sortedKeys[i + 1] >> 32) != currentTile)
     {
        //printf("Tile %u ends at index %d\n", currentTile, i);
        tileBounds[currentTile].y = i + 1;
    }
}


