#include "gaussianTiles.h"

//L : Total number of gaussian instances
//keysArray : 64-bit keys of the gaussian instances
//tileSliceStartEnd : Array of uint2, where each element stores the start and end index of the corresponding tile
__global__ void gaussianTiles(int L, uint64_t* keysArray, uint2* tileSliceStartEnd)
{
    // Thread index = position in the sorted keysArray array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L)
        return;

    // Get current key and extract tile ID (top 32 bits)
    uint64_t key = keysArray[idx];
    uint32_t currtile = key >> 32;

    // Case 1: First entry in list (idx == 0)
    // → Start of the first tile’s range
    if (idx == 0) {
        tileSliceStartEnd[currtile].x = 0;
    }
    else {
        // Get previous tile ID (from previous key)
        uint32_t prevtile = keysArray[idx - 1] >> 32;

        // Case 2: Tile ID changes between prev and current
        // → Previous tile ends here, current tile starts here
        if (currtile != prevtile) {
            tileSliceStartEnd[prevtile].y = idx;    // End of prev tile's range
            tileSliceStartEnd[currtile].x = idx;    // Start of current tile's range
        }
    }

    // Case 3: Last element in the array
    // → Mark the end of the current tile's range
    if (idx == L - 1) {
        tileSliceStartEnd[currtile].y = L;
    }
}

