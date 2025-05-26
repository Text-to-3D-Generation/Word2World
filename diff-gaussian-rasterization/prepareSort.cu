#include "prepareSort.h"
#include <cub/cub.cuh>
#include <cstdint>




// ----------------------------------------------------------------------------
// duplicateWithKeys
//
// For every *Gaussian i* this kernel duplicates the Gaussian once for each
// screen–tile it overlaps and emits:
//
//   • unsortedGVals[k]  = i                     (gaussian-ID)
//   • unsortedGKeys[k]  = (depthBits<<32)|tile  (64-bit key)
//
// so that a later radix sort can group by tile and order by depth.
//
// grid-stride:   <<< (numGaussians+255)/256 , 256 >>>
// ----------------------------------------------------------------------------
__global__ void duplicateWithKeys(
    int              numGaussians,       // total P
       float2*    __restrict__ XYpoints,      // [P] centre in pixel space
       float*     __restrict__ zDistances,    // [P] depth (positive)
       uint32_t*  __restrict__ offsets,       // [P] prefix-sum of #tiles
    uint64_t*        __restrict__ unsortedGKeys, // [N] output keys
    uint32_t*        __restrict__ unsortedGVals, // [N] output gaussian IDs
       int*       __restrict__ radss,         // [P] pixel radius
    dim3             grid)                       // grid.x = tilesX, grid.y = tilesY
{
    // ------------------------------------------------------------
    // 1.  Identify which Gaussian this thread handles
    // ------------------------------------------------------------
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numGaussians)     return;          // out-of-range threads exit
    if (radss[idx] <= 0)         return;          // invisible → nothing to duplicate

    // ------------------------------------------------------------
    // 2.  Compute *first* write position for this Gaussian’s duplicates
    //     offsets = inclusive scan over #tiles touched
    // ------------------------------------------------------------
    //so simply, if you are the first gaussian-> start at 0
    //else, go to offsets[idx-1]
    uint32_t outPtr = (idx == 0) ? 0 : offsets[idx - 1];


    uint2 topLetfGaussianCornerrad, bottomRightGaussianCornerrad;
topLetfGaussianCornerrad = {
    min(grid.x, max(0, (int)((XYpoints[idx].x - radss[idx]) / 16))),
    min(grid.y, max(0, (int)((XYpoints[idx].y - radss[idx]) / 16)))
};
bottomRightGaussianCornerrad = {
    min(grid.x, max(0, (int)((XYpoints[idx].x + radss[idx] + 15) / 16))),
    min(grid.y, max(0, (int)((XYpoints[idx].y + radss[idx] + 15) / 16)))
};


    // ------------------------------------------------------------
    // 4.  Convenience: treat the 64-bit key buffer as array of uint2
    //     so we can write low/high 32-bit parts in one store.
    //     keys32bits[k] = {high, low}.
    // ------------------------------------------------------------
    auto* keys32bits = reinterpret_cast<uint2*>(unsortedGKeys);

    // ------------------------------------------------------------
    // 5.  Encode depth once (float → bit-cast) so numerical order = float order
    //     We keep it in the **high** 32 bits of the key.
    // ------------------------------------------------------------
       uint32_t depthBits = *reinterpret_cast<   uint32_t*>(&zDistances[idx]);

    // ------------------------------------------------------------
    // 6.  Loop over all tiles inside the bounding rectangle and emit
    //     one (key,value) pair per tile
    // ------------------------------------------------------------
    for (int ty = topLetfGaussianCornerrad.y; ty < bottomRightGaussianCornerrad.y; ++ty)
    {
        for (int tx = topLetfGaussianCornerrad.x; tx < bottomRightGaussianCornerrad.x; ++tx, ++outPtr)
        {
            // Flatten (tx,ty) to a linear tile index: row-major order
               uint32_t tileId = ty * grid.x + tx;

            // ----------- 6.2  Build 64-bit sort key  ------------------------
            // key =  (uint64_t)tileId << 32  |  depthBits
            //
            // Because tileId occupies the HIGH 32 bits, a plain ascending
            // radix-sort first groups by tile, and only when the high half is
            // equal does the low half (depthBits) matter → produces a list
            // “all gaussians of tile 0, back→front; then tile 1, ...”.
            //
            // uint2{x,y} writes  x = LOW 32 bits, y = HIGH 32 bits.
            keys32bits[outPtr] = make_uint2(depthBits, tileId);   // {low,high}

            // -- write value: gaussian index i
            unsortedGVals[outPtr] = idx;
        }
    }
}



BinningCarved prepareAndSort(
    std::function<char*(size_t)> BBuffer,
    int                           numGaussians,
       uint32_t*               locationsOfPointsWRT,
       float2*                 means2D,
       float*                  zDistances,
       int*                    radss,
    dim3                          gridOfTiles,
    int&                          totalGaussianInstances)
{
    // 1) Read last offset to get #rendered
cudaMemcpy(
        &totalGaussianInstances,
        locationsOfPointsWRT + (numGaussians - 1),
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    auto align128 = [](size_t x) { return (x + 127) & ~static_cast<size_t>(127); };

    size_t offset = 0;
    offset = align128(offset) + totalGaussianInstances * sizeof(uint32_t);  // MainGaussianIdArray
    offset = align128(offset) + totalGaussianInstances * sizeof(uint32_t);  // unsortedPoints
    offset = align128(offset) + totalGaussianInstances * sizeof(uint64_t);  // unsortedKeys
    offset = align128(offset) + totalGaussianInstances * sizeof(uint64_t);  // keys

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        (uint64_t*)nullptr, (uint64_t*)nullptr,
        (uint32_t*)nullptr, (uint32_t*)nullptr,
        totalGaussianInstances
    );
    offset = align128(offset) + temp_bytes;

    size_t chunk_size = offset + 128;
    char* chunk = BBuffer(chunk_size);

    std::uintptr_t addr;

    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
    uint32_t* MainGaussianIdArray = reinterpret_cast<uint32_t*>(addr);
    chunk = reinterpret_cast<char*>(MainGaussianIdArray + totalGaussianInstances);

    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
    uint32_t* unsortedPoints = reinterpret_cast<uint32_t*>(addr);
    chunk = reinterpret_cast<char*>(unsortedPoints + totalGaussianInstances);

    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
    uint64_t* unsortedPointsKeys = reinterpret_cast<uint64_t*>(addr);
    chunk = reinterpret_cast<char*>(unsortedPointsKeys + totalGaussianInstances);

    addr = align128(reinterpret_cast<std::uintptr_t>(chunk));
    uint64_t* keysArray = reinterpret_cast<uint64_t*>(addr);
    chunk = reinterpret_cast<char*>(keysArray + totalGaussianInstances);

    // Launch kernel to fill unsorted arrays
    duplicateWithKeys<<<(numGaussians + 255) / 256, 256>>>(
        numGaussians, means2D, zDistances, locationsOfPointsWRT,
        unsortedPointsKeys, unsortedPoints,
        radss, gridOfTiles
    );

    // Sort using CUB
    void* temporaryDeviceStorgae = chunk;
    cub::DeviceRadixSort::SortPairs(
        temporaryDeviceStorgae, temp_bytes,
        unsortedPointsKeys, keysArray,
        unsortedPoints, MainGaussianIdArray,
        totalGaussianInstances
    );

    BinningCarved carved;
    carved.MainGaussianIdArray = MainGaussianIdArray;
    carved.keysArray = keysArray;
    return carved;
}
