#pragma once

#include <functional>
#include <cuda_runtime.h>
#include <cstdint>
#include <glm/glm.hpp>


/// Carve out pointers for peparing arrays, no BinningState
struct BinningCarved {
    uint32_t* MainGaussianIdArray;
    uint32_t* unsortedPoints;
    uint64_t* keysArray;
    uint64_t* unsortedPointsKeys;
  };

// Forward declaration of the function
BinningCarved prepareAndSort(
    std::function<char*(size_t)> BBuffer,
    int                           numGaussians,
       uint32_t*               locationsOfPointsWRT,
       float2*                 means2D,
       float*                  zDistances,
       int*                    radss,
    dim3                          gridOfTiles,
    int&                          totalGaussianInstances
);
