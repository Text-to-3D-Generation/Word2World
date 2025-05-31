#pragma once
#include <functional>
#include <cuda_runtime.h>
#include <cstdint>
void prepareAndSort(
    std::function<char*(size_t)> BBuffer,int numGaussians,
    uint32_t* locationsOfPointsWRT,
    float2* means2D,float* zDistances,int* radss,
    int& totalGaussianInstances,
    uint32_t*& MainGaussianIdArray,
    uint32_t*& unsortedPoints,
    uint64_t*& keysArray,
    uint64_t*& unsortedPointsKeys
);

