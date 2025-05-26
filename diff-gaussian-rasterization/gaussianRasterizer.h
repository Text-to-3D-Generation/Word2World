

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}
	
	struct GeometryState
	{
		size_t scannedAmount;
		float* zDistances;
		char* scannedSppacee;
		bool* clamped;
		int* internalRs;
		float2* means2D;
		float* cov3D;
		float4* alphaConicOfgauss;
		float* rgb;
		uint32_t* locationsOfPointsWRT;
		uint32_t* intersectedTiles;

		static GeometryState fromChunk(char*& chunk, size_t numGaussians);
	};

	struct ImageState
	{
		uint2* tileSliceStartEnd;
		uint32_t* NContributedGaussians;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t spaceForSorting;
		uint64_t* unsortedPointsKeys;
		uint64_t* keysArray;
		uint32_t* unsortedPoints;
		uint32_t* MainGaussianIdArray;
		char* sortSpaceList;

		static BinningState fromChunk(char*& chunk, size_t numGaussians);
	};


	template<typename T> 
	size_t required(size_t numGaussians)
	{
		char* size = nullptr;
		T::fromChunk(size, numGaussians);
		return ((size_t)size) + 128;
	}
};

