#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

__device__ float compute_3d_gaussian_coefficient_device(float x, float y, float z, float a, float b, float c, float d, float e, float f) {
    float det = a * d * f + 2 * e * c * b - e * e * a - c * c * d - b * b * f;
    float inv_det = 1.0f / (det + 1e-24f);
    
    float inv_a = (d * f - e * e) * inv_det;
    float inv_b = (e * c - b * f) * inv_det;
    float inv_c = (e * b - c * d) * inv_det;
    float inv_d = (a * f - c * c) * inv_det;
    float inv_e = (b * c - e * a) * inv_det;
    float inv_f = (a * d - b * b) * inv_det;
    
    float power = -0.5f * (x * x * inv_a + y * y * inv_d + z * z * inv_f)
                  - x * y * inv_b - x * z * inv_c - y * z * inv_e;
    
    if (power > 0) power = -1e10f;
    
    return expf(power);
}

__global__ void compute_voxel_field_kernel(float* occ, const float* means, const float* covs, const float* opacities, const int n_gaussians, const int resolution, const int num_blocks, const float relax_ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= resolution || idy >= resolution || idz >= resolution) return;
    
    float x = -1.0f + 2.0f * idx / (resolution - 1);
    float y = -1.0f + 2.0f * idy / (resolution - 1);
    float z = -1.0f + 2.0f * idz / (resolution - 1);
    
    float block_size = 2.0f / num_blocks;
    int block_x = (int)((x + 1.0f) / block_size);
    int block_y = (int)((y + 1.0f) / block_size);
    int block_z = (int)((z + 1.0f) / block_size);
    
    float vmin_x = -1.0f + block_x * block_size - block_size * relax_ratio;
    float vmax_x = -1.0f + (block_x + 1) * block_size + block_size * relax_ratio;
    float vmin_y = -1.0f + block_y * block_size - block_size * relax_ratio;
    float vmax_y = -1.0f + (block_y + 1) * block_size + block_size * relax_ratio;
    float vmin_z = -1.0f + block_z * block_size - block_size * relax_ratio;
    float vmax_z = -1.0f + (block_z + 1) * block_size + block_size * relax_ratio;
    
    float val = 0.0f;
    
    for (int g = 0; g < n_gaussians; g++) {
        float mx = means[g * 3 + 0];
        float my = means[g * 3 + 1];
        float mz = means[g * 3 + 2];
        
        if (mx < vmin_x || mx > vmax_x || 
            my < vmin_y || my > vmax_y || 
            mz < vmin_z || mz > vmax_z) {
            continue;
        }
        
        float dx = x - mx;
        float dy = y - my;
        float dz = z - mz;
        
        float a = covs[g * 6 + 0];
        float b = covs[g * 6 + 1];
        float c = covs[g * 6 + 2];
        float d = covs[g * 6 + 3];
        float e = covs[g * 6 + 4];
        float f = covs[g * 6 + 5];
        
        float w = compute_3d_gaussian_coefficient_device(dx, dy, dz, a, b, c, d, e, f);
        
        val += opacities[g] * w;
    }
    
    int linear_idx = idx + idy * resolution + idz * resolution * resolution;
    occ[linear_idx] = val;
}

extern "C" {
    __declspec(dllexport) void compute_voxel_field_cuda(float* h_occ, const float* h_means, const float* h_covs, const float* h_opacities, const int n_gaussians, const int resolution, const int num_blocks, const float relax_ratio) {
        size_t occ_size = resolution * resolution * resolution * sizeof(float);
        size_t means_size = n_gaussians * 3 * sizeof(float);
        size_t covs_size = n_gaussians * 6 * sizeof(float);
        size_t opacities_size = n_gaussians * sizeof(float);
        
        float *d_occ, *d_means, *d_covs, *d_opacities;
        
        cudaMalloc(&d_occ, occ_size);
        cudaMalloc(&d_means, means_size);
        cudaMalloc(&d_covs, covs_size);
        cudaMalloc(&d_opacities, opacities_size);
        
        cudaMemcpy(d_means, h_means, means_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_covs, h_covs, covs_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_opacities, h_opacities, opacities_size, cudaMemcpyHostToDevice);
        
        cudaMemset(d_occ, 0, occ_size);
        
        dim3 blockSize(8, 8, 8);  
        dim3 gridSize(
            (resolution + blockSize.x - 1) / blockSize.x,
            (resolution + blockSize.y - 1) / blockSize.y,
            (resolution + blockSize.z - 1) / blockSize.z
        );
        
        compute_voxel_field_kernel<<<gridSize, blockSize>>>(
            d_occ, d_means, d_covs, d_opacities,
            n_gaussians, resolution, num_blocks, relax_ratio
        );
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
        }
        
        cudaMemcpy(h_occ, d_occ, occ_size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_occ);
        cudaFree(d_means);
        cudaFree(d_covs);
        cudaFree(d_opacities);
    }
}
