#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "projection.h"

__global__ void ProjectGCuda(
    int numGaussians,
       float* DefaultPoints,
       glm::vec3* scales,
       glm::vec4* orientationss,
       float* matViewCam,
       float cameraFLenHorz, float cameraFLenVert,
    float* covvarance3D,
    int* radss,
    uint32_t* intersectedTiles,
    float3* covOP,  
    float* detOP,   
    float3* defOP,
       float* gaussianAlphas,
    float4* alphaConicOfgauss,
    float2* muu2dPixelCoord,
       float* mattProjCam,
    float* zDistances
) {
        float camHorizLengthTan = 800.0f / (2.0f * cameraFLenHorz);
        float camVertLengthTan = 800.0f / (2.0f * cameraFLenVert);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numGaussians) return;

    radss[idx] = 0;
    intersectedTiles[idx] = 0;

    float3 origPointt = {
        DefaultPoints[3 * idx],
        DefaultPoints[3 * idx + 1],
        DefaultPoints[3 * idx + 2]
    };

    glm::mat3 S = glm::mat3(
        glm::vec3( scales[idx].x, 0.0f, 0.0f),
        glm::vec3(0.0f,  scales[idx].y, 0.0f),
        glm::vec3(0.0f, 0.0f,  scales[idx].z)
    );

       float r = orientationss[idx].x;
       float x = orientationss[idx].y;
       float y = orientationss[idx].z;
       float z = orientationss[idx].w;

       float xx = x * x, yy = y * y, zz = z * z;
       float xy = x * y, xz = x * z, yz = y * z;
       float rx = r * x, ry = r * y, rz = r * z;

    glm::mat3 R = 0.5f * glm::mat3(
        glm::vec3(2.0f - 4.0f * (yy + zz),     4.0f * (xy - rz),         4.0f * (xz + ry)),
        glm::vec3(4.0f * (xy + rz),            2.0f - 4.0f * (xx + zz),  4.0f * (yz - rx)),
        glm::vec3(4.0f * (xz - ry),            4.0f * (yz + rx),         2.0f - 4.0f * (xx + yy))
    );

    glm::mat3 Sigma = glm::transpose(R) * glm::transpose(S) * S * R;

    float* cov3D = covvarance3D + idx * 6;
    int ii = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
            cov3D[ii++] = Sigma[i][j];

    glm::mat4 matrixx = glm::make_mat4(matViewCam);
    glm::vec4 point4(glm::vec3(origPointt.x, origPointt.y, origPointt.z), 1.0f);
    glm::vec4 transformed = matrixx * point4;
    float w = transformed.w + 1e-8f;
    float3 t = make_float3(transformed.x / w, transformed.y / w, transformed.z / w);
    if (t.z <= 1.0f)
    { return;
    }

    float2 pseudo_ndc = make_float2(t.x / t.z, t.y / t.z);
    float2 limit = make_float2(1.3f * camHorizLengthTan, 1.3f * camVertLengthTan);
    float nx = pseudo_ndc.x / limit.x;
    float ny = pseudo_ndc.y / limit.y;
    float squaredRadius = nx * nx + ny * ny;

    if (squaredRadius > 1.0f) {
        float scale = rsqrtf(squaredRadius);
        pseudo_ndc.x *= scale;
        pseudo_ndc.y *= scale;
    }

    t.x = pseudo_ndc.x * t.z;
    t.y = pseudo_ndc.y * t.z;

    glm::vec3 right   = glm::vec3(matViewCam[0], matViewCam[4], matViewCam[8]);
    glm::vec3 up      = glm::vec3(matViewCam[1], matViewCam[5], matViewCam[9]);
    glm::vec3 gaussianForwardPass = glm::vec3(matViewCam[2], matViewCam[6], matViewCam[10]);

    float inv_z = 1.0f / t.z;
    float inv_z2 = inv_z * inv_z;
    glm::vec3 dproj_dx = cameraFLenHorz * (right * inv_z - t.x * gaussianForwardPass * inv_z2);
    glm::vec3 dproj_dy = cameraFLenVert * (up    * inv_z - t.y * gaussianForwardPass * inv_z2);

    glm::mat3 projectedSigma = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    float2 row0 = {
        dot(dproj_dx, projectedSigma * dproj_dx) + 0.3f,
        dot(dproj_dx, projectedSigma * dproj_dy)
    };
    float2 row1 = {
        row0.y,
        dot(dproj_dy, projectedSigma * dproj_dy) + 0.3f
    };

    float3 covarianceMat = make_float3(row0.x, row0.y, row1.y);

    float determinent = covarianceMat.x * covarianceMat.z - covarianceMat.y * covarianceMat.y;
    if (fabsf(determinent) < 1e-8f) return;

    float determinentInverse = __frcp_rn(determinent);

    float a =  covarianceMat.z * determinentInverse;
    float b = -covarianceMat.y * determinentInverse;
    float c =  covarianceMat.x * determinentInverse;

    alphaConicOfgauss[idx] = make_float4(a, b, c, gaussianAlphas[idx]);

    covOP[idx]  = covarianceMat;
    detOP[idx]  = determinent;
    defOP[idx]  = origPointt;
    zDistances[idx]    = t.z;

    glm::mat4 Pe = *reinterpret_cast<   glm::mat4*>(mattProjCam);
    glm::vec4 clip = Pe * glm::vec4(origPointt.x, origPointt.y, origPointt.z, 1.0f);
    float invW = 1.0f / (clip.w + 1e-7f);
    glm::vec3 ndc  = glm::vec3(clip.x, clip.y, clip.z) * invW;

    float pixX = (ndc.x + 1.0f) * 0.5f * float(800) - 0.5f;
    float pixY = (ndc.y + 1.0f) * 0.5f * float(800) - 0.5f;

    muu2dPixelCoord[idx] = make_float2(pixX, pixY);
    zDistances[idx] = ndc.z;
}

namespace PROJECTION {

void ProjectG(
    int numGaussians,
       float* means3D,
       glm::vec3* scales,
       glm::vec4* orientationss,
       float* matViewCam,
       float cameraFLenHorz, float cameraFLenVert,
    float* covvarance3D,
    int* radss,
    uint32_t* intersectedTiles,
    float3* covOP,
    float* detOP,
    float3* defOP,
       float* gaussianAlphas,
    float4* alphaConicOfgauss,
    float2* means2D,
       float* mattProjCam,
    float* zDistances
) {
    ProjectGCuda<<<(numGaussians + 255) / 256, 256>>>(
        numGaussians,
        means3D,
        scales,
        orientationss,
        matViewCam,
        cameraFLenHorz, cameraFLenVert,
        covvarance3D,
        radss,
        intersectedTiles,
        covOP,
        detOP,
        defOP,
        gaussianAlphas,
        alphaConicOfgauss,
        means2D,
        mattProjCam,
        zDistances
    );
}

} 
