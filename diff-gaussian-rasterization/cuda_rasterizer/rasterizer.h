#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

int gaussianForwardPass(
    float* opacityOP,                                           
   float cameraFLenVert,                                  
    std::function<char* (size_t)> imageBuffer,                  
    int* radss,                                                 
  float* orientationss,                                
     int numGaussians,                                     
     float* scales,                                        
     int M,                                                
     float* positionOfCamera,                              
    std::function<char* (size_t)> BBuffer,                       
    float* matViewCam,                                     
    float* ZOP,                                                 
    float* mattProjCam,                                  
    std::function<char* (size_t)> gaussianInfoMemSpace,               
     int shNum,                                                
     float* sphericalHarmonics,                            
     float* means3D,                                       
     float* gaussianAlphas,                                
    float cameraFLenHorz,                                  
    float* colOP                                                 
);

void backward(
    const int numGaussians, int shNum, int M, int R,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* sphericalHarmonics,
    const float* colors_precomp,
    const float* alphas,
    const float* scales,
    const float scale_modifier,
    const float* orientationss,
    const float* cov3D_precomp,
    const float* matViewCam,
    const float* mattProjCam,
    const float* campos,
    const float camHorizLengthTan, float camVertLengthTan,
    const int* radss,
    char* geom_buffer,
    char* binning_buffer,
    char* image_buffer,
    const float* dL_dpix,
    const float* dL_dpix_depth,
    const float* dL_dalphas,
    float* dL_dmean2D,
    float* dL_dconic,
    float* dL_dopacity,
    float* dL_dcolor,
    float* dL_ddepth,
    float* dL_dmean3D,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dscale,
    float* dL_drot,
    bool debug);

#endif
