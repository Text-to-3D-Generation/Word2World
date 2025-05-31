
 #ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
// 	// Perform initial steps for each Gaussian prior to rasterization.
	 void calcColour(
    int               numGaussians,
    int               shNum,
    int               M,
       float*      DefaultPoints,  
                glm::vec3*  positionOfCamera,      // pointer to a single camera pos
        float*      sphericalHarmonics,          // spherical harmonics coefficients
    bool*       backwardBoolean,
          
    
float*      rgb           // length numGaussians*3
);





}


#endif