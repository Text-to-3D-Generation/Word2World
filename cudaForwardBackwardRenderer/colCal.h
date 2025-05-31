#ifndef COLCAL_H_INCLUDED
#define COLCAL_H_INCLUDED

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

//#include "colCalDev.h"

namespace   COLCAL {

void calcColour(
    int numGaussians, int shNum, int M,
       float* means3D,
       glm::vec3* positionOfCamera,
       float* sphericalHarmonics,
    bool* clamped,
    float* rgb
);

} // namespace FORWARD

#endif // COLCAL_H_INCLUDED
