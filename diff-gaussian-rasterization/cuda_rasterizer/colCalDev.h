#pragma once
#include <glm/glm.hpp>

__device__    float SH_C0 = 0.28209479177387814f;
__device__    float SH_C1 = 0.4886025119029199f;
__device__    float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__    float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ __forceinline__ void evalSHDeg1(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    out += -SH_C1 * d.y * sh[1];
    out +=  SH_C1 * d.z * sh[2];
    out += -SH_C1 * d.x * sh[3];
}

__device__ __forceinline__ void evalSHDeg2(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
    float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;

    out += SH_C2[0] * xy                 * sh[4];
    out += SH_C2[1] * yz                 * sh[5];
    out += SH_C2[2] * (2 * zz - xx - yy) * sh[6];
    out += SH_C2[3] * xz                 * sh[7];
    out += SH_C2[4] * (xx - yy)          * sh[8];
}

__device__ __forceinline__ void evalSHDeg3(   glm::vec3& d,    glm::vec3* sh, glm::vec3& out) {
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;

    out += SH_C3[0] * d.y * (3 * xx - yy)              * sh[9];
    out += SH_C3[1] * d.x * d.y * d.z                  * sh[10];
    out += SH_C3[2] * d.y * (4 * zz - xx - yy)         * sh[11];
    out += SH_C3[3] * d.z * (2 * zz - 3 * xx - 3 * yy) * sh[12];
    out += SH_C3[4] * d.x * (4 * zz - xx - yy)         * sh[13];
    out += SH_C3[5] * d.z * (xx - yy)                  * sh[14];
    out += SH_C3[6] * d.x * (xx - 3 * yy)              * sh[15];
}
