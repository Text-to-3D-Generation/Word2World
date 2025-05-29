#define BLOCK_X 16
#define BLOCK_Y 16
#define NUM_CHANNELS 3

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	glm::vec3 gaussian_center = means[idx];
	float offset_x = gaussian_center.x - campos.x;
	float offset_y = gaussian_center.y - campos.y;
	float offset_z = gaussian_center.z - campos.z;
	
	glm::vec3 raw_direction(offset_x, offset_y, offset_z);
	
	float mag_squared = offset_x * offset_x + offset_y * offset_y + offset_z * offset_z;
	float mag = sqrtf(mag_squared + 1e-10f);
	float inv_mag = 1.0f / mag;
	
	float dir_x = offset_x * inv_mag;
	float dir_y = offset_y * inv_mag;
	float dir_z = offset_z * inv_mag;
	
	float x2 = dir_x * dir_x;
	float y2 = dir_y * dir_y;
	float z2 = dir_z * dir_z;
	float xy = dir_x * dir_y;
	float xz = dir_x * dir_z;
	float yz = dir_y * dir_z;
	float x2_minus_y2 = x2 - y2;
	float three_x2 = 3.0f * x2;
	float three_y2 = 3.0f * y2;
	float four_z2 = 4.0f * z2;
	float two_z2 = 2.0f * z2;
	
	glm::vec3 clamp_factors(
		clamped[3 * idx + 0] ? 0.0f : 1.0f,
		clamped[3 * idx + 1] ? 0.0f : 1.0f,
		clamped[3 * idx + 2] ? 0.0f : 1.0f
	);
	
	glm::vec3 clamped_dL_dRGB = dL_dcolor[idx] * clamp_factors;
	
	glm::vec3* sh_data = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3* gradient_sh = dL_dshs + idx * max_coeffs;
	
	float basis_functions[16];
	basis_functions[0] = SH_C0;
	basis_functions[1] = -SH_C1 * dir_y;
	basis_functions[2] = SH_C1 * dir_z;
	basis_functions[3] = -SH_C1 * dir_x;
	basis_functions[4] = SH_C2[0] * xy;
	basis_functions[5] = SH_C2[1] * yz;
	basis_functions[6] = SH_C2[2] * (two_z2 - x2 - y2);
	basis_functions[7] = SH_C2[3] * xz;
	basis_functions[8] = SH_C2[4] * x2_minus_y2;
	basis_functions[9] = SH_C3[0] * dir_y * (three_x2 - y2);
	basis_functions[10] = SH_C3[1] * xy * dir_z;
	basis_functions[11] = SH_C3[2] * dir_y * (four_z2 - x2 - y2);
	basis_functions[12] = SH_C3[3] * dir_z * (two_z2 - three_x2 - three_y2);
	basis_functions[13] = SH_C3[4] * dir_x * (four_z2 - x2 - y2);
	basis_functions[14] = SH_C3[5] * dir_z * x2_minus_y2;
	basis_functions[15] = SH_C3[6] * dir_x * (x2 - three_y2);
	
	for (int i = 0; i < 16; ++i) {
		gradient_sh[i] = basis_functions[i] * clamped_dL_dRGB;
	}
	
	glm::vec3 accumulated_gradient(0.0f, 0.0f, 0.0f);
	
	accumulated_gradient.x -= SH_C1 * glm::dot(sh_data[3], clamped_dL_dRGB);
	accumulated_gradient.y -= SH_C1 * glm::dot(sh_data[1], clamped_dL_dRGB);
	accumulated_gradient.z += SH_C1 * glm::dot(sh_data[2], clamped_dL_dRGB);
	
	glm::vec3 deg2_contrib_x(
		SH_C2[0] * dir_y * sh_data[4] + 
		SH_C2[2] * (-2.0f * dir_x) * sh_data[6] + 
		SH_C2[3] * dir_z * sh_data[7] + 
		SH_C2[4] * (2.0f * dir_x) * sh_data[8]
	);
	
	glm::vec3 deg2_contrib_y(
		SH_C2[0] * dir_x * sh_data[4] + 
		SH_C2[1] * dir_z * sh_data[5] + 
		SH_C2[2] * (-2.0f * dir_y) * sh_data[6] + 
		SH_C2[4] * (-2.0f * dir_y) * sh_data[8]
	);
	
	glm::vec3 deg2_contrib_z(
		SH_C2[1] * dir_y * sh_data[5] + 
		SH_C2[2] * (4.0f * dir_z) * sh_data[6] + 
		SH_C2[3] * dir_x * sh_data[7]
	);
	
	accumulated_gradient.x += glm::dot(deg2_contrib_x, clamped_dL_dRGB);
	accumulated_gradient.y += glm::dot(deg2_contrib_y, clamped_dL_dRGB);
	accumulated_gradient.z += glm::dot(deg2_contrib_z, clamped_dL_dRGB);
	
	float deg3_factors_x[7] = {
		SH_C3[0] * 6.0f * xy,
		SH_C3[1] * yz,
		SH_C3[2] * (-2.0f * xy),
		SH_C3[3] * (-6.0f * xz),
		SH_C3[4] * (four_z2 - 3.0f * x2 - y2),
		SH_C3[5] * 2.0f * xz,
		SH_C3[6] * 3.0f * x2_minus_y2
	};
	
	float deg3_factors_y[7] = {
		SH_C3[0] * 3.0f * (x2 - y2),
		SH_C3[1] * xz,
		SH_C3[2] * (four_z2 - 3.0f * y2 - x2),
		SH_C3[3] * (-6.0f * yz),
		SH_C3[4] * (-2.0f * xy),
		SH_C3[5] * (-2.0f * yz),
		SH_C3[6] * (-6.0f * xy)
	};
	
	float deg3_factors_z[7] = {
		0.0f,
		SH_C3[1] * xy,
		SH_C3[2] * 8.0f * yz,
		SH_C3[3] * 3.0f * (two_z2 - x2 - y2),
		SH_C3[4] * 8.0f * xz,
		SH_C3[5] * x2_minus_y2,
		0.0f
	};
	
	glm::vec3 deg3_x(0.0f), deg3_y(0.0f), deg3_z(0.0f);
	for (int i = 0; i < 7; ++i) {
		deg3_x += deg3_factors_x[i] * sh_data[9 + i];
		deg3_y += deg3_factors_y[i] * sh_data[9 + i];
		deg3_z += deg3_factors_z[i] * sh_data[9 + i];
	}
	
	accumulated_gradient.x += glm::dot(deg3_x, clamped_dL_dRGB);
	accumulated_gradient.y += glm::dot(deg3_y, clamped_dL_dRGB);
	accumulated_gradient.z += glm::dot(deg3_z, clamped_dL_dRGB);
	
	float inv_mag_cubed = inv_mag * inv_mag * inv_mag;
	
	glm::mat3 outer_prod(
		raw_direction.x * raw_direction.x, raw_direction.x * raw_direction.y, raw_direction.x * raw_direction.z,
		raw_direction.y * raw_direction.x, raw_direction.y * raw_direction.y, raw_direction.y * raw_direction.z,
		raw_direction.z * raw_direction.x, raw_direction.z * raw_direction.y, raw_direction.z * raw_direction.z
	);
	
	glm::mat3 scaled_I = glm::mat3(mag_squared);
	
	glm::mat3 jacobian = (scaled_I - outer_prod) * inv_mag_cubed;
	
	glm::vec3 final_mean_gradient = jacobian * accumulated_gradient;
	
	dL_dmeans[idx] += final_mean_gradient;
}

__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float* packed_cov = cov3Ds + 6 * idx;
	
	float3 center = means[idx];
	
	float view_x = view_matrix[0] * center.x + view_matrix[4] * center.y + view_matrix[8] * center.z + view_matrix[12];
	float view_y = view_matrix[1] * center.x + view_matrix[5] * center.y + view_matrix[9] * center.z + view_matrix[13];
	float view_z = view_matrix[2] * center.x + view_matrix[6] * center.y + view_matrix[10] * center.z + view_matrix[14];
	float view_w = view_matrix[3] * center.x + view_matrix[7] * center.y + view_matrix[11] * center.z + view_matrix[15];
	
	float w_inv = 1.0f / (view_w + 1e-8f);
	view_x *= w_inv;
	view_y *= w_inv;
	view_z *= w_inv;
	
	float3 view_pos = make_float3(view_x, view_y, view_z);
	
	const float bound_x = 1.3f * tan_fovx;
	const float bound_y = 1.3f * tan_fovy;
	
	const float proj_x = view_x / view_z;
	const float proj_y = view_y / view_z;
	
	float clipped_proj_x = fminf(bound_x, fmaxf(-bound_x, proj_x));
	float clipped_proj_y = fminf(bound_y, fmaxf(-bound_y, proj_y));
	view_pos.x = clipped_proj_x * view_z;
	view_pos.y = clipped_proj_y * view_z;
	
	const float mask_x = (proj_x >= -bound_x && proj_x <= bound_x) ? 1.0f : 0.0f;
	const float mask_y = (proj_y >= -bound_y && proj_y <= bound_y) ? 1.0f : 0.0f;
	
	float z_inv = 1.0f / view_pos.z;
	float z_inv2 = z_inv * z_inv;
	
	glm::mat3 jacobian_proj = glm::mat3(
		h_x * z_inv, 0.0f, -h_x * view_pos.x * z_inv2,
		0.0f, h_y * z_inv, -h_y * view_pos.y * z_inv2,
		0.0f, 0.0f, 0.0f
	);
	
	glm::mat3 rotation_view = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]
	);
	
	glm::mat3 covariance_3d = glm::mat3(
		packed_cov[0], packed_cov[1], packed_cov[2],
		packed_cov[1], packed_cov[3], packed_cov[4],
		packed_cov[2], packed_cov[4], packed_cov[5]
	);
	
	glm::mat3 transform_matrix = rotation_view * jacobian_proj;
	
	glm::mat3 temp_cov = glm::transpose(covariance_3d) * transform_matrix;
	glm::mat3 covariance_2d = glm::transpose(transform_matrix) * temp_cov;
	
	covariance_2d[0][0] += 0.3f;
	covariance_2d[1][1] += 0.3f;
	
	float a = covariance_2d[0][0];
	float b = covariance_2d[0][1];
	float c = covariance_2d[1][1];
	
	float determinant = a * c - b * b;
	float det_inv_sq = 1.0f / (determinant * determinant + 1e-7f);
	
	float3 conic_grad = make_float3(
		dL_dconics[4 * idx],
		dL_dconics[4 * idx + 1],
		dL_dconics[4 * idx + 3]
	);
	
	float grad_a = 0.0f, grad_b = 0.0f, grad_c = 0.0f;
	
	if (det_inv_sq > 0) {
		grad_a = det_inv_sq * (
			-c * c * conic_grad.x + 
			2.0f * b * c * conic_grad.y + 
			(determinant - a * c) * conic_grad.z
		);
		
		grad_b = det_inv_sq * 2.0f * (
			b * c * conic_grad.x - 
			(determinant + 2.0f * b * b) * conic_grad.y + 
			a * b * conic_grad.z
		);
		
		grad_c = det_inv_sq * (
			-a * a * conic_grad.z + 
			2.0f * a * b * conic_grad.y + 
			(determinant - a * c) * conic_grad.x
		);
		
		dL_dcov[6 * idx + 0] = transform_matrix[0][0] * transform_matrix[0][0] * grad_a + 
		                       transform_matrix[0][0] * transform_matrix[1][0] * grad_b + 
		                       transform_matrix[1][0] * transform_matrix[1][0] * grad_c;
		
		dL_dcov[6 * idx + 3] = transform_matrix[0][1] * transform_matrix[0][1] * grad_a + 
		                       transform_matrix[0][1] * transform_matrix[1][1] * grad_b + 
		                       transform_matrix[1][1] * transform_matrix[1][1] * grad_c;
		
		dL_dcov[6 * idx + 5] = transform_matrix[0][2] * transform_matrix[0][2] * grad_a + 
		                       transform_matrix[0][2] * transform_matrix[1][2] * grad_b + 
		                       transform_matrix[1][2] * transform_matrix[1][2] * grad_c;
		
		dL_dcov[6 * idx + 1] = 2.0f * (transform_matrix[0][0] * transform_matrix[0][1] * grad_a + 
		                               (transform_matrix[0][0] * transform_matrix[1][1] + 
		                                transform_matrix[0][1] * transform_matrix[1][0]) * grad_b * 0.5f + 
		                               transform_matrix[1][0] * transform_matrix[1][1] * grad_c);
		
		dL_dcov[6 * idx + 2] = 2.0f * (transform_matrix[0][0] * transform_matrix[0][2] * grad_a + 
		                               (transform_matrix[0][0] * transform_matrix[1][2] + 
		                                transform_matrix[0][2] * transform_matrix[1][0]) * grad_b * 0.5f + 
		                               transform_matrix[1][0] * transform_matrix[1][2] * grad_c);
		
		dL_dcov[6 * idx + 4] = 2.0f * (transform_matrix[0][1] * transform_matrix[0][2] * grad_a + 
		                               (transform_matrix[0][1] * transform_matrix[1][2] + 
		                                transform_matrix[0][2] * transform_matrix[1][1]) * grad_b * 0.5f + 
		                               transform_matrix[1][1] * transform_matrix[1][2] * grad_c);
	} else {
		for (int i = 0; i < 6; i++) {
			dL_dcov[6 * idx + i] = 0.0f;
		}
	}
	
	float grad_T00 = 2.0f * (transform_matrix[0][0] * covariance_3d[0][0] + 
	                         transform_matrix[0][1] * covariance_3d[0][1] + 
	                         transform_matrix[0][2] * covariance_3d[0][2]) * grad_a +
	                 (transform_matrix[1][0] * covariance_3d[0][0] + 
	                  transform_matrix[1][1] * covariance_3d[0][1] + 
	                  transform_matrix[1][2] * covariance_3d[0][2]) * grad_b;
	
	float grad_T01 = 2.0f * (transform_matrix[0][0] * covariance_3d[1][0] + 
	                         transform_matrix[0][1] * covariance_3d[1][1] + 
	                         transform_matrix[0][2] * covariance_3d[1][2]) * grad_a +
	                 (transform_matrix[1][0] * covariance_3d[1][0] + 
	                  transform_matrix[1][1] * covariance_3d[1][1] + 
	                  transform_matrix[1][2] * covariance_3d[1][2]) * grad_b;
	
	float grad_T02 = 2.0f * (transform_matrix[0][0] * covariance_3d[2][0] + 
	                         transform_matrix[0][1] * covariance_3d[2][1] + 
	                         transform_matrix[0][2] * covariance_3d[2][2]) * grad_a +
	                 (transform_matrix[1][0] * covariance_3d[2][0] + 
	                  transform_matrix[1][1] * covariance_3d[2][1] + 
	                  transform_matrix[1][2] * covariance_3d[2][2]) * grad_b;
	
	float grad_T10 = 2.0f * (transform_matrix[1][0] * covariance_3d[0][0] + 
	                         transform_matrix[1][1] * covariance_3d[0][1] + 
	                         transform_matrix[1][2] * covariance_3d[0][2]) * grad_c +
	                 (transform_matrix[0][0] * covariance_3d[0][0] + 
	                  transform_matrix[0][1] * covariance_3d[0][1] + 
	                  transform_matrix[0][2] * covariance_3d[0][2]) * grad_b;
	
	float grad_T11 = 2.0f * (transform_matrix[1][0] * covariance_3d[1][0] + 
	                         transform_matrix[1][1] * covariance_3d[1][1] + 
	                         transform_matrix[1][2] * covariance_3d[1][2]) * grad_c +
	                 (transform_matrix[0][0] * covariance_3d[1][0] + 
	                  transform_matrix[0][1] * covariance_3d[1][1] + 
	                  transform_matrix[0][2] * covariance_3d[1][2]) * grad_b;
	
	float grad_T12 = 2.0f * (transform_matrix[1][0] * covariance_3d[2][0] + 
	                         transform_matrix[1][1] * covariance_3d[2][1] + 
	                         transform_matrix[1][2] * covariance_3d[2][2]) * grad_c +
	                 (transform_matrix[0][0] * covariance_3d[2][0] + 
	                  transform_matrix[0][1] * covariance_3d[2][1] + 
	                  transform_matrix[0][2] * covariance_3d[2][2]) * grad_b;
	
	float grad_J00 = rotation_view[0][0] * grad_T00 + rotation_view[0][1] * grad_T01 + rotation_view[0][2] * grad_T02;
	float grad_J02 = rotation_view[2][0] * grad_T00 + rotation_view[2][1] * grad_T01 + rotation_view[2][2] * grad_T02;
	float grad_J11 = rotation_view[1][0] * grad_T10 + rotation_view[1][1] * grad_T11 + rotation_view[1][2] * grad_T12;
	float grad_J12 = rotation_view[2][0] * grad_T10 + rotation_view[2][1] * grad_T11 + rotation_view[2][2] * grad_T12;
	
	float z_inv3 = z_inv2 * z_inv;
	float grad_view_x = mask_x * (-h_x * z_inv2 * grad_J02);
	float grad_view_y = mask_y * (-h_y * z_inv2 * grad_J12);
	float grad_view_z = -h_x * z_inv2 * grad_J00 - h_y * z_inv2 * grad_J11 + 
	                    2.0f * h_x * view_pos.x * z_inv3 * grad_J02 + 
	                    2.0f * h_y * view_pos.y * z_inv3 * grad_J12;
	
	float3 world_grad;
	world_grad.x = view_matrix[0] * grad_view_x + view_matrix[1] * grad_view_y + view_matrix[2] * grad_view_z;
	world_grad.y = view_matrix[4] * grad_view_x + view_matrix[5] * grad_view_y + view_matrix[6] * grad_view_z;
	world_grad.z = view_matrix[8] * grad_view_x + view_matrix[9] * grad_view_y + view_matrix[10] * grad_view_z;
	
	dL_dmeans[idx] = world_grad;
}

__device__ void computeCovarianceDerivatives(
    int element_index,
    const glm::vec3 dimensions,
    float scaling_factor,
    const glm::vec4 rotation_quat,
    const float* incoming_grads,
    glm::vec3* dimension_derivs,
    glm::vec4* rotation_derivs)
{
    // Unpack quaternion components
    const float qw = rotation_quat.x;
    const float qx = rotation_quat.y;
    const float qy = rotation_quat.z;
    const float qz = rotation_quat.w;

    // Compute auxiliary rotation terms
    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float xw = qx * qw, yw = qy * qw, zw = qz * qw;

    // Construct scaled dimensions
    const glm::vec3 scaled_dims = scaling_factor * dimensions;
    const float sx = scaled_dims.x, sy = scaled_dims.y, sz = scaled_dims.z;

    // Compute transformation matrix elements
    const float m11 = sx * (1.f - 2.f*(yy + zz));
    const float m12 = sx * 2.f*(xy - zw);
    const float m13 = sx * 2.f*(xz + yw);
    
    const float m21 = sy * 2.f*(xy + zw);
    const float m22 = sy * (1.f - 2.f*(xx + zz));
    const float m23 = sy * 2.f*(yz - xw);
    
    const float m31 = sz * 2.f*(xz - yw);
    const float m32 = sz * 2.f*(yz + xw);
    const float m33 = sz * (1.f - 2.f*(xx + yy));

    // Extract covariance gradients
    const float* grad = incoming_grads + 6 * element_index;
    const float g11 = grad[0], g12 = grad[1], g13 = grad[2];
    const float g22 = grad[3], g23 = grad[4], g33 = grad[5];

    // Compute intermediate products for gradient calculations
    const float a11 = 2.f * (m11*g11 + m12*g12 + m13*g13);
    const float a12 = 2.f * (m11*g12 + m12*g22 + m13*g23);
    const float a13 = 2.f * (m11*g13 + m12*g23 + m13*g33);
    
    const float a21 = 2.f * (m21*g11 + m22*g12 + m23*g13);
    const float a22 = 2.f * (m21*g12 + m22*g22 + m23*g23);
    const float a23 = 2.f * (m21*g13 + m22*g23 + m23*g33);
    
    const float a31 = 2.f * (m31*g11 + m32*g12 + m33*g13);
    const float a32 = 2.f * (m31*g12 + m32*g22 + m33*g23);
    const float a33 = 2.f * (m31*g13 + m32*g23 + m33*g33);

    // Compute dimension derivatives
    glm::vec3* dim_deriv = dimension_derivs + element_index;
    dim_deriv->x = (1.f - 2.f*(yy + zz))*a11 + 2.f*(xy - zw)*a12 + 2.f*(xz + yw)*a13;
    dim_deriv->y = 2.f*(xy + zw)*a21 + (1.f - 2.f*(xx + zz))*a22 + 2.f*(yz - xw)*a23;
    dim_deriv->z = 2.f*(xz - yw)*a31 + 2.f*(yz + xw)*a32 + (1.f - 2.f*(xx + yy))*a33;

    // Compute quaternion derivatives
    const float dq0 = 4.f * (
        -sz*(a31*yw - a32*xw + a33*xy) + 
        sy*(a21*zw - a22*xz + a23*yy) + 
        sx*(a12*zw - a13*yw - a11*yz)
    );

    const float dq1 = 4.f * (
        sz*(a31*zw + a32*zz - a33*yz) - 
        sy*(a21*yw + a22*xy + a23*xx) + 
        sx*(a12*yy + a13*zw + a11*xz)
    );

    const float dq2 = 4.f * (
        -sz*(a31*xx + a32*xw + a33*xz) + 
        sy*(a21*zw + a22*yy + a23*yw) - 
        sx*(a12*xw + a13*xx + a11*xy)
    );

    const float dq3 = 4.f * (
        sz*(a31*xw - a32*xx + a33*xy) - 
        sy*(a21*xz - a22*zw + a23*yz) + 
        sx*(a12*yz - a13*zz + a11*yy)
    );

    // Store rotation derivatives
    float4* rot_deriv = (float4*)(rotation_derivs + element_index);
    *rot_deriv = float4{dq0, dq1, dq2, dq3};
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	glm::vec3 dL_dmean2;
	float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
	dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
	dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_dmean2;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCovarianceDerivatives(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ alphas,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dalphas,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float accum_depth_rec = 0;
	float dL_dpixel_depth;
	float accum_alpha_rec = 0;
	float dL_dalpha;
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dalpha = dL_dalphas[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dopa = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dopa += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			
			// Propagate gradients from pixel depth to opacity
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dopa += (c_d - accum_depth_rec) * dL_dpixel_depth;
			atomicAdd(&(dL_ddepths[global_id]), dpixel_depth_ddepth * dL_dpixel_depth);

			// Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dopa += (1 - accum_alpha_rec) * dL_dalpha; //- (alpha - accum_alpha_rec) * dL_dalpha;

			dL_dopa *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dopa += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dopa;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dopa);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* alphas,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixel_depths,
	const float* dL_dalphas,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		alphas,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dalphas,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths
		);
}