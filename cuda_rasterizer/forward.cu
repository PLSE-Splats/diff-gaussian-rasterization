/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

// ReSharper disable CppTooWideScopeInitStatement
// ReSharper disable CppUseStructuredBinding
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_scan.cuh>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE)
skm_cluster_passCUDA(
	const int starting_splat_index,
	const int P,
	const int width,
	const int height,
	const dim3 grid_size,
	const int *radii,
	const float2 *means2d,
	const float4 *conic_opacity,
	const float *depths,
	const float *features,
	uint32_t *n_contrib,
	float *cluster_depth,
	int *cluster_splat_count,
	float *cluster_alpha_sum,
	float *cluster_alpha,
	float *cluster_premultiplied_r,
	float *cluster_premultiplied_g,
	float *cluster_premultiplied_b,
	int *cluster_uninitialized_cluster_index
) {
	// Gather thread information.
	const auto block = cg::this_thread_block();
	const auto group_index = block.group_index();
	const auto thread_index = block.thread_index();
	const auto thread_rank = block.thread_rank();

	// Gather pixel information.
	const uint2 minimum_pixel_coordinate = {group_index.x * BLOCK_X, group_index.y * BLOCK_Y};
	const uint2 pixel_coordinate = {
		minimum_pixel_coordinate.x + thread_index.x, minimum_pixel_coordinate.y + thread_index.y
	};
	const uint32_t pixel_index = width * pixel_coordinate.y + pixel_coordinate.x;

	// Compute if this thread is associated with a visible pixel.
	const bool pixel_in_bounds = pixel_coordinate.x < width && pixel_coordinate.y < height;

	// Contribution counters for backwards pass.
	uint32_t contributing_splat_count = 0;

	// Storage for hit-checked splats. Default to miss (-1).
	__shared__ int hit_indices[INGEST_SIZE];

	// Phase 1: Hit-check splats against this tile.
	for (int stride = static_cast<int>(thread_rank); stride < INGEST_SIZE; stride += BLOCK_SIZE) {
		// Get target splat index.
		const int target_splat_index = starting_splat_index + stride;

		// Default to miss (-1).
		hit_indices[stride] = -1;

		// Stop if splat is out of bounds.
		if (target_splat_index >= starting_splat_index + INGEST_SIZE || target_splat_index >= P)
			break;

		// Get splat radius and check if it intersects with the tile.
		const int splat_radius = radii[target_splat_index];
		const float2 splat_mean = means2d[target_splat_index];
		uint2 bounds_min, bounds_max;
		getRect(splat_mean, splat_radius, bounds_min, bounds_max, grid_size);

		// Mark hit indices.
		if (splat_radius > 0 && group_index.x >= bounds_min.x && group_index.x < bounds_max.x && group_index.y >=
		    bounds_min.y &&
		    group_index.y < bounds_max.y)
			hit_indices[stride] = target_splat_index;
	}

	// Sync hit-checking.
	block.sync();

	// Exit if this pixel is not in bounds.
	if (!pixel_in_bounds)
		return;

	// Phase 2: Cluster splats in this ingest.

	// Clustering data for this pixel.
	float pixel_cluster_data[CLUSTER_DATA_LENGTH] = {};
	int uninitialized_cluster_index;

	// Initialize or read from cluster data.
	if (starting_splat_index == 0) {
		// Set transmittance to 1.0 for all clusters.
		for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
			pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, ALPHA_INDEX)] = 1.0f;
		}
	} else {
		for (int i = 0; i < NUMBER_OF_CLUSTERS; ++i) {
			pixel_cluster_data[DATA_IN_CLUSTER(i, DEPTH_INDEX)] = cluster_depth[i * width * height + pixel_index];
			pixel_cluster_data[DATA_IN_CLUSTER(i, SPLAT_COUNT_INDEX)] = static_cast<float>(cluster_splat_count[
				i * width * height + pixel_index]);
			pixel_cluster_data[DATA_IN_CLUSTER(i, ALPHA_SUM_INDEX)] = cluster_alpha_sum[
				i * width * height + pixel_index];
			pixel_cluster_data[DATA_IN_CLUSTER(i, ALPHA_INDEX)] = cluster_alpha[i * width * height + pixel_index];
			pixel_cluster_data[DATA_IN_CLUSTER(i, PREMULTIPLIED_R_INDEX)] = cluster_premultiplied_r[
				i * width * height + pixel_index];
			pixel_cluster_data[DATA_IN_CLUSTER(i, PREMULTIPLIED_G_INDEX)] = cluster_premultiplied_g[
				i * width * height + pixel_index];
			pixel_cluster_data[DATA_IN_CLUSTER(i, PREMULTIPLIED_B_INDEX)] = cluster_premultiplied_b[
				i * width * height + pixel_index];
		}
		uninitialized_cluster_index = cluster_uninitialized_cluster_index[pixel_index];
	}

	// Iterate over hit splats if this pixel is in bounds.
	for (int sample_index = 0; sample_index < INGEST_SIZE; ++sample_index) {
		// Get splat index.
		const int sample_splat_index = hit_indices[sample_index];

		// Skip index if it was not hit.
		if (sample_splat_index == -1)
			continue;

		// Compute splat alpha.

		// Resample using conic matrix (cf. "Surface
		// Splatting" by Zwicker et al., 2001)
		const float2 sample_coordinate = means2d[sample_splat_index];
		const float2 d = {
			sample_coordinate.x - static_cast<float>(pixel_coordinate.x),
			sample_coordinate.y - static_cast<float>(pixel_coordinate.y)
		};
		const float4 con_o = conic_opacity[sample_splat_index];
		const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
		if (power > 0.0f)
			continue;

		// Eq. (2) from 3D Gaussian splatting paper.
		// Obtain alpha by multiplying with Gaussian opacity
		// and its exponential falloff from mean.
		// Avoid numerical instabilities (see paper appendix).
		const float sample_alpha = min(0.99f, con_o.w * exp(power));
		if (sample_alpha < 1.0f / 255.0f)
			continue;

		// Collect the color
		const float sample_r = features[sample_splat_index * CHANNELS + 0];
		const float sample_g = features[sample_splat_index * CHANNELS + 1];
		const float sample_b = features[sample_splat_index * CHANNELS + 2];

		// Collect the depth.
		const float sample_depth = depths[sample_splat_index];

		// Do initial cluster guesses or argmin to find cluster.
		int target_cluster_index = 0;

		// Pick cluster to use. Initialize empty ones or find the argmin.
		if (uninitialized_cluster_index < NUMBER_OF_CLUSTERS) {
			// Start with the next open cluster index.
			target_cluster_index = uninitialized_cluster_index;

			// Increment the uninitialized cluster if this is the first sample since we will for sure use it.
			if (target_cluster_index == 0) {
				uninitialized_cluster_index++;
			}

			// Check initialized clusters for an exact match (if this wasn't the first cluster).
			for (int cluster_index = 0; cluster_index < target_cluster_index; ++cluster_index) {
				// Use it if found.
				if (pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, DEPTH_INDEX)] == sample_depth) {
					target_cluster_index = cluster_index;
					break;
				}

				// If we didn't find a match, increment the uninitialized cluster index for next time.
				if (cluster_index == target_cluster_index - 1) {
					uninitialized_cluster_index++;
				}
			}
		} else {
			float current_closest_depth_distance = pixel_cluster_data[DATA_IN_CLUSTER(0, DEPTH_INDEX)];
			// If all clusters are initialized, find the closest cluster.
			for (int cluster_index = 1; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
				// Replace the target index if it's closer.
				const float distance_to_cluster = fabsf(
					pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, DEPTH_INDEX)] - sample_depth);
				if (distance_to_cluster < current_closest_depth_distance) {
					current_closest_depth_distance = distance_to_cluster;
					target_cluster_index = cluster_index;
				}
			}
		}

		// Update cluster information (note: cluster alpha is computed as 1 - transmittance).
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, SPLAT_COUNT_INDEX)]++;
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, ALPHA_SUM_INDEX)] += sample_alpha;
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, ALPHA_INDEX)] *= 1 - sample_alpha;
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, PREMULTIPLIED_R_INDEX)] += sample_alpha * sample_r;
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, PREMULTIPLIED_G_INDEX)] += sample_alpha * sample_g;
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, PREMULTIPLIED_B_INDEX)] += sample_alpha * sample_b;

		// Update cluster mean.
		const float current_mean = pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, DEPTH_INDEX)];
		pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, DEPTH_INDEX)] =
				current_mean + (sample_depth - current_mean) / pixel_cluster_data[DATA_IN_CLUSTER(
					target_cluster_index, SPLAT_COUNT_INDEX)];

		// Mark this splat as contributing.
		contributing_splat_count++;
	}

	// Write to cluster data.
	n_contrib[pixel_index] = contributing_splat_count;

	for (int i = 0; i < NUMBER_OF_CLUSTERS; ++i) {
		cluster_depth[i * width * height + pixel_index] = pixel_cluster_data[DATA_IN_CLUSTER(i, DEPTH_INDEX)];
		cluster_splat_count[i * width * height + pixel_index] = static_cast<int>(pixel_cluster_data[
			DATA_IN_CLUSTER(i, SPLAT_COUNT_INDEX)]);
		cluster_alpha_sum[i * width * height + pixel_index] = pixel_cluster_data[DATA_IN_CLUSTER(i, ALPHA_SUM_INDEX)];
		cluster_alpha[i * width * height + pixel_index] = pixel_cluster_data[DATA_IN_CLUSTER(i, ALPHA_INDEX)];
		cluster_premultiplied_r[i * width * height + pixel_index] = pixel_cluster_data[DATA_IN_CLUSTER(
			i, PREMULTIPLIED_R_INDEX)];
		cluster_premultiplied_g[i * width * height + pixel_index] = pixel_cluster_data[DATA_IN_CLUSTER(
			i, PREMULTIPLIED_G_INDEX)];
		cluster_premultiplied_b[i * width * height + pixel_index] = pixel_cluster_data[DATA_IN_CLUSTER(
			i, PREMULTIPLIED_B_INDEX)];
	}
	cluster_uninitialized_cluster_index[pixel_index] = uninitialized_cluster_index;
}

template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE)
cluster_renderCUDA(
	const int width,
	const int height,
	const float * __restrict__ cluster_depth,
	const int * __restrict__ cluster_splat_count,
	const float * __restrict__ cluster_alpha_sum,
	const float * __restrict__ cluster_alpha,
	const float * __restrict__ cluster_premultiplied_r,
	const float * __restrict__ cluster_premultiplied_g,
	const float * __restrict__ cluster_premultiplied_b,
	const float * __restrict__ bg_color,
	float * __restrict__ final_transmittance,
	float * __restrict__ invdepth,
	float * __restrict__ out_color
) {
	// Gather thread information.
	auto block = cg::this_thread_block();
	const auto group_index = block.group_index();
	const auto thread_index = block.thread_index();

	// Gather pixel information.
	const uint2 minimum_pixel_coordinate = {group_index.x * BLOCK_X, group_index.y * BLOCK_Y};
	const uint2 pixel_coordinate = {
		minimum_pixel_coordinate.x + thread_index.x, minimum_pixel_coordinate.y + thread_index.y
	};
	const uint32_t pixel_index = width * pixel_coordinate.y + pixel_coordinate.x;

	// Compute if this thread is associated with a visible pixel.
	const bool pixel_in_bounds = pixel_coordinate.x < width && pixel_coordinate.y < height;

	// Exit if this pixel is not in bounds.
	if (!pixel_in_bounds)
		return;

	// Fetch this pixel's cluster data.
	float pixel_cluster_data[CLUSTER_DATA_LENGTH];
	for (int i = 0; i < NUMBER_OF_CLUSTERS; ++i) {
		pixel_cluster_data[DATA_IN_CLUSTER(i, DEPTH_INDEX)] = cluster_depth[i * width * height + pixel_index];
		pixel_cluster_data[DATA_IN_CLUSTER(i, SPLAT_COUNT_INDEX)] = cluster_splat_count[
			i * width * height + pixel_index];
		pixel_cluster_data[DATA_IN_CLUSTER(i, ALPHA_SUM_INDEX)] = cluster_alpha_sum[
			i * width * height + pixel_index];
		pixel_cluster_data[DATA_IN_CLUSTER(i, ALPHA_INDEX)] = cluster_alpha[i * width * height + pixel_index];
		pixel_cluster_data[DATA_IN_CLUSTER(i, PREMULTIPLIED_R_INDEX)] = cluster_premultiplied_r[
			i * width * height + pixel_index];
		pixel_cluster_data[DATA_IN_CLUSTER(i, PREMULTIPLIED_G_INDEX)] = cluster_premultiplied_g[
			i * width * height + pixel_index];
		pixel_cluster_data[DATA_IN_CLUSTER(i, PREMULTIPLIED_B_INDEX)] = cluster_premultiplied_b[
			i * width * height + pixel_index];
	}

	// For each cluster, convert transmittance to alpha and compute the final RGB values.
	for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
		pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, ALPHA_INDEX)] = 1 - pixel_cluster_data[DATA_IN_CLUSTER(
			                                                                  cluster_index, ALPHA_INDEX)];
		pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, PREMULTIPLIED_R_INDEX)] /= pixel_cluster_data[DATA_IN_CLUSTER(
			cluster_index, ALPHA_SUM_INDEX)];
		pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, PREMULTIPLIED_G_INDEX)] /= pixel_cluster_data[DATA_IN_CLUSTER(
			cluster_index, ALPHA_SUM_INDEX)];
		pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, PREMULTIPLIED_B_INDEX)] /= pixel_cluster_data[DATA_IN_CLUSTER(
			cluster_index, ALPHA_SUM_INDEX)];
	}

	// Initialize rendering variables.
	float pixel_transmittance = 1.0f;
	float expected_invdepth = 0.0f;
	float pixel_color[CHANNELS] = {};
	float last_minimum_depth = 0.0f;

	// Compute every cluster.
	for (int i = 0; i < NUMBER_OF_CLUSTERS; ++i) {
		// Exit if transmittance is too low.
		if (pixel_transmittance <= MINIMUM_TRANSMITTANCE)
			break;

		// Find the next closest cluster.
		int target_cluster_index = 0;
		float current_minimum_depth = FLT_MAX;
		for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
			const float this_cluster_depth = pixel_cluster_data[DATA_IN_CLUSTER(cluster_index, DEPTH_INDEX)];
			if (this_cluster_depth > last_minimum_depth && this_cluster_depth < current_minimum_depth) {
				current_minimum_depth = this_cluster_depth;
				target_cluster_index = cluster_index;
			}
		}
		// Update the last minimum depth after finding the next cluster.
		last_minimum_depth = current_minimum_depth;

		// Get cluster data.
		const float cluster_alpha_value = pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, ALPHA_INDEX)];
		const float cluster_r = pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, PREMULTIPLIED_R_INDEX)];
		const float cluster_g = pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, PREMULTIPLIED_G_INDEX)];
		const float cluster_b = pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, PREMULTIPLIED_B_INDEX)];

		// Contribute the cluster to the final output color.
		pixel_color[0] += cluster_alpha_value * cluster_r * pixel_transmittance;
		pixel_color[1] += cluster_alpha_value * cluster_g * pixel_transmittance;
		pixel_color[2] += cluster_alpha_value * cluster_b * pixel_transmittance;

		// Update invdepth.
		if (invdepth)
			expected_invdepth += 1 / pixel_cluster_data[DATA_IN_CLUSTER(target_cluster_index, DEPTH_INDEX)] * cluster_alpha_value *
					pixel_transmittance;

		// Update the transmittance.
		pixel_transmittance *= 1 - min(1.0f, cluster_alpha_value);
	}

	// Write to output buffers.
	final_transmittance[pixel_index] = pixel_transmittance;
	if (invdepth)
		invdepth[pixel_index] = expected_invdepth;

	// Write to output buffer and apply background color.
	for (int channel = 0; channel < CHANNELS; ++channel) {
		out_color[channel * height * width + pixel_index] =
				pixel_color[channel] + pixel_transmittance * bg_color[channel];
	}
}

void FORWARD::cluster_render(dim3 grid_size, dim3 block_size, const int width, const int height,
                             const float *cluster_depth, const int *cluster_splat_count, const float
                             *cluster_alpha_sum, const float *cluster_alpha, const float *cluster_premultiplied_r,
                             const float *cluster_premultiplied_g, const float *
                             cluster_premultiplied_b, const float *bg_color, float *final_transmittance,
                             float *invdepth, float *out_color) {
	cluster_renderCUDA<NUM_CHANNELS> <<<grid_size, block_size>>>(width, height, cluster_depth, cluster_splat_count,
	                                                             cluster_alpha_sum, cluster_alpha,
	                                                             cluster_premultiplied_r, cluster_premultiplied_g,
	                                                             cluster_premultiplied_b, bg_color,
	                                                             final_transmittance, invdepth, out_color);
}

void FORWARD::skm_cluster_pass(dim3 grid_size, dim3 block_size, const int starting_splat_index, const int P, const int
                               width,
                               const int height, const int *radii, const float2 *means_2d, const float4 *conic_opacity,
                               const float *depths, const float *features, uint32_t *n_contrib, float *cluster_depth,
                               int *cluster_splat_count, float
                               *cluster_alpha_sum, float *cluster_alpha, float *cluster_premultiplied_r,
                               float *cluster_premultiplied_g, float *
                               cluster_premultiplied_b, int *cluster_uninitialized_cluster_index) {
	skm_cluster_passCUDA<NUM_CHANNELS> <<<grid_size, block_size>>>(starting_splat_index, P, width, height, grid_size,
	                                                               radii,
	                                                               means_2d, conic_opacity, depths, features, n_contrib,
	                                                               cluster_depth, cluster_splat_count,
	                                                               cluster_alpha_sum, cluster_alpha,
	                                                               cluster_premultiplied_r, cluster_premultiplied_g,
	                                                               cluster_premultiplied_b,
	                                                               cluster_uninitialized_cluster_index);
}

void FORWARD::preprocess(int P, int D, int M,
                         const float *means3D,
                         const glm::vec3 *scales,
                         const float scale_modifier,
                         const glm::vec4 *rotations,
                         const float *opacities,
                         const float *shs,
                         bool *clamped,
                         const float *cov3D_precomp,
                         const float *colors_precomp,
                         const float *viewmatrix,
                         const float *projmatrix,
                         const glm::vec3 *cam_pos,
                         const int W, int H,
                         const float focal_x, float focal_y,
                         const float tan_fovx, float tan_fovy,
                         int *radii,
                         float2 *means2D,
                         float *depths,
                         float *cov3Ds,
                         float *rgb,
                         float4 *conic_opacity,
                         const dim3 grid,
                         uint32_t *tiles_touched,
                         bool prefiltered,
                         bool antialiasing) {
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
