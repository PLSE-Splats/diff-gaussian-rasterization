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
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
skm_renderCUDA(const int P, const dim3 grid_size, const int batches_count, int *radii,
               const float2 *means_2d, const float4 *conic_opacity, const float *depths, const int width,
               const int height, const float * __restrict__ features, float * __restrict__ invdepth) {
	// Phase 0: Set up data.

	// 0.1. Gather thread information.
	auto block = cg::this_thread_block();
	const auto group_index = block.group_index();
	const auto thread_index = block.thread_index();
	const auto thread_rank = block.thread_rank();

	// FIXME: This could be optimized and not declared for out-of-bound pixels.
	// 0.2. Initialize helper variables for rendering.
	float pixel_transmittance = 1.0f;
	uint32_t contributing_gaussians_count = 0;
	uint32_t last_contributing_count = 0;
	float colors[CHANNELS] = {};
	float expected_invdepth = 0.0f;
	float cluster_data[NUMBER_OF_CLUSTERS * NUMBER_OF_DATA_POINTS] = {};
	bool initial_guesses_found = false;
	bool done = false;

	// 0.2.1. Set transmittance to 1.0 for all clusters.
	for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
		cluster_data[DATA_AT(cluster_index, TRANSMITTANCE_INDEX)] = 1.0f;
	}

	// 0.3. Declare shared data structures.
	__shared__ int collected_index[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// Phase 1: As a block.

	// Loop over batches of Gaussian data.
	for (int batch_index = 0; batch_index < batches_count; ++batch_index) {
		// 1.1. Finish rendering if the block is done.
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;

		// 1.2. Reset collected_index.
		collected_index[thread_rank] = -1;

		// 1.3. Fetch batch of Gaussian data from global memory to shared memory.
		const int target_gaussian_index = batch_index * BLOCK_SIZE + thread_rank;
		if (target_gaussian_index < P) {
			// 1.4 Check if this Gaussian intersects with the tile.
			const float2 xy = means_2d[target_gaussian_index];
			uint2 bounds_min, bounds_max;
			getRect(xy, radii[target_gaussian_index], bounds_min, bounds_max, grid_size);

			if (group_index.x >= bounds_min.x && group_index.x < bounds_max.x &&
			    group_index.y >= bounds_min.y && group_index.y < bounds_max.y) {
				// 1.5. If it does, copy the Gaussian data to shared memory.
				collected_index[thread_rank] = target_gaussian_index;
				collected_xy[thread_rank] = xy;
				collected_conic_opacity[thread_rank] = conic_opacity[target_gaussian_index];
				collected_depth[thread_rank] = depths[target_gaussian_index];
			}
		}

		// 1.4 Sync collection.
		block.sync();

		// Phase 2: Per pixel

		// 2.1. Skip if this pixel is done with rendering.
		if (done)
			continue;

		// 2.1. Gather pixel information.
		const uint2 minimum_pixel_coordinate = {group_index.x * BLOCK_X, group_index.y * BLOCK_Y};
		const uint2 pixel_coordinate = {
			minimum_pixel_coordinate.x + thread_index.x, minimum_pixel_coordinate.y + thread_index.y
		};

		// 2.2. Mark pixel as done with rendering if it is out of bounds.
		if (pixel_coordinate.x >= width || pixel_coordinate.y >= height) {
			done = true;
			continue;
		}

		// 2.4. Iterate over Gaussian batch.
		for (int sample_index = 0; sample_index < BLOCK_SIZE; ++sample_index) {
			// 2.5.1. Skip if data was not collected.
			if (collected_index[sample_index] == -1)
				continue;

			// FIXME: Shouldn't this happen after all the data collection since we can cancel out before actually contributing?
			// 2.5.2. Mark this Gaussian as a contributor.
			contributing_gaussians_count++;

			// 2.5.2. Compute the alpha.

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			const float2 sample_coordinate = collected_xy[sample_index];
			const float2 d = {
				sample_coordinate.x - static_cast<float>(pixel_coordinate.x),
				sample_coordinate.y - static_cast<float>(pixel_coordinate.y)
			};
			const float4 con_o = collected_conic_opacity[sample_index];
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

			// 2.5.3. End rendering if transmittance is too low.
			if (pixel_transmittance * (1 - sample_alpha) < 0.0001f) {
				done = true;
				continue;
			}

			// 2.5.4. Collect the color
			const float sample_r = features[collected_index[sample_index] * CHANNELS + 0];
			const float sample_g = features[collected_index[sample_index] * CHANNELS + 1];
			const float sample_b = features[collected_index[sample_index] * CHANNELS + 2];

			// 2.5.5. Collect the depth.
			const float sample_depth = collected_depth[sample_index];

			// 2.5.6. Do initial cluster guesses or argmin to find cluster.
			int target_cluster_index = 0;

			if (!initial_guesses_found) {
				// Loop through each non-zero cluster and check for an exact match.
				for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
					// Use the cluster if it's exactly the same depth.
					if (cluster_data[DATA_AT(cluster_index, DEPTH_INDEX)] == sample_depth) {
						target_cluster_index = cluster_index;
						break;
					}

					// Skip if cluster is not empty.
					if (cluster_data[DATA_AT(cluster_index, DEPTH_INDEX)] != 0.0f)
						continue;

					// Use the cluster if it's empty.
					target_cluster_index = cluster_index;

					// If all clusters have been initialized, stop looking.
					if (cluster_index == NUMBER_OF_CLUSTERS - 1)
						initial_guesses_found = true;

					// We've found a cluster if we got here, so stop looking.
					break;
				}
			} else {
				float current_closest_depth_distance = FLT_MAX;
				// If all clusters are initialized, find the closest cluster.
				for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS; ++cluster_index) {
					// Replace the target index if it's closer.
					const float distance_to_cluster = fabsf(
						cluster_data[DATA_AT(cluster_index, DEPTH_INDEX)] - sample_depth);
					if (distance_to_cluster < current_closest_depth_distance) {
						current_closest_depth_distance = distance_to_cluster;
						target_cluster_index = cluster_index;
					}
				}
			}

			// 2.5.7. Update cluster information.
			cluster_data[DATA_AT(target_cluster_index, SPLAT_COUNT_INDEX)]++;
			cluster_data[DATA_AT(target_cluster_index, ALPHA_SUM_INDEX)] += sample_alpha;
			cluster_data[DATA_AT(target_cluster_index, TRANSMITTANCE_INDEX)] *= 1 - sample_alpha;
			cluster_data[DATA_AT(target_cluster_index, PREMULTIPLIED_R_INDEX)] += sample_alpha * sample_r;
			cluster_data[DATA_AT(target_cluster_index, PREMULTIPLIED_G_INDEX)] += sample_alpha * sample_g;
			cluster_data[DATA_AT(target_cluster_index, PREMULTIPLIED_B_INDEX)] += sample_alpha * sample_b;

			// Update cluster mean.
			const float current_mean = cluster_data[DATA_AT(target_cluster_index, DEPTH_INDEX)];
			cluster_data[DATA_AT(target_cluster_index, DEPTH_INDEX)] = current_mean + (sample_depth - current_mean) /
			                                                           cluster_data[DATA_AT(
				                                                           target_cluster_index, SPLAT_COUNT_INDEX)];

			// Update invdepth.
			if (invdepth)
				expected_invdepth += 1 / collected_depth[sample_index] * sample_alpha * pixel_transmittance;

			pixel_transmittance *= 1 - sample_alpha;

			// Update last contributing count.
			last_contributing_count = contributing_gaussians_count;
		}

		// 2.2. If there are still batches to process, go back to 1.2.
	}

	// 2.3. Once all batches are done, compute final transmittance and color for each cluster.

	// Phase 3: Alpha composite the clusters.

	// 3.1. Iterate over each cluster.

	// 3.1.1. Find the closest cluster.

	// 3.1.2. Do any shortcut exits for compositing.

	// 3.1.3. Contribute the cluster to the final output color.

	// 3.1.4. Update the transmittance.

	// 3.2. Write to output buffer and apply background color.
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths,
		depth);
}

void FORWARD::skm_render(const int P, const dim3 grid_size, const dim3 block_size, int *radii,
                         const float2 *means_2d, const float4 *conic_opacity, const float *depths, const int width,
                         const int height, const float *colors_precomp, const float *rgb, float *depth) {
	// Compute number of batches needed to process all Gaussian data.
	const int batches_count = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const float *features = colors_precomp != nullptr ? colors_precomp : rgb;
	skm_renderCUDA<NUM_CHANNELS> <<<grid_size, block_size>>>(P, grid_size, batches_count, radii, means_2d,
	                                                         conic_opacity,
	                                                         depths, width, height, features, depth);
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
