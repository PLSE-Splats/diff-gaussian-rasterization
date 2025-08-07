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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// SKM parameters.
#define NUMBER_OF_CLUSTERS 8
#define INGEST_SIZE 1750
// FIXME: Assumes channels is always 3.
#define NUMBER_OF_DATA_POINTS 7

#define DEPTH_INDEX 0
#define SPLAT_COUNT_INDEX 1
#define ALPHA_SUM_INDEX 2
#define ALPHA_INDEX 3
#define PREMULTIPLIED_R_INDEX 4
#define PREMULTIPLIED_G_INDEX 5
#define PREMULTIPLIED_B_INDEX 6

#define UNINITIALIZED_CLUSTER_INDEX_INDEX NUMBER_OF_CLUSTERS * NUMBER_OF_DATA_POINTS
#define CLUSTER_DATA_LENGTH (NUMBER_OF_CLUSTERS * NUMBER_OF_DATA_POINTS + 1) // +1 for uninitialized cluster index.
#define CLUSTERS_AT_PIXEL(PIXEL_INDEX) PIXEL_INDEX * CLUSTER_DATA_LENGTH
#define DATA_IN_CLUSTER(CLUSTER_INDEX, DATA) (CLUSTER_INDEX * NUMBER_OF_DATA_POINTS + DATA)
#define MINIMUM_TRANSMITTANCE 0.0001f

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	/**
	 * Perform clustering of Gaussian points using the SKM algorithm per pixel.
	 * 
	 * @param grid_size Number of blocks to launch (number of tiles in the image).
	 * @param block_size Number of threads per block (size of a tile).
	 * @param starting_splat_index
	 * @param P Total number of Gaussian points.
	 * @param width Image width.
	 * @param height Image height.
	 * @param radii Array of radii for each Gaussian.
	 * @param means_2d Array of 2D coordinates of each Gaussian.
	 * @param conic_opacity Array of conic opacity values for each Gaussian.
	 * @param depths Array of depth values for each Gaussian.
	 * @param features Color features for each Gaussian.
	 * @param n_contrib Array of number of gaussians that contribute to each pixel.
	 * @param cluster_data Cluster data for each pixel in the image.
	 */
	void skm_cluster(
		dim3 grid_size,
		dim3 block_size,
		int starting_splat_index,
		int P,
		int width,
		int height,
		const int *radii,
		const float2 *means_2d,
		const float4 *conic_opacity,
		const float *depths,
		const float *features,
		uint32_t *n_contrib, float *cluster_data
	);

	/**
	 * Alpha composite the clusters to render the image.
	 * 
	 * @param grid_size Number of blocks to launch (number of tiles in the image).
	 * @param block_size Number of threads per block (size of a tile).
	 * @param width Image width.
	 * @param height Image height.
	 * @param cluster_data Cluster data for each pixel in the image.
	 * @param bg_color Background color for the image.
	 * @param final_transmittance Final transmittance values for each pixel in the image.
	 * @param invdepth
	 * @param out_color Output color for each pixel in the image.
	 */
	void cluster_render(
		dim3 grid_size,
		dim3 block_size,
		int width,
		int height,
		const float *cluster_data,
		const float *bg_color,
		float *final_transmittance,
		float *invdepth, float *out_color
	);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth);
}


#endif
