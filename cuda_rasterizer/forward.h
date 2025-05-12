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
#define NUMBER_OF_CLUSTERS 12
// FIXME: Assumes channels is always 3.
#define NUMBER_OF_DATA_POINTS 7
#define DEPTH_INDEX 0
#define SPLAT_COUNT_INDEX 1
#define ALPHA_SUM_INDEX 2
#define TRANSMITTANCE_INDEX 3
#define PREMULTIPLIED_R_INDEX 4
#define PREMULTIPLIED_G_INDEX 5
#define PREMULTIPLIED_B_INDEX 6
#define MINIMUM_TRANSMITTANCE 0.0001f
#define DATA_AT(INDEX, DATA) (INDEX * NUMBER_OF_DATA_POINTS + DATA)

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
	 * One-kernel rasterizer using SKM as the clustering algorithm.
	 * 
	 * @param P Total number of Gaussian points.
	 * @param grid_size Number of blocks to launch (number of tiles in the image).
	 * @param block_size Number of threads per block (size of a tile).
	 * @param radii
	 * @param means_2d Array of 2D coordinates of each Gaussian.
	 * @param conic_opacity
	 * @param depths
	 * @param width
	 * @param height
	 * @param colors_precomp
	 * @param rgb
	 */
	void skm_render(int P, dim3 grid_size, dim3 block_size, int *radii, const float2 *means_2d, const float4 *conic_opacity, const float *
	                depths, int width, int height, const float *colors_precomp, const float *rgb);

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
