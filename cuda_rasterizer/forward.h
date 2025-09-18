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
#include "cuda_fp16.h"

// Clustering parameters.
#define NUMBER_OF_CLUSTERS 4
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
	 * Cluster splats per pixel.
	 * 
	 * @param grid_size Number of blocks to launch (number of tiles in the image).
	 * @param block_size Number of threads per block (size of a tile).
	 * @param width Image width.
	 * @param height Image height.
	 * @param radii Input radii of each splat.
	 * @param means_2d Input 2D means of each splat.
	 * @param conic_opacity Input conic opacity of each splat.
	 * @param depths Input depth of each splat.
	 * @param features Input features of each splat (RGB).
	 * @param n_contrib Output number of splats contributing to each pixel.
	 * @param cluster_depth Output cluster depth.
	 * @param cluster_alpha Output cluster alpha.
	 * @param cluster_r Output cluster premultiplied red channel.
	 * @param cluster_g Output cluster premultiplied green channel.
	 * @param cluster_b Output cluster premultiplied blue channel.
	 */
	void cluster(
		dim3 grid_size,
		dim3 block_size,
		int width,
		int height,
		const int *radii,
		const float2 *means_2d,
		const float4 *conic_opacity,
		const float *depths,
		const float *features,
		uint32_t *n_contrib,
		__half *cluster_depth,
		__half *cluster_alpha,
		__half *cluster_r,
		__half *cluster_g,
		__half *cluster_b
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
