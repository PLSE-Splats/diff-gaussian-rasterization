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
#define NUMBER_OF_CLUSTERS 8
#define MINIMUM_TRANSMITTANCE 0.0001f
#define DEBUG_PIXEL -1 // 1051640

// Half constants.
#define CUDART_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7BFFU)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)

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
	 * @param splat_ids List of splat indices per tile.
	 * @param splat_id_ranges Index ranges in splat list for each tile.
	 * @param radii Input radii of each splat.
	 * @param means_2d Input 2D means of each splat.
	 * @param conic_opacities Input conic opacity of each splat.
	 * @param depths Input depth of each splat.
	 * @param features Input features of each splat (RGB).
	 * @param n_contributions Output number of splats contributing to each pixel.
	 * @param cluster_depths Output cluster depth.
	 * @param cluster_alphas Output cluster alpha.
	 * @param cluster_reds Output cluster premultiplied red channel.
	 * @param cluster_greens Output cluster premultiplied green channel.
	 * @param cluster_blues Output cluster premultiplied blue channel.
	 */
	void cluster_render(
		dim3 grid_size,
		dim3 block_size,
		int width,
		int height,
		const uint32_t *splat_ids,
		const ushort2 *splat_id_ranges,
		const float2 *means_2d,
		const float4 *conic_opacities,
		const float *depths,
		const float *features,
		uint32_t *n_contributions,
		const float* bg_color,
		float* final_transmittance,
		float* invdepth,
		float *out_color
	);
}


#endif
