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

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, int D, int M, const float* orig_points,
                const glm::vec3* scales, const float scale_modifier,
                const glm::vec4* rotations, const float* opacities,
                const float* shs, bool* clamped, const float* cov3D_precomp,
                const float* colors_precomp, const float* viewmatrix,
                const float* projmatrix, const glm::vec3* cam_pos, const int W,
                int H, const float focal_x, float focal_y, const float tan_fovx,
                float tan_fovy, int* radii, float2* points_xy_image,
                float* depths, float* cov3Ds, float* colors,
                float4* conic_opacity, const dim3 grid, uint32_t* tiles_touched,
                bool prefiltered, bool antialiasing);

/**
 * Seed cluster depths per pixel.
 *
 * @param grid_size Number of blocks to launch (number of tiles in the image).
 * @param block_size Number of threads per block (size of a tile).
 * @param width Image width.
 * @param height Image height.
 * @param splat_ids List of splat indices per tile.
 * @param splat_id_ranges Index ranges in splat list for each tile.
 * @param means_2d Input 2D means of each splat.
 * @param conic_opacities Input conic opacity of each splat.
 * @param depths Input depth of each splat.
 * @param cluster_depth_seeds Output cluster depth seeds per pixel (organized in
 * sets of cluster pairs per pixel).
 */
void seed_cluster_depths(dim3 grid_size, dim3 block_size, int width, int height,
                         const uint32_t* splat_ids,
                         const uint2* splat_id_ranges, const float2* means_2d,
                         const float4* conic_opacities, const float* depths,
                         __half2* cluster_depth_seeds);

/**
 * Cluster splats per pixel.
 *
 * @param grid_size Number of blocks to launch (number of tiles in the image).
 * @param block_size Number of threads per block (size of a tile).
 * @param width Image width.
 * @param height Image height.
 * @param splat_ids List of splat indices per tile.
 * @param splat_id_ranges Index ranges in splat list for each tile.
 * @param means_2d Input 2D means of each splat.
 * @param conic_opacities Input conic opacity of each splat.
 * @param depths Input depth of each splat.
 * @param features Input colors of each splat.
 * @param bg_color Background color.
 * @param cluster_depth_seeds Depth seeds for each cluster (in cluster pairs per
 * pixel order).
 * @param n_contributions Output number of splats contributing to each pixel.
 * @param inv_depth Output inverse depth per pixel.
 * @param final_transmittance Output final transmittance oper pixel.
 * @param out_color Output color per pixel.
 */
void cluster_render(dim3 grid_size, dim3 block_size, int width, int height,
                    const uint32_t* splat_ids, const uint2* splat_id_ranges,
                    const float2* means_2d, const float4* conic_opacities,
                    const float* depths, const float* features,
                    const float* bg_color, const __half2* cluster_depth_seeds,
                    uint32_t* n_contributions, float* inv_depth,
                    float* final_transmittance, float* out_color);
}  // namespace FORWARD

#endif
