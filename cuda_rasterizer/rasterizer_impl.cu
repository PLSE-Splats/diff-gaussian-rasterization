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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Populate tile-splat key-value buffers for tile-group-by sorting.
// Run once per Gaussian (1:P mapping).
__global__ void duplicateWithKeys(
	const int P,
	const float2* points_xy,
	const uint32_t *tiles_per_splat_offsets,
	uint16_t *tile_id_unsorted,
	uint32_t *splat_id_unsorted,
	const int *splat_radii,
	const dim3 tile_grid) {
	const auto splat_id = cg::this_grid().thread_rank();

	// Skip out-of-bounds threads.
	if (splat_id >= P)
		return;

	// Generate no key/value pair for invisible splats.
	if (splat_radii[splat_id] > 0) {
		// Find this splat's offset in buffer for writing.
		uint32_t off = splat_id == 0 ? 0 : tiles_per_splat_offsets[splat_id - 1];
		uint2 rect_min, rect_max;

		// Compute bounding rect of splat in tile space.
		getRect(points_xy[splat_id], splat_radii[splat_id], rect_min, rect_max, tile_grid);

		// For each tile that the bounding rect overlaps, write its tile ID and splat ID to the buffers.
		for (unsigned int y = rect_min.y; y < rect_max.y; y++) {
			for (unsigned int x = rect_min.x; x < rect_max.x; x++) {
				tile_id_unsorted[off] = y * tile_grid.x + x;
				splat_id_unsorted[off] = splat_id;
				off++;
			}
		}
	}
}

// Calculate the start and end of each tile's range of splats.
__global__ void identifyTileRanges(const int L, const uint16_t* sorted_tile_ids, ushort2* ranges)
{
	const auto tile_splat_pair_index = cg::this_grid().thread_rank();
	
	// Skip out-of-bounds pairs.
	if (tile_splat_pair_index >= L)
		return;

	// Read tile ID. Update start/end of tile range if at limit.
	const uint16_t tile_id = sorted_tile_ids[tile_splat_pair_index];
	if (tile_splat_pair_index == 0)
		ranges[tile_id].x = 0;
	else
	{
		uint16_t previous_tile = sorted_tile_ids[tile_splat_pair_index - 1];
		if (tile_id != previous_tile)
		{
			ranges[previous_tile].y = tile_splat_pair_index;
			ranges[tile_id].x = tile_splat_pair_index;
		}
	}
	if (tile_splat_pair_index == L - 1)
		ranges[tile_id].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char *img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// Allocate space for unsorted keys/values, sorted keys/values.
	uint16_t *d_unsorted_tile_ids, *d_sorted_tile_ids;
	CHECK_CUDA(cudaMalloc(&d_unsorted_tile_ids, num_rendered * sizeof(uint16_t)), debug)
	CHECK_CUDA(cudaMalloc(&d_sorted_tile_ids, num_rendered * sizeof(uint16_t)), debug)

	uint32_t *d_unsorted_splat_ids, *d_sorted_splat_ids;
	CHECK_CUDA(cudaMalloc(&d_unsorted_splat_ids, num_rendered * sizeof(uint32_t)), debug)
	CHECK_CUDA(cudaMalloc(&d_sorted_splat_ids, num_rendered * sizeof(uint32_t)), debug)

	// Populate unsorted key/value buffers with tile IDs and splat IDs.
	duplicateWithKeys << <(P + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
		P,
		geomState.means2D,
		geomState.point_offsets,
		d_unsorted_tile_ids,
		d_unsorted_splat_ids,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	// Sort by tile ID.
	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	CHECK_CUDA(
		cub::DeviceRadixSort::SortPairs(
			d_temp_storage, temp_storage_bytes,
			d_unsorted_tile_ids, d_sorted_tile_ids,
			d_unsorted_splat_ids, d_sorted_splat_ids,
			num_rendered),
		debug);

	CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes), debug);

	CHECK_CUDA(
		cub::DeviceRadixSort::SortPairs(
			d_temp_storage, temp_storage_bytes,
			d_unsorted_tile_ids, d_sorted_tile_ids,
			d_unsorted_splat_ids, d_sorted_splat_ids,
			num_rendered),
		debug);

	// Identify start and end of per-tile workloads in sorted list
	ushort2 *d_tile_ranges;
	CHECK_CUDA(cudaMalloc(&d_tile_ranges, tile_grid.x * tile_grid.y * sizeof(ushort2)), debug)
	CHECK_CUDA(cudaMemset(d_tile_ranges, 0, tile_grid.x * tile_grid.y * sizeof(ushort2)), debug);

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			d_sorted_tile_ids,
			d_tile_ranges);
	CHECK_CUDA(, debug)

	// Allocate clustering frame buffer.
	__half *d_cluster_depth;
	__half *d_cluster_alpha;
	__half *d_cluster_r;
	__half *d_cluster_g;
	__half *d_cluster_b;

	CHECK_CUDA(cudaMalloc(&d_cluster_depth, width*height*NUMBER_OF_CLUSTERS*sizeof(__half)), debug);
	CHECK_CUDA(cudaMalloc(&d_cluster_alpha, width*height*NUMBER_OF_CLUSTERS*sizeof(__half)), debug);
	CHECK_CUDA(cudaMalloc(&d_cluster_r, width*height*NUMBER_OF_CLUSTERS*sizeof(__half)), debug);
	CHECK_CUDA(cudaMalloc(&d_cluster_g, width*height*NUMBER_OF_CLUSTERS*sizeof(__half)), debug);
	CHECK_CUDA(cudaMalloc(&d_cluster_b, width*height*NUMBER_OF_CLUSTERS*sizeof(__half)), debug);

	// Cluster.
	const float *features = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(
		FORWARD::cluster(tile_grid, block, width, height, radii, geomState.means2D, geomState.conic_opacity, geomState.
			depths, features, imgState.n_contrib, d_cluster_depth, d_cluster_alpha, d_cluster_r, d_cluster_g,
			d_cluster_b), debug);

	// Render.

	// // Let each tile blend its range of Gaussians independently in parallel
	// const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	// CHECK_CUDA(FORWARD::render(
	// 	tile_grid, block,
	// 	imgState.ranges,
	// 	binningState.point_list,
	// 	width, height,
	// 	geomState.means2D,
	// 	feature_ptr,
	// 	geomState.conic_opacity,
	// 	imgState.accum_alpha,
	// 	imgState.n_contrib,
	// 	background,
	// 	out_color,
	// 	geomState.depths,
	// 	depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
