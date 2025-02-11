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

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
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
	char* img_chunkptr = imageBuffer(img_chunk_size);
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

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	// Setup data structures for splat collection.
	int number_of_samples = width * height;
	constexpr int NUM_GAUSSIANS_PER_SAMPLE = 500;

	// CPU side buffers for splat collection.
	auto id_values = static_cast<int *>(calloc(number_of_samples * NUM_GAUSSIANS_PER_SAMPLE, sizeof(int)));
	auto alpha_values = static_cast<float *>(calloc(number_of_samples * NUM_GAUSSIANS_PER_SAMPLE, sizeof(float)));
	auto depth_values = static_cast<float *>(calloc(number_of_samples * NUM_GAUSSIANS_PER_SAMPLE, sizeof(float)));
	auto color_values = static_cast<float *>(calloc(number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * 3, sizeof(float)));
	auto background_color = static_cast<float *>(calloc(3, sizeof(float)));
	auto rendered_color = static_cast<float *>(calloc(number_of_samples * 3, sizeof(float)));

	// Transfer to GPU.
	int *device_id_values;
	float *device_alpha_values, *device_depth_values, *device_color_values;

	// Allocate device memory for splat collection.
	CHECK_CUDA(cudaMalloc(&device_id_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(int)), debug);
	CHECK_CUDA(cudaMalloc(&device_alpha_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(float)), debug);
	CHECK_CUDA(cudaMalloc(&device_depth_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(float)), debug);
	CHECK_CUDA(cudaMalloc(&device_color_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * 3 * sizeof(float)),
	           debug);

	// Transfer buffers to GPU.
	CHECK_CUDA(cudaMemcpy(device_id_values, id_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(int),
		           cudaMemcpyHostToDevice), debug);
	CHECK_CUDA(
		cudaMemcpy(device_alpha_values, alpha_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(float),
			cudaMemcpyHostToDevice), debug);
	CHECK_CUDA(
		cudaMemcpy(device_depth_values, depth_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(float),
			cudaMemcpyHostToDevice), debug);
	CHECK_CUDA(
		cudaMemcpy(device_color_values, color_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * 3 * sizeof(float),
			cudaMemcpyHostToDevice), debug);

	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list_keys,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth,
		NUM_GAUSSIANS_PER_SAMPLE,
		device_id_values,
		device_alpha_values,
		device_depth_values,
		device_color_values), debug);

	// Transfer buffers back to CPU.
	CHECK_CUDA(cudaMemcpy(id_values, device_id_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(int),
		           cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(
		cudaMemcpy(alpha_values, device_alpha_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(float),
			cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(
		cudaMemcpy(depth_values, device_depth_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * sizeof(float),
			cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(
		cudaMemcpy(color_values, device_color_values, number_of_samples * NUM_GAUSSIANS_PER_SAMPLE * 3 * sizeof(float),
			cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(cudaMemcpy(background_color, background, 3 * sizeof(float), cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(cudaMemcpy(rendered_color, out_color, number_of_samples * 3 * sizeof(float), cudaMemcpyDeviceToHost),
	           debug);

	// Create output file.
	char stream_buffer[8096];
	std::ofstream output_file;
	output_file.rdbuf()->pubsetbuf(stream_buffer, sizeof(stream_buffer));

	output_file.open("collected_splats.csv");

	// Write header.
	output_file << "sample_index,out_color_r,out_color_g,out_color_b,background_r,background_g,background_b,";
	for (int gaussian_index = 0; gaussian_index < NUM_GAUSSIANS_PER_SAMPLE; ++gaussian_index) {
		output_file << "gaussian_" << gaussian_index << "_id,gaussian_" << gaussian_index << "_alpha,gaussian_" <<
				gaussian_index << "_depth,gaussian_" << gaussian_index << "_color_r,gaussian_" << gaussian_index <<
				"_color_g,gaussian_" << gaussian_index << "_color_b";
		if (gaussian_index < NUM_GAUSSIANS_PER_SAMPLE - 1) {
			output_file << ',';
		}
	}
	output_file << '\n';

	// Write data.
	for (int sample_index = 0; sample_index < number_of_samples; ++sample_index) {
		// Write preambles.
		output_file << sample_index << ',' << rendered_color[sample_index] << ',' << rendered_color[
					number_of_samples + sample_index] <<
				',' <<
				rendered_color[2 * number_of_samples + sample_index] << ',' << background_color[0] << ',' <<
				background_color[1] << ',' <<
				background_color[2] << ',';

		// Write Gaussian data.
		for (int gaussian_index = 0; gaussian_index < NUM_GAUSSIANS_PER_SAMPLE; ++gaussian_index) {
			output_file << id_values[sample_index * NUM_GAUSSIANS_PER_SAMPLE + gaussian_index] << ',' << alpha_values[
				sample_index * NUM_GAUSSIANS_PER_SAMPLE + gaussian_index] << ',' << depth_values[
				sample_index * NUM_GAUSSIANS_PER_SAMPLE + gaussian_index] << ',' << color_values[
				sample_index * NUM_GAUSSIANS_PER_SAMPLE * 3 + gaussian_index * 3] << ',' << color_values[
				sample_index * NUM_GAUSSIANS_PER_SAMPLE * 3 + gaussian_index * 3 + 1] << ',' << color_values[
				sample_index * NUM_GAUSSIANS_PER_SAMPLE * 3 + gaussian_index * 3 + 2];

			if (gaussian_index < NUM_GAUSSIANS_PER_SAMPLE - 1) {
				output_file << ',';
			}
		}

		if (sample_index < number_of_samples - 1) {
			output_file << '\n';
		}

		std::cout << sample_index << " / " << number_of_samples << "\n" << std::endl;
	}
	output_file << std::endl;

	// Close file.
	output_file.close();

	// Cleanup.
	CHECK_CUDA(cudaFree(device_alpha_values), debug);
	CHECK_CUDA(cudaFree(device_depth_values), debug);
	CHECK_CUDA(cudaFree(device_color_values), debug);
	free(alpha_values);
	free(depth_values);
	free(color_values);
	free(background_color);
	free(rendered_color);

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
