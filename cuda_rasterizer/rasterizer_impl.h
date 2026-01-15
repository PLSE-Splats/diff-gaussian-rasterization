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

#pragma once

#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include "rasterizer.h"

namespace CudaRasterizer {
template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count,
                   std::size_t alignment) {
  std::size_t offset =
      (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) &
      ~(alignment - 1);
  ptr = reinterpret_cast<T*>(offset);
  chunk = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState {
  size_t scan_size;
  float* depths;
  char* scanning_space;
  bool* clamped;
  int* internal_radii;
  float2* means2D;
  float* cov3D;
  float4* conic_opacity;
  float* rgb;
  uint32_t* point_offsets;
  uint32_t* tiles_touched;

  static GeometryState fromChunk(char*& chunk, size_t P);
};

/**
 * Clustering data structure.
 *
 * Organized in cluster layers of pixels (i.e. i and i+1 are the same cluster
 * level for pixels i and i+1).
 */
struct ClusterState {
  /**
   * Cluster depths.
   */
  __half* depths;

  /**
   * Number of splats per cluster.
   */
  unsigned short* splat_counts;

  /**
   * Sum of alpha values during clustering.
   */
  __half* alpha_sums;

  /**
   * Final alpha of the cluster.
   */
  __half* alphas;

  /**
   * Final red value of the cluster.
   */
  __half* reds;

  /**
   * Final green value of the cluster.
   */
  __half* greens;

  /**
   * Final blue value of the cluster.
   */
  __half* blues;

  /**
   * Allocate ClusterState structure from memory chunk.
   *
   * @param chunk Memory chunk location.
   * @param N Number of elements to allocate (pixels * clusters per pixel).
   * @return Allocated ClusterState structure.
   */
  static ClusterState fromChunk(char*& chunk, size_t N);
};

struct ImageState {
  uint2* ranges;
  uint32_t* n_contrib;
  float* accum_alpha;

  static ImageState fromChunk(char*& chunk, size_t N);
};

struct GroupingState {
  size_t grouping_size;
  uint16_t* unsorted_tile_ids;
  uint16_t* tile_ids;
  uint32_t* unsorted_splat_ids;
  uint32_t* splat_ids;
  char* grouping_space;

  static GroupingState fromChunk(char*& chunk, size_t P);
};

struct BinningState {
  size_t sorting_size;
  uint64_t* point_list_keys_unsorted;
  uint64_t* point_list_keys;
  uint32_t* point_list_unsorted;
  uint32_t* point_list;
  char* list_sorting_space;

  static BinningState fromChunk(char*& chunk, size_t P);
};

template <typename T>
size_t required(size_t P) {
  char* size = nullptr;
  T::fromChunk(size, P);
  return ((size_t)size) + 128;
}
};  // namespace CudaRasterizer