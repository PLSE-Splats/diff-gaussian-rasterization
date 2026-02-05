//
// Created by kenneth on 2/4/26.
//

#ifndef DIFFRAST_CLUSTERING_H
#define DIFFRAST_CLUSTERING_H

#include "cuda_fp16.h"

// Clustering parameters.
#define NUMBER_OF_CLUSTERS 12
#define MINIMUM_TRANSMITTANCE 0.0001f
#define MINIMUM_SPLAT_ALPHA 0.003921f
#define DEBUG_PIXEL 682741

// Half constants.
#define CUDART_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7BFFU)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)

__device__ __forceinline__ void build_cluster_selector_mask(
    const __half* __restrict__ cluster_depths, const __half sample_depth,
    __half* __restrict__ mask) {
  // Compute distances to each cluster.
  __half distances[NUMBER_OF_CLUSTERS];

#pragma unroll
  for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS;
       ++cluster_index) {
    distances[cluster_index] =
        __habs(cluster_depths[cluster_index] - sample_depth);
  }

  // Find the min distance.
  __half min_distance = __hmin(distances[0], distances[1]);
#pragma unroll
  for (int cluster_index = 2; cluster_index < NUMBER_OF_CLUSTERS;
       ++cluster_index) {
    min_distance = __hmin(min_distance, distances[cluster_index]);
  }

  // Build a mask where 1 is on the min and 0 everywhere else.
#pragma unroll
  for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS;
       ++cluster_index) {
    mask[cluster_index] = distances[cluster_index] == min_distance
                              ? CUDART_ONE_FP16
                              : CUDART_ZERO_FP16;
  }
}

#endif  // DIFFRAST_CLUSTERING_H
