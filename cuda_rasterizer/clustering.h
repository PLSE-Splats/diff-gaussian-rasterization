//
// Created by kenneth on 2/4/26.
//

#ifndef DIFFRAST_CLUSTERING_H
#define DIFFRAST_CLUSTERING_H

#include <cuda_fp16.h>

// Clustering parameters.
#define NUMBER_OF_CLUSTERS 12
#define NUMBER_OF_CLUSTER_PAIRS 6  // 12 / 2
#define MINIMUM_TRANSMITTANCE 0.0001f
#define MINIMUM_SPLAT_ALPHA 0.003921f  // 1 / 255
#define DEBUG_PIXEL (-1)               // 682741

// Half constants.
#define CUDART_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7BFFU)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)

/**
 * In-place swap sort for depth seeds.
 *
 * @param seeds Depth seeds.
 */
__device__ __forceinline__ void sort_seeds(__half2* __restrict__ seeds) {
#pragma unroll
  for (int pass = 0; pass < NUMBER_OF_CLUSTER_PAIRS; ++pass) {
    // Even phase: swap within a pair.
    // Comparing and swapping on A and B in (A, B) (C, D).
#pragma unroll
    for (int cluster_pair = 0; cluster_pair < NUMBER_OF_CLUSTER_PAIRS;
         ++cluster_pair) {
      const auto this_pair = seeds[cluster_pair];
      seeds[cluster_pair] =
          this_pair.x > this_pair.y ? __lowhigh2highlow(this_pair) : this_pair;
    }

    // Odd phase: swap across two pairs.
    // Comparing and swapping B and C in (A, B) (C, D).
#pragma unroll
    for (int cluster_pair = 0; cluster_pair < NUMBER_OF_CLUSTER_PAIRS - 1;
         ++cluster_pair) {
      const auto this_pair = seeds[cluster_pair];
      const auto next_pair = seeds[cluster_pair + 1];
      seeds[cluster_pair] = this_pair.y > next_pair.x
                                ? __half2(this_pair.x, next_pair.x)
                                : this_pair;
      seeds[cluster_pair + 1] = this_pair.y > next_pair.x
                                    ? __half2(this_pair.y, next_pair.y)
                                    : next_pair;
    }
  }
}

/**
 * Compute a mask on the clusters to apply a sample to.
 *
 * @param cluster_depths Current cluster depths.
 * @param sample_depth Sample depth to find cluster for.
 * @param mask Output mask on depths.
 */
__device__ __forceinline__ void build_cluster_selector_mask(
    const __half2* __restrict__ cluster_depths, const __half sample_depth,
    __half2* __restrict__ mask) {
  // Compute distances to each cluster.
  const auto sample_depth_2 = __half2half2(sample_depth);
  __half2 distances[NUMBER_OF_CLUSTER_PAIRS];
#pragma unroll
  for (int pair_index = 0; pair_index < NUMBER_OF_CLUSTER_PAIRS; ++pair_index) {
    distances[pair_index] =
        __habs2(cluster_depths[pair_index] - sample_depth_2);
  }

  // Find the min distance.
  __half min_distance = CUDART_MAX_NORMAL_FP16;
#pragma unroll
  for (int pair_index = 0;  // NOLINT(*-loop-convert)
       pair_index < NUMBER_OF_CLUSTER_PAIRS; ++pair_index) {
    min_distance = __hmin(
        min_distance, __hmin(distances[pair_index].x, distances[pair_index].y));
  }

  // Build a mask where 1 is on the min and 0 everywhere else.
#pragma unroll
  for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS;
       ++cluster_index) {
    mask[cluster_index] =
        __half2(distances[cluster_index].x == min_distance ? CUDART_ONE_FP16
                                                           : CUDART_ZERO_FP16,
                distances[cluster_index].y == min_distance ? CUDART_ONE_FP16
                                                           : CUDART_ZERO_FP16);
  }
}

#endif  // DIFFRAST_CLUSTERING_H
