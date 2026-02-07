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

#include <cooperative_groups.h>

#include "auxiliary.h"
#include "clustering.h"
#include "cuda_fp16.h"
#include "forward.h"
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs,
                                        const glm::vec3* means,
                                        glm::vec3 campos, const float* shs,
                                        bool* clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);

  glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
  glm::vec3 result = SH_C0 * sh[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                 SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                 SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y,
                               float tan_fovx, float tan_fovy,
                               const float* cov3D, const float* viewmatrix) {
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3(mean, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J =
      glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f,
                focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[4], viewmatrix[8],
                          viewmatrix[1], viewmatrix[5], viewmatrix[9],
                          viewmatrix[2], viewmatrix[6], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3],
                            cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod,
                             const glm::vec4 rot, float* cov3D) {
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot;  // / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),
                          2.f * (x * z + r * y), 2.f * (x * y + r * z),
                          1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x),
                          1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(
    int P, int D, int M, const float* orig_points, const glm::vec3* scales,
    const float scale_modifier, const glm::vec4* rotations,
    const float* opacities, const float* shs, bool* clamped,
    const float* cov3D_precomp, const float* colors_precomp,
    const float* viewmatrix, const float* projmatrix, const glm::vec3* cam_pos,
    const int W, int H, const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y, int* radii, float2* points_xy_image,
    float* depths, float* cov3Ds, float* rgb, float4* conic_opacity,
    const dim3 grid, uint32_t* tiles_touched, bool prefiltered,
    bool antialiasing) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered,
                  p_view))
    return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1],
                   orig_points[3 * idx + 2]};
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  // If 3D covariance matrix is precomputed, use it, otherwise compute
  // from scaling and rotation parameters.
  const float* cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D,
                            viewmatrix);

  constexpr float h_var = 0.3f;
  const float det_cov = cov.x * cov.z - cov.y * cov.y;
  cov.x += h_var;
  cov.z += h_var;
  const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
  float h_convolution_scaling = 1.0f;

  if (antialiasing)
    h_convolution_scaling =
        sqrt(max(0.000025f,
                 det_cov / det_cov_plus_h_cov));  // max for numerical stability

  // Invert covariance (EWA algorithm)
  const float det = det_cov_plus_h_cov;

  if (det == 0.0f) return;
  float det_inv = 1.f / det;
  float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles.
  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

  // If colors have been precomputed, use them, otherwise convert
  // spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr) {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points,
                                          *cam_pos, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx] = p_view.z;
  radii[idx] = my_radius;
  points_xy_image[idx] = point_image;
  // Inverse 2D covariance and opacity neatly pack into one float4
  float opacity = opacities[idx];

  conic_opacity[idx] = {conic.x, conic.y, conic.z,
                        opacity * h_convolution_scaling};

  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
    seedClusterDepthsCUDA(const int width, const int height,
                          const unsigned int grid_width,
                          const uint32_t* __restrict__ splat_ids,
                          const uint2* __restrict__ splat_id_ranges,
                          const float2* __restrict__ means_2d,
                          const float4* __restrict__ conic_opacities,
                          const float* __restrict__ depths,
                          __half2* __restrict__ cluster_depth_seeds) {
  // Gather thread information.
  const auto block = cg::this_thread_block();
  const auto group_index = block.group_index();
  const auto thread_index = block.thread_index();
  const auto thread_rank = block.thread_rank();

  // Gather pixel information.
  const uint2 pixel_coordinate = {group_index.x * BLOCK_X + thread_index.x,
                                  group_index.y * BLOCK_Y + thread_index.y};
  const float2 pixel_coordinate_float = {
      static_cast<float>(pixel_coordinate.x),
      static_cast<float>(pixel_coordinate.y)};
  const auto pixel_index = pixel_coordinate.y * width + pixel_coordinate.x;

  // Compute if this thread is associated with a visible pixel.
  const bool pixel_in_bounds =
      pixel_coordinate.x < width && pixel_coordinate.y < height;

  // Load input range for this tile.
  const auto splat_id_range =
      splat_id_ranges[group_index.y * grid_width + group_index.x];
  int todo = static_cast<int>(splat_id_range.y - splat_id_range.x);
  const int rounds = (todo + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Allocate storage for batches of collectively fetched data.
  __shared__ __half collected_splat_depths[BLOCK_SIZE];
  __shared__ float2 collected_means_2d[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Local cluster data.
  __half2 pixel_cluster_depths[NUMBER_OF_CLUSTER_PAIRS] = {};

  // Which cluster needs to be seeded. Thread stops when all clusters are
  // seeded.
  unsigned short unseeded_cluster_index = 0;

  // Iterate over batches until all done or range is complete.
  for (int batch_index = 0; batch_index < rounds;
       ++batch_index, todo -= BLOCK_SIZE) {
    // Sync threads to prepare for collaborative fetching.
    // Exit if everyone is done seeding (or isn't a seeder).
    if (__syncthreads_and(!pixel_in_bounds ||
                          unseeded_cluster_index == NUMBER_OF_CLUSTERS)) {
      break;
    }

    // Collectively fetch per-splat data from global to shared.
    const unsigned int progress = batch_index * BLOCK_SIZE + thread_rank;
    if (splat_id_range.x + progress < splat_id_range.y) {
      const uint32_t collected_splat_id =
          splat_ids[splat_id_range.x + progress];
      collected_splat_depths[thread_rank] =
          __float2half(depths[collected_splat_id]);
      collected_means_2d[thread_rank] = means_2d[collected_splat_id];
      collected_conic_opacity[thread_rank] =
          conic_opacities[collected_splat_id];
    }

    // Sync on collaborative fetching before per-thread seeding.
    block.sync();

    // Iterate over current batch (per thread).
    // Does nothing if the thread is not mapped to a valid pixel (done).
    if (pixel_in_bounds) {
      for (int sample_index = 0; unseeded_cluster_index < NUMBER_OF_CLUSTERS &&
                                 sample_index < min(BLOCK_SIZE, todo);
           ++sample_index) {
        // Compute splat alpha (determines if it's in this pixel).

        // Resample using conic matrix (cf. "Surface
        // Splatting" by Zwicker et al., 2001)
        const float2 xy = collected_means_2d[sample_index];
        const float2 d = {xy.x - pixel_coordinate_float.x,
                          xy.y - pixel_coordinate_float.y};
        const float4 con_o = collected_conic_opacity[sample_index];
        const float power =
            -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
            con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        // Eq. (2) from 3D Gaussian splatting paper.
        // Obtain alpha by multiplying with Gaussian opacity
        // and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix).
        const float sample_alpha_float = min(0.99f, con_o.w * __expf(power));
        if (sample_alpha_float < MINIMUM_SPLAT_ALPHA) continue;

        // Collect sample depth.
        const __half sample_depth = collected_splat_depths[sample_index];

        // Seed the next unseeded cluster with this splat's depth.
        pixel_cluster_depths[unseeded_cluster_index / 2].x =
            unseeded_cluster_index % 2 == 0
                ? sample_depth
                : pixel_cluster_depths[unseeded_cluster_index / 2].x;
        pixel_cluster_depths[unseeded_cluster_index / 2].y =
            unseeded_cluster_index % 2 == 1
                ? sample_depth
                : pixel_cluster_depths[unseeded_cluster_index / 2].y;

        // FIXME: Consider checking if sample_depth was already used (unlikely).

        // Move to next unseeded cluster for next time.
        unseeded_cluster_index++;
      }
    }
  }

  // Seeding is complete, write out to global memory in depth order.

  // Exit if this thread is not mapped to a valid pixel.
  if (!pixel_in_bounds) {
    return;
  }

  // Sort seeds.
  sort_seeds(pixel_cluster_depths);

  // Write to output.
  const auto output_base = pixel_index * NUMBER_OF_CLUSTER_PAIRS;
#pragma unroll
  for (int pair_index = 0; pair_index < NUMBER_OF_CLUSTER_PAIRS; ++pair_index) {
    cluster_depth_seeds[output_base + pair_index] =
        pixel_cluster_depths[pair_index];
  }
}

__global__ void __launch_bounds__(BLOCK_SIZE) clusterRenderCUDA(
    const int width, const int height, const unsigned int grid_width,
    const uint32_t* __restrict__ splat_ids,
    const uint2* __restrict__ splat_id_ranges,
    const float2* __restrict__ means_2d,
    const float4* __restrict__ conic_opacities,
    const float* __restrict__ depths, const float* __restrict__ features,
    const float* __restrict__ bg_color,
    const __half2* __restrict__ cluster_depth_seeds,
    uint32_t* __restrict__ n_contributions, float* __restrict__ inv_depth,
    float* __restrict__ final_transmittance, float* __restrict__ out_color) {
  // Gather thread information.
  const auto block = cg::this_thread_block();
  const auto group_index = block.group_index();
  const auto thread_index = block.thread_index();
  const auto thread_rank = block.thread_rank();

  // Gather pixel information.
  const uint2 pixel_coordinate = {group_index.x * BLOCK_X + thread_index.x,
                                  group_index.y * BLOCK_Y + thread_index.y};
  const float2 pixel_coordinate_float = {
      static_cast<float>(pixel_coordinate.x),
      static_cast<float>(pixel_coordinate.y)};
  const auto pixel_index = pixel_coordinate.y * width + pixel_coordinate.x;

  // Compute if this thread is associated with a visible pixel.
  const bool pixel_in_bounds =
      pixel_coordinate.x < width && pixel_coordinate.y < height;

  // Load input range for this tile.
  const auto splat_id_range =
      splat_id_ranges[group_index.y * grid_width + group_index.x];
  int todo = static_cast<int>(splat_id_range.y - splat_id_range.x);
  const int rounds = (todo + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Allocate storage for batches of collectively fetched data.
  __shared__ __half collected_splat_depths[BLOCK_SIZE];
  __shared__ float2 collected_means_2d[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
  __shared__ __half collected_colors[3 * BLOCK_SIZE];

  // Clustering helper variables.
  uint32_t contributor = 0;

  // Cluster data.
  __half2 pair_depths[NUMBER_OF_CLUSTER_PAIRS] = {};
  __half2 pair_splat_counts[NUMBER_OF_CLUSTER_PAIRS] = {};
  __half2 pair_alpha_sums[NUMBER_OF_CLUSTER_PAIRS] = {};
  __half2 pair_transmittances[NUMBER_OF_CLUSTER_PAIRS];
  __half2 pair_reds[NUMBER_OF_CLUSTER_PAIRS] = {};
  __half2 pair_greens[NUMBER_OF_CLUSTER_PAIRS] = {};
  __half2 pair_blues[NUMBER_OF_CLUSTER_PAIRS] = {};

  // Setup clustering for rendering pixels.
  if (pixel_in_bounds) {
    // Pull in depth seeds.
    const auto input_base = pixel_index * NUMBER_OF_CLUSTER_PAIRS;
#pragma unroll
    for (int pair_index = 0; pair_index < NUMBER_OF_CLUSTER_PAIRS;
         ++pair_index) {
      pair_depths[pair_index] = cluster_depth_seeds[input_base + pair_index];
    }

    // Initialize cluster transmittance to 1.0.
#pragma unroll
    for (int pair_index = 0;  // NOLINT(*-loop-convert)
         pair_index < NUMBER_OF_CLUSTER_PAIRS; ++pair_index) {
      pair_transmittances[pair_index] = ONE_FP16_2;
    }
  }

  // Iterate over batches until all done or range is complete.
  for (int batch_index = 0; batch_index < rounds;
       ++batch_index, todo -= BLOCK_SIZE) {
    // Collectively fetch per-splat data from global to shared.
    const unsigned int progress = batch_index * BLOCK_SIZE + thread_rank;
    if (splat_id_range.x + progress < splat_id_range.y) {
      const uint32_t collected_splat_id =
          splat_ids[splat_id_range.x + progress];
      collected_splat_depths[thread_rank] =
          __float2half(depths[collected_splat_id]);
      collected_means_2d[thread_rank] = means_2d[collected_splat_id];
      collected_conic_opacity[thread_rank] =
          conic_opacities[collected_splat_id];
      collected_colors[3 * thread_rank] = features[3 * collected_splat_id];
      collected_colors[3 * thread_rank + 1] =
          features[3 * collected_splat_id + 1];
      collected_colors[3 * thread_rank + 2] =
          features[3 * collected_splat_id + 2];
    }

    // Sync on collaborative fetching before per-thread seeding.
    block.sync();

    // Iterate over current batch (per thread).
    // Does nothing if the thread is not mapped to a valid pixel.
    if (pixel_in_bounds) {
      for (int sample_index = 0; sample_index < min(BLOCK_SIZE, todo);
           ++sample_index) {
        // Keep track of current position in range.
        contributor++;

        // Compute splat alpha (determines if it's in this pixel).

        // Resample using conic matrix (cf. "Surface
        // Splatting" by Zwicker et al., 2001)
        const float2 xy = collected_means_2d[sample_index];
        const float2 d = {xy.x - pixel_coordinate_float.x,
                          xy.y - pixel_coordinate_float.y};
        const float4 con_o = collected_conic_opacity[sample_index];
        const float power =
            -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
            con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        // Eq. (2) from 3D Gaussian splatting paper.
        // Obtain alpha by multiplying with Gaussian opacity
        // and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix).
        const float sample_alpha_float = min(0.99f, con_o.w * __expf(power));
        if (sample_alpha_float < MINIMUM_SPLAT_ALPHA) continue;

        // Broadcast alpha.
        const __half2 sample_alpha = __float2half2_rn(sample_alpha_float);

        // Collect color and broadcast.
        const __half2 sample_r =
            __half2half2(collected_colors[3 * sample_index]);
        const __half2 sample_g =
            __half2half2(collected_colors[3 * sample_index + 1]);
        const __half2 sample_b =
            __half2half2(collected_colors[3 * sample_index + 2]);

        // Collect sample depth.
        const __half sample_depth = collected_splat_depths[sample_index];
        const auto sample_depth_2 = __half2half2(sample_depth);

        // Mask for the closest cluster.
        __half2 mask[NUMBER_OF_CLUSTER_PAIRS] = {};
        build_cluster_selector_mask(pair_depths, sample_depth, mask);

        // Add splat to target cluster.
#pragma unroll
        for (int pair_index = 0; pair_index < NUMBER_OF_CLUSTER_PAIRS;
             ++pair_index) {
          const __half2 selector = mask[pair_index];
          if (selector.x == ONE_FP16 || selector.y == ONE_FP16) {
            const __half2 selected_alpha = selector * sample_alpha;

            pair_splat_counts[pair_index] += selector;
            pair_alpha_sums[pair_index] += selected_alpha;
            pair_transmittances[pair_index] *= ONE_FP16_2 - selected_alpha;
            pair_reds[pair_index] =
                __hfma2(selected_alpha, sample_r, pair_reds[pair_index]);
            pair_greens[pair_index] =
                __hfma2(selected_alpha, sample_g, pair_greens[pair_index]);
            pair_blues[pair_index] =
                __hfma2(selected_alpha, sample_b, pair_blues[pair_index]);

            // Update cluster depth.
            const __half2 depth_diff = sample_depth_2 - pair_depths[pair_index];
            const __half2 safe_count =
                selector *
                h2rcp(__hmax2(SMALL_FP16_2, pair_splat_counts[pair_index]));
            pair_depths[pair_index] =
                __hfma2(depth_diff, safe_count, pair_depths[pair_index]);
          }
        }
      }
    }

    // Sync on per-thread seeding before next fetch batch.
    block.sync();
  }

  // Clustering is complete.

  // Exit if this thread is not mapped to a valid pixel.
  if (!pixel_in_bounds) {
    return;
  }

  // Write out number of contributions.
  n_contributions[pixel_index] = contributor;

  // Initialize rendering variables.
  __half pixel_transmittance = ONE_FP16;
  __half pixel_red = ZERO_FP16;
  __half pixel_green = ZERO_FP16;
  __half pixel_blue = ZERO_FP16;
  __half expected_invdepth = ZERO_FP16;

  // Composite cluster to pixel color.
#pragma unroll
  for (int pair_index = 0; pair_index < NUMBER_OF_CLUSTER_PAIRS; ++pair_index) {
    // Convert transmittance to alpha.
    const auto pair_alpha = ONE_FP16_2 - pair_transmittances[pair_index];

    // Normalize RGB.
    const auto alpha_sum_reciprocal =
        h2rcp(__hmax2(SMALL_FP16_2, pair_alpha_sums[pair_index])) * pair_alpha;
    const auto red = pair_reds[pair_index] * alpha_sum_reciprocal;
    const auto green = pair_greens[pair_index] * alpha_sum_reciprocal;
    const auto blue = pair_blues[pair_index] * alpha_sum_reciprocal;

    // Composite color.
    pixel_red = __hfma(red.x, pixel_transmittance, pixel_red);
    pixel_green = __hfma(green.x, pixel_transmittance, pixel_green);
    pixel_blue = __hfma(blue.x, pixel_transmittance, pixel_blue);
    if (inv_depth) {
      expected_invdepth = __hfma(hrcp(pair_depths[pair_index].x) * pair_alpha.x,
                                 pixel_transmittance, expected_invdepth);
    }
    pixel_transmittance *= ONE_FP16 - __hmin(ONE_FP16, pair_alpha.x);

    pixel_red = __hfma(red.y, pixel_transmittance, pixel_red);
    pixel_green = __hfma(green.y, pixel_transmittance, pixel_green);
    pixel_blue = __hfma(blue.y, pixel_transmittance, pixel_blue);
    if (inv_depth) {
      expected_invdepth = __hfma(hrcp(pair_depths[pair_index].y) * pair_alpha.y,
                                 pixel_transmittance, expected_invdepth);
    }
    pixel_transmittance *= ONE_FP16 - __hmin(ONE_FP16, pair_alpha.y);
  }

  // Write out final inverse depth.
  if (inv_depth) {
    inv_depth[pixel_index] = __half2float(expected_invdepth);
  }

  // Write out final transmittance.
  final_transmittance[pixel_index] = __half2float(pixel_transmittance);

  // Write to color output (and apply background color).
  const auto number_of_pixels = width * height;
  out_color[pixel_index] = __half2float(
      __hfma(pixel_transmittance, __float2half(bg_color[0]), pixel_red));
  out_color[1 * number_of_pixels + pixel_index] = __half2float(
      __hfma(pixel_transmittance, __float2half(bg_color[1]), pixel_green));
  out_color[2 * number_of_pixels + pixel_index] = __half2float(
      __hfma(pixel_transmittance, __float2half(bg_color[2]), pixel_blue));
}

void FORWARD::preprocess(int P, int D, int M, const float* means3D,
                         const glm::vec3* scales, const float scale_modifier,
                         const glm::vec4* rotations, const float* opacities,
                         const float* shs, bool* clamped,
                         const float* cov3D_precomp,
                         const float* colors_precomp, const float* viewmatrix,
                         const float* projmatrix, const glm::vec3* cam_pos,
                         const int W, int H, const float focal_x, float focal_y,
                         const float tan_fovx, float tan_fovy, int* radii,
                         float2* means2D, float* depths, float* cov3Ds,
                         float* rgb, float4* conic_opacity, const dim3 grid,
                         uint32_t* tiles_touched, bool prefiltered,
                         bool antialiasing) {
  preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
      P, D, M, means3D, scales, scale_modifier, rotations, opacities, shs,
      clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos,
      W, H, tan_fovx, tan_fovy, focal_x, focal_y, radii, means2D, depths,
      cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered,
      antialiasing);
}

void FORWARD::seed_cluster_depths(dim3 grid_size, dim3 block_size,
                                  const int width, const int height,
                                  const uint32_t* __restrict__ splat_ids,
                                  const uint2* __restrict__ splat_id_ranges,
                                  const float2* __restrict__ means_2d,
                                  const float4* __restrict__ conic_opacities,
                                  const float* __restrict__ depths,
                                  __half2* __restrict__ cluster_depth_seeds) {
  seedClusterDepthsCUDA<<<grid_size, block_size>>>(
      width, height, grid_size.x, splat_ids, splat_id_ranges, means_2d,
      conic_opacities, depths, cluster_depth_seeds);
}
void FORWARD::cluster_render(
    dim3 grid_size, dim3 block_size, const int width, const int height,
    const uint32_t* __restrict__ splat_ids,
    const uint2* __restrict__ splat_id_ranges,
    const float2* __restrict__ means_2d,
    const float4* __restrict__ conic_opacities,
    const float* __restrict__ depths, const float* __restrict__ features,
    const float* __restrict__ bg_color,
    const __half2* __restrict__ cluster_depth_seeds,
    uint32_t* __restrict__ n_contributions, float* __restrict__ inv_depth,
    float* __restrict__ final_transmittance, float* __restrict__ out_color) {
  clusterRenderCUDA<<<grid_size, block_size>>>(
      width, height, grid_size.x, splat_ids, splat_id_ranges, means_2d,
      conic_opacities, depths, features, bg_color, cluster_depth_seeds,
      n_contributions, inv_depth, final_transmittance, out_color);
}
