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
#include <cooperative_groups/reduce.h>

#include "auxiliary.h"
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

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_SIZE)
    clusterRenderCUDA(const int width, const int height,
                      const uint32_t* splat_ids, const uint2* splat_id_ranges,
                      const float2* means_2d, const float4* conic_opacity,
                      const float* depths, const float* features,
                      uint32_t* n_contributions, const float* bg_color,
                      float* final_transmittance, float* invdepth,
                      float* out_color) {
  // Gather thread information.
  const auto block = cg::this_thread_block();
  const uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
  const auto group_index = block.group_index();
  const auto thread_index = block.thread_index();
  const auto thread_rank = block.thread_rank();

  // Gather pixel information.
  const uint2 minimum_pixel_coordinate = {group_index.x * BLOCK_X,
                                          group_index.y * BLOCK_Y};
  const uint2 pixel_coordinate = {minimum_pixel_coordinate.x + thread_index.x,
                                  minimum_pixel_coordinate.y + thread_index.y};
  const uint32_t pixel_index = width * pixel_coordinate.y + pixel_coordinate.x;

  // Compute if this thread is associated with a visible pixel.
  const bool pixel_in_bounds =
      pixel_coordinate.x < width && pixel_coordinate.y < height;
  bool done = !pixel_in_bounds;

  // Load input range for this tile.
  const auto splat_id_range =
      splat_id_ranges[group_index.y * horizontal_blocks + group_index.x];
  int todo = static_cast<int>(splat_id_range.y - splat_id_range.x);
  const int rounds = (todo + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Allocate storage for batches of collectively fetched data.
  __shared__ uint32_t collected_splat_ids[BLOCK_SIZE];
  __shared__ __half collected_splat_depths[BLOCK_SIZE];
  __shared__ float2 collected_means_2d[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Clustering helper variables.
  uint32_t contributor = 0;

  // Local cluster data.
  __half pixel_cluster_depths[NUMBER_OF_CLUSTERS] = {};
  __half pixel_cluster_splat_counts[NUMBER_OF_CLUSTERS] = {};
  __half pixel_cluster_alpha_sums[NUMBER_OF_CLUSTERS] = {};
  __half pixel_cluster_alphas[NUMBER_OF_CLUSTERS];
  __half pixel_cluster_reds[NUMBER_OF_CLUSTERS] = {};
  __half pixel_cluster_greens[NUMBER_OF_CLUSTERS] = {};
  __half pixel_cluster_blues[NUMBER_OF_CLUSTERS] = {};
  unsigned short uninitialized_cluster_index = 0;

  // Initialize pixel_cluster_alphas to 1.0 (transmittance accumulator starts at
  // 1).
  for (auto& pixel_cluster_alpha : pixel_cluster_alphas) {
    pixel_cluster_alpha = CUDART_ONE_FP16;
  }

  // Iterate over batches until all done or range is complete.
  for (int batch_index = 0; batch_index < rounds;
       ++batch_index, todo -= BLOCK_SIZE) {
    // Collectively fetch per-splat data from global to shared.
    const unsigned int progress = batch_index * BLOCK_SIZE + thread_rank;
    if (splat_id_range.x + progress < splat_id_range.y) {
      const uint32_t collected_splat_id =
          splat_ids[splat_id_range.x + progress];
      collected_splat_ids[thread_rank] = collected_splat_id;
      collected_splat_depths[thread_rank] =
          __float2half(depths[collected_splat_id]);
      collected_means_2d[thread_rank] = means_2d[collected_splat_id];
      collected_conic_opacity[thread_rank] = conic_opacity[collected_splat_id];
    }
    block.sync();

    // Iterate over current batch (per thread).
    // Does nothing if the thread is not mapped to a valid pixel (done).
    for (int sample_index = 0; !done && sample_index < min(BLOCK_SIZE, todo);
         ++sample_index) {
      // Collect sample ID.
      const uint32_t sample_splat_id = collected_splat_ids[sample_index];

      // Keep track of current position in range.
      contributor++;

      // Compute splat alpha.

      // Resample using conic matrix (cf. "Surface
      // Splatting" by Zwicker et al., 2001)
      float2 xy = collected_means_2d[sample_index];
      float2 d = {xy.x - static_cast<float>(pixel_coordinate.x),
                  xy.y - static_cast<float>(pixel_coordinate.y)};
      float4 con_o = collected_conic_opacity[sample_index];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                    con_o.y * d.x * d.y;
      if (power > 0.0f) continue;

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity
      // and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      __half sample_alpha = __float2half(min(0.99f, con_o.w * exp(power)));
      if (sample_alpha < __float2half(1.0f / 255.0f)) continue;

      // Collect color.
      const __half sample_r =
          __float2half(features[sample_splat_id * CHANNELS + 0]);
      const __half sample_g =
          __float2half(features[sample_splat_id * CHANNELS + 1]);
      const __half sample_b =
          __float2half(features[sample_splat_id * CHANNELS + 2]);

      // Collect sample depth.
      const __half sample_depth = collected_splat_depths[sample_index];

      // Pick a target cluster.
      unsigned short target_cluster_index = 0;

      // Use the next uninitialized cluster.
      if (uninitialized_cluster_index < NUMBER_OF_CLUSTERS) {
        // Start with the next open cluster index.
        target_cluster_index = uninitialized_cluster_index;

        // Check for any exact matches before the uninitialized index.
        bool found_exact_match = false;
        for (int cluster_index = 0; cluster_index < uninitialized_cluster_index;
             ++cluster_index) {
          // Use the cluster if it's an exact match.
          if (pixel_cluster_depths[cluster_index] == sample_depth) {
            target_cluster_index = cluster_index;
            found_exact_match = true;
            break;
          }
        }

        // No exact match found, so will use the uninitialized cluster.
        if (!found_exact_match) {
          // Move to next uninitialized cluster for next time.
          uninitialized_cluster_index++;
        }
      }
      // Clusters are initialized, use the closest in depth.
      else {
        __half current_closest_depth_distance = CUDART_MAX_NORMAL_FP16;
        for (int cluster_index = 0; cluster_index < NUMBER_OF_CLUSTERS;
             ++cluster_index) {
          const __half distance_to_cluster =
              __habs(pixel_cluster_depths[cluster_index] - sample_depth);
          // Otherwise, find the closest in depth.
          // | cluster depth - sample depth | < current_closest_depth_distance
          if (distance_to_cluster < current_closest_depth_distance) {
            current_closest_depth_distance = distance_to_cluster;
            target_cluster_index = cluster_index;
          }
        }
      }

      // Target cluster found. Add the splat to it.
      pixel_cluster_splat_counts[target_cluster_index] =
          pixel_cluster_splat_counts[target_cluster_index] + CUDART_ONE_FP16;
      pixel_cluster_alpha_sums[target_cluster_index] =
          pixel_cluster_alpha_sums[target_cluster_index] + sample_alpha;
      pixel_cluster_alphas[target_cluster_index] =
          pixel_cluster_alphas[target_cluster_index] *
          (CUDART_ONE_FP16 - sample_alpha);
      pixel_cluster_reds[target_cluster_index] = __hfma(
          sample_alpha, sample_r, pixel_cluster_reds[target_cluster_index]);
      pixel_cluster_greens[target_cluster_index] = __hfma(
          sample_alpha, sample_g, pixel_cluster_greens[target_cluster_index]);
      pixel_cluster_blues[target_cluster_index] = __hfma(
          sample_alpha, sample_b, pixel_cluster_blues[target_cluster_index]);

      // Update cluster depth.
      const __half current_depth = pixel_cluster_depths[target_cluster_index];
      const __half count_h = pixel_cluster_splat_counts[target_cluster_index];
      const __half depth_diff = sample_depth - current_depth;
      const __half depth_delta = depth_diff / count_h;
      pixel_cluster_depths[target_cluster_index] = current_depth + depth_delta;
    }

    // Sync before next ingest batch.
    block.sync();
  }

  // Clustering is complete, need to finalize values and render.

  // Exit if this thread is not mapped to a valid pixel.
  if (done) {
    return;
  }

  // Write-out number of contributions.
  n_contributions[pixel_index] = contributor;

  // Initialize rendering variables.
  __half pixel_transmittance = CUDART_ONE_FP16;
  float expected_invdepth = 0.0f;
  __half pixel_color[CHANNELS] = {};

  // For each cluster, convert transmittance accumulator to alpha, normalize
  // RGB, and perform alpha over.
  __half lowest_depth = CUDART_ZERO_FP16;
  for (int output_cluster_index = 0; output_cluster_index < NUMBER_OF_CLUSTERS;
       ++output_cluster_index) {
    // Find the next lowest depth such that lowest_depth <
    // pixel_cluster_depths[candidate_index] < current_lowest_depth.
    int collection_index = 0;
    __half current_lowest_depth = CUDART_MAX_NORMAL_FP16;
    for (int candidate_index = 0; candidate_index < NUMBER_OF_CLUSTERS;
         ++candidate_index) {
      if (pixel_cluster_depths[candidate_index] < current_lowest_depth &&
          pixel_cluster_depths[candidate_index] > lowest_depth) {
        current_lowest_depth = pixel_cluster_depths[candidate_index];
        collection_index = candidate_index;
      }
    }

    // Update the lowest depth for next pass.
    lowest_depth = current_lowest_depth;

    // Convert cluster transmittance to alpha.
    const __half cluster_alpha =
        CUDART_ONE_FP16 - pixel_cluster_alphas[collection_index];

    // Compute the reciprocal of alpha sums and premultiplied alpha to avoid
    // repeated divisions.
    const __half alpha_sum_reciprocal =
        hrcp(pixel_cluster_alpha_sums[collection_index]) * cluster_alpha;

    // Get cluster data (and premultiply alphas).
    const __half cluster_red =
        alpha_sum_reciprocal * pixel_cluster_reds[collection_index];
    const __half cluster_green =
        alpha_sum_reciprocal * pixel_cluster_greens[collection_index];
    const __half cluster_blue =
        alpha_sum_reciprocal * pixel_cluster_blues[collection_index];

    // Contribute colors to pixel.
    pixel_color[0] = __hfma(cluster_red, pixel_transmittance, pixel_color[0]);
    pixel_color[1] = __hfma(cluster_green, pixel_transmittance, pixel_color[1]);
    pixel_color[2] = __hfma(cluster_blue, pixel_transmittance, pixel_color[2]);

    // Update invdepth.
    if (invdepth)
      expected_invdepth =
          fmaf(1.0f / __half2float(pixel_cluster_depths[collection_index]) *
                   __half2float(cluster_alpha),
               __half2float(pixel_transmittance), expected_invdepth);

    // Update transmittance.
    pixel_transmittance =
        pixel_transmittance *
        (CUDART_ONE_FP16 - __hmin(CUDART_ONE_FP16, cluster_alpha));
  }

  // Write to output color, adding background.
  final_transmittance[pixel_index] = __half2float(pixel_transmittance);
  if (invdepth) invdepth[pixel_index] = expected_invdepth;

  for (int channel = 0; channel < CHANNELS; ++channel) {
    out_color[channel * height * width + pixel_index] =
        __half2float(pixel_color[channel]) +
        __half2float(pixel_transmittance) * bg_color[channel];
  }
}

void FORWARD::cluster_render(dim3 grid_size, dim3 block_size, const int width,
                             const int height, const uint32_t* splat_ids,
                             const uint2* splat_id_ranges,
                             const float2* means_2d,
                             const float4* conic_opacities, const float* depths,
                             const float* features, uint32_t* n_contributions,
                             const float* bg_color, float* final_transmittance,
                             float* invdepth, float* out_color) {
  clusterRenderCUDA<NUM_CHANNELS><<<grid_size, block_size>>>(
      width, height, splat_ids, splat_id_ranges, means_2d, conic_opacities,
      depths, features, n_contributions, bg_color, final_transmittance,
      invdepth, out_color);
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
