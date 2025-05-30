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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

// SKM parameters.
#define NUMBER_OF_CLUSTERS 12
// FIXME: Assumes channels is always 3.
#define NUMBER_OF_DATA_POINTS 7
#define DEPTH_INDEX 0
#define SPLAT_COUNT_INDEX 1
#define ALPHA_SUM_INDEX 2
#define TRANSMITTANCE_INDEX 3
#define PREMULTIPLIED_R_INDEX 4
#define PREMULTIPLIED_G_INDEX 5
#define PREMULTIPLIED_B_INDEX 6
#define MINIMUM_TRANSMITTANCE 0.0001f
#define DATA_AT(INDEX, DATA) (INDEX * NUMBER_OF_DATA_POINTS + DATA)

#define NUMBER_OF_CLUSTER_DATA_POINTS 6
#define CLUSTER_DEPTH_INDEX 0
#define CLUSTER_ALPHA_INDEX 1
#define CLUSTER_ALPHA_SUM_INDEX 2
#define CLUSTER_COLOR_R_INDEX 3
#define CLUSTER_COLOR_G_INDEX 4
#define CLUSTER_COLOR_B_INDEX 5
#define CLUSTER_AT(INDEX, CLUSTER, DATA) \
    (INDEX * NUMBER_OF_CLUSTERS * NUMBER_OF_CLUSTER_DATA_POINTS + CLUSTER * NUMBER_OF_CLUSTER_DATA_POINTS + DATA)

#endif