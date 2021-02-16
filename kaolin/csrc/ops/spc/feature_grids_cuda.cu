// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <ATen/ATen.h>

#include "../../spc_math.h"
#include "../../utils.h"

#define THREADS_PER_BLOCK 64

namespace kaolin {

uint GetPyramid(uint* pyramid_ptr, int batch, int k, int level, int olevel);

__global__ void ToDenseKernelForward(
    const uint num,
    const point_data* Pdata,
    const uint C,
    const uint E,
    const float* Y,   //input
    float* X) {
    uint i_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (i_idx < num) {
        point_data p = Pdata[i_idx];
        for (int m = 0; m < C; m++) {
            X[E * (E * (E * m + p.x) + p.y) + p.z] = Y[C * i_idx + m];
        }
    }
}

__global__ void ToDenseKernelBackward(
    const uint num,
    const point_data* Pdata,
    const uint C,
    const uint E,
    float* Y,
    const float* X) {
  uint i_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (i_idx < num) {
      point_data p = Pdata[i_idx];
      for (int m = 0; m < C; m++) {
          Y[C * i_idx + m] = X[E * (E * (E * m + p.x) + p.y) + p.z];
      }
  }
}

void to_dense_forward_cuda_kernel_launch(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features,
    at::Tensor outputs) {

	uint batch_size = pyramid.size(0);
  uint feature_size = features.size(1);
  uint grid_size = outputs.size(2);
  uint num_features = grid_size * grid_size * grid_size;
  uint max_level = pyramid.size(2) - 2;

  point_data* points_ptr = reinterpret_cast<point_data*>(points.data_ptr<short>());
  uint* pyramid_ptr = reinterpret_cast<uint*>(pyramid.data_ptr<int>());

  float* features_ptr = features.data_ptr<float>();
  float* outputs_ptr = outputs.data_ptr<float>();

  for (int bidx = 0; bidx < batch_size; bidx++) {
    uint point_size = GetPyramid(pyramid_ptr, bidx, 0, level, max_level);

    // map input->output
    ToDenseKernelForward<<<(point_size + 63) / 64, 64>>>(
        point_size,
        points_ptr + GetPyramid(pyramid_ptr, bidx, 1, level, max_level),
        feature_size,
        grid_size,
        features_ptr,
        outputs_ptr);

    features_ptr += feature_size * point_size;
    outputs_ptr += feature_size * num_features;
    points_ptr += GetPyramid(pyramid_ptr, bidx, 1, max_level + 1, max_level);
  }
}

void to_dense_backward_cuda_kernel_launch(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features,
    at::Tensor grad_outputs,
    at::Tensor grad_features) {
	int batch_size = pyramid.size(0);
  int feature_size = features.size(1);
  uint grid_size = grad_outputs.size(2);
  uint num_features = grid_size * grid_size * grid_size;
  int max_level = pyramid.size(2) - 2;

  point_data* points_ptr = reinterpret_cast<point_data*>(points.data_ptr<short>());
  uint* pyramid_ptr = reinterpret_cast<uint*>(pyramid.data_ptr<int>());

  float* grad_features_ptr = grad_features.data_ptr<float>();
  float* grad_outputs_ptr = grad_outputs.data_ptr<float>();

  for (int bidx = 0; bidx < batch_size; bidx++) {
      uint point_size = GetPyramid(pyramid_ptr, bidx, 0, level, max_level);

      // backprop output->input
      ToDenseKernelBackward<<<(point_size + 63) / 64, 64>>>(
          point_size,// num of points
          points_ptr + GetPyramid(pyramid_ptr, bidx, 1, level, max_level),
          feature_size, // channel dimension
          grid_size,
          grad_features_ptr,
          grad_outputs_ptr);

      grad_features_ptr += feature_size * point_size;
      grad_outputs_ptr += feature_size * num_features;
      points_ptr += GetPyramid(pyramid_ptr, bidx, 1, max_level + 1, max_level);
  }

}

}  // namespace kaolin
