// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "triangle_hash.h"

namespace kaolin {


TriangleHash::TriangleHash(at::Tensor triangles, int resolution) {
  at::TensorArg triangles_arg{triangles, "triangles", 1};
  TORCH_CHECK(resolution > 0, "resolution must be positive");
  at::checkDeviceType(__func__, triangles, at::DeviceType::CPU);
  at::checkScalarTypes(__func__, triangles_arg, {at::kHalf, at::kFloat, at::kDouble});
  at::checkDim(__func__, triangles_arg, 3);
  at::checkSize(__func__, triangles_arg, 1, 3);
  at::checkSize(__func__, triangles_arg, 2, 2);
  _spatial_hash.resize(resolution * resolution);
  _resolution = resolution;
  _build_hash(triangles);
}

void TriangleHash::_build_hash(at::Tensor triangles) {
  using namespace at::indexing;
  const int n_tri = triangles.size(0);
  int bbox_min[2];
  int bbox_max[2];
  int ubound = _resolution - 1;
  const float* triangles_ptr = triangles.data_ptr<float>();
  float a, b, c, _min, _max;

  for (int i_tri = 0; i_tri < n_tri; i_tri++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      a = triangles_ptr[i_tri * 6 + j];
      b = triangles_ptr[i_tri * 6 + j + 2];
      c = triangles_ptr[i_tri * 6 + j + 4];
      if (a < b) {
        _min = a;
	_max = b;
      } else {
        _min = b;
        _max = a;
      }
      if (c > _max)
        _max = c;
      if (c < _min)
        _min = c;
      bbox_min[j] = static_cast<int>(_min);
      bbox_max[j] = static_cast<int>(_max);
      if (bbox_min[j] < 0) {
        bbox_min[j] = 0;
        if (bbox_max[j] < 0)
          bbox_max[j] = 0;
      } else if (bbox_min[j] > ubound) {
         bbox_min[j] = ubound;
         bbox_max[j] = ubound;
      } else if (bbox_max[j] > ubound) {
         bbox_max[j] = ubound;
      }
    }
    for (int x = (bbox_min[0]); x < (bbox_max[0]) + 1; x++) {
      for (int y = (bbox_min[1]); y < (bbox_max[1]) + 1; y++) {
        _spatial_hash[_resolution * x + y].push_back(i_tri);
      }
    }
  }
}

std::vector<at::Tensor> TriangleHash::query(at::Tensor points) {
  at::TensorArg points_arg{points, "points", 1};
  at::checkDeviceType(__func__, points, at::DeviceType::CPU);
  at::checkScalarTypes(__func__, points_arg, {at::kHalf, at::kFloat, at::kDouble});
  at::checkDim(__func__, points_arg, 2);
  at::checkSize(__func__, points_arg, 1, 2);

  using namespace at::indexing;

  const int n_points = points.size(0);
  std::vector<int> points_indices { };
  std::vector<int> tri_indices { };
  const float* points_ptr = points.data_ptr<float>();
  const float ubound = static_cast<float>(_resolution);

  for (int i_point = 0; i_point < n_points; i_point++) {
    float x = points_ptr[i_point * 2];
    float y = points_ptr[i_point * 2 + 1];
    if (x < 0.f || x >= ubound || y < 0.f || ubound >= y)
      continue;
    const int spatial_idx = _resolution * static_cast<int>(x) + static_cast<int>(y);
    for (int i_tri : _spatial_hash[spatial_idx]) {
      points_indices.push_back(i_point);
      tri_indices.push_back(i_tri);
    }
  }
  auto options = points.options();
  auto points_indices_tensor = at::zeros(points_indices.size(), options.dtype(at::kLong));
  auto tri_indices_tensor = at::zeros(tri_indices.size(), options.dtype(at::kLong));

  for (int k = 0; k < points_indices.size(); k++) {
    points_indices_tensor[k] = points_indices[k];
  }
  for (int k = 0; k < tri_indices.size(); k++) {
    tri_indices_tensor[k] = tri_indices[k];
  }

  return {points_indices_tensor, tri_indices_tensor};
}

}  // namespace kaolin
