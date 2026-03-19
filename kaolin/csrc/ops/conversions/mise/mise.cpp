// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

// Copyright 2019 Lars Mescheder, Michael Oechsle,
// Michael Niemeyer, Andreas Geiger, Sebastian Nowozin
//
// Permission is hereby granted, free of charge,
// to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to 
// in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <ATen/ATen.h>
#include "mise.h"

namespace kaolin {

MISE::MISE(int32_t in_resolution_0, int32_t in_depth, double in_threshold) {
  TORCH_CHECK(in_resolution_0 > 0);
  resolution_0 = in_resolution_0;
  depth = in_depth;
  threshold = in_threshold;
  voxel_size_0 = (1 << depth);
  resolution = resolution_0 * voxel_size_0;
  
  voxels.reserve(resolution_0 * resolution_0 * resolution_0);

  Voxel voxel;
  GridPoint point;
  Vector3D loc;

  for (int i = 0; i < resolution_0; i++) {
    for (int j = 0; j < resolution_0; j++) {
      for (int k = 0; k < resolution_0; k++) {
        loc = Vector3D(i * voxel_size_0, j * voxel_size_0, k * voxel_size_0);
        voxel = Voxel(loc, 0, true);
        TORCH_CHECK(voxels.size() == i * resolution_0 * resolution_0 + j * resolution_0 + k);
        voxels.push_back(voxel);
      }
    }
  }

  int resolution_1 = resolution_0 + 1;
  grid_points.reserve(resolution_1 * resolution_1 * resolution_1);
  for (int i = 0; i < resolution_1; i++) {
    for (int j = 0; j < resolution_1; j++) {
      for (int k = 0; k < resolution_1; k++) {
        loc = Vector3D(i * voxel_size_0, j * voxel_size_0, k * voxel_size_0);
        TORCH_CHECK(grid_points.size() == i * resolution_1 * resolution_1 + j * resolution_1 + k);
        add_grid_point(loc);
      }
    }
  }
}

void MISE::update(at::Tensor points, at::Tensor values) {
  TORCH_CHECK(points.size(0) == values.size(0));
  TORCH_CHECK(points.size(1) == 3);
  long* points_ptr = points.data_ptr<long>();
  double* values_ptr = values.data_ptr<double>();
  Vector3D loc;
  long idx;

  for (int i = 0; i < points.size(0); i++) {
    loc = Vector3D(points_ptr[i * 3], points_ptr[i * 3 + 1], points_ptr[i * 3 + 2]);
    idx = get_grid_point_idx(loc);
    if (idx == -1)
      AT_ERROR("Point not in grid!");
    grid_points[idx].value = values_ptr[i];
    grid_points[idx].known = true;
  }
  subdivide_voxels();
}

at::Tensor MISE::query() {
  int n_unknown = 0;
  for (auto p : grid_points) {
    if (!p.known)
      n_unknown += 1;
  }

  at::Tensor points = at::zeros({n_unknown, 3}, std::nullopt, at::kLong,
                                std::nullopt, at::kCPU, std::nullopt);
  auto points_ptr = points.data_ptr<int64_t>();
  int idx = 0;
  for (auto p : grid_points) {
    if (!p.known) {
      points_ptr[idx * 3 + 0] = p.loc.x;
      points_ptr[idx * 3 + 1] = p.loc.y;
      points_ptr[idx * 3 + 2] = p.loc.z;
      idx++;
    }
  }
  return points;
}

at::Tensor MISE::to_dense() {
  int resolution_1 = resolution + 1;
  at::Tensor out = at::full({resolution_1, resolution_1, resolution_1}, std::numeric_limits<float>::quiet_NaN(), at::kFloat, std::nullopt, at::kCPU, std::nullopt);
  float* out_ptr = out.data_ptr<float>();
  for (auto point : grid_points) {
    out_ptr[point.loc.x * resolution_1 * resolution_1 + point.loc.y * resolution_1 + point.loc.z] = point.value;
  }

  for (int i = 1; i < resolution_1; i++) {
    for (int j = 0; j < resolution_1; j++) {
      for (int k = 0; k < resolution_1; k++) {
        if (std::isnan(out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k]))
          out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k] = out_ptr[(i-1) * resolution_1 * resolution_1 + j * resolution_1 + k];
      }
    }
  }

  for (int i = 0; i < resolution_1; i++) {
    for (int j = 1; j < resolution_1; j++) {
      for (int k = 0; k < resolution_1; k++) {
        if (std::isnan(out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k]))
          out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k] = out_ptr[i * resolution_1 * resolution_1 + (j-1) * resolution_1 + k];
      }
    }
  }

  for (int i = 0; i < resolution_1; i++) {
    for (int j = 0; j < resolution_1; j++) {
      for (int k = 1; k < resolution_1; k++) {
        if (std::isnan(out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k]))
          out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k] = out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k - 1];
        TORCH_CHECK(!std::isnan(out_ptr[i * resolution_1 * resolution_1 + j * resolution_1 + k]));
      }
    }
  }
  return out;
}

void MISE::subdivide_voxels() {
  std::vector<bool> next_to_positive;
  std::vector<bool> next_to_negative;
  long idx;
  Vector3D loc, adj_loc;

  next_to_positive.resize(voxels.size(), false);
  next_to_negative.resize(voxels.size(), false);

  for (auto grid_point : grid_points) {
    loc = grid_point.loc;
    if (!grid_point.known)
      continue;

    for (int i = -1; i < 1; i++) {
      for (int j = -1; j < 1; j++) {
        for (int k = -1; k < 1; k++) {
          adj_loc = Vector3D(loc.x + i, loc.y + j, loc.z + k);
          idx = get_voxel_idx(adj_loc);
          if (idx == -1)
            continue;

          if (grid_point.value >= threshold)
            next_to_positive[idx] = true;
          if (grid_point.value <= threshold)
            next_to_negative[idx] = true;
        }
      }
    }
  }

  int n_subdivide = 0;
  for (size_t idx = 0; idx < voxels.size(); idx++) {
    if (!voxels[idx].is_leaf || voxels[idx].level == depth)
      continue;
    if (next_to_positive[idx] && next_to_negative[idx])
      n_subdivide += 1;
  }

  voxels.reserve(voxels.size() + 8 * n_subdivide);
  grid_points.reserve(voxels.size() + 19 * n_subdivide);
  int init_voxel_size = voxels.size();

  for (int idx = 0; idx < init_voxel_size; idx++) {
    if (!voxels[idx].is_leaf || voxels[idx].level == depth)
      continue;
    if (next_to_positive[idx] && next_to_negative[idx])
      subdivide_voxel(idx);
  }
}

void MISE::subdivide_voxel(long idx) {
  Voxel voxel;
  GridPoint point;
  Vector3D loc0 = voxels[idx].loc;
  Vector3D loc;
  int new_level = voxels[idx].level + 1;
  int new_size = 1 << (depth - new_level);
  TORCH_CHECK(new_level <= depth);
  TORCH_CHECK(1 <= new_size && new_size <= voxel_size_0);

  voxels[idx].is_leaf = false;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        loc = Vector3D(loc0.x + i * new_size, loc0.y + j * new_size, loc0.z + k * new_size);
        voxel = Voxel(loc, new_level, true);
        voxels[idx].children[i][j][k] = voxels.size();
        voxels.push_back(voxel);
      }
    }
  }
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        loc = Vector3D(loc0.x + i * new_size, loc0.y + j * new_size, loc0.z + k * new_size);
        if (get_grid_point_idx(loc) == -1)
          add_grid_point(loc);
      }
    }
  }
}

long MISE::get_voxel_idx(Vector3D loc) {
  if (!(0 <= loc.x && loc.x < resolution && 0 <= loc.y && loc.y < resolution && 0 <= loc.z && loc.z < resolution))
    return -1;

  Vector3D loc0 = Vector3D(loc.x >> depth, loc.y >> depth, loc.z >> depth);
  int idx = loc0.x * resolution_0 * resolution_0 + loc0.y * resolution_0 + loc0.z;
  Voxel voxel = voxels[idx];
  TORCH_CHECK(voxel.loc.x == loc0.x * voxel_size_0);
  TORCH_CHECK(voxel.loc.y == loc0.y * voxel_size_0);
  TORCH_CHECK(voxel.loc.z == loc0.z * voxel_size_0);

  Vector3D loc_rel = Vector3D(
    loc.x - (loc0.x << depth),
    loc.y - (loc0.y << depth),
    loc.z - (loc0.z << depth)
  );

  Vector3D loc_offset;
  long voxel_size = voxel_size_0;

  while (!voxel.is_leaf) {
    voxel_size = voxel_size >> 1;
    TORCH_CHECK(voxel_size >= 1);
    loc_offset.x = (loc_rel.x >= voxel_size) ? 1 : 0;
    loc_offset.y = (loc_rel.y >= voxel_size) ? 1 : 0;
    loc_offset.z = (loc_rel.z >= voxel_size) ? 1 : 0;
    idx = voxel.children[loc_offset.x][loc_offset.y][loc_offset.z];
    voxel = voxels[idx];
    loc_rel.x -= loc_offset.x * voxel_size;
    loc_rel.y -= loc_offset.y * voxel_size;
    loc_rel.z -= loc_offset.z * voxel_size;
    TORCH_CHECK(0 <= loc_rel.x && loc_rel.x < voxel_size);
    TORCH_CHECK(0 <= loc_rel.y && loc_rel.y < voxel_size);
    TORCH_CHECK(0 <= loc_rel.z && loc_rel.z < voxel_size);
  }
  return idx;
}

void MISE::add_grid_point(Vector3D loc) {
  GridPoint point = GridPoint(loc, 0., false);
  int resolution_1 = resolution + 1;
  grid_point_hash[resolution_1 * resolution_1 * loc.x + resolution_1 * loc.y + loc.z] = grid_points.size();
  grid_points.push_back(point);
}

int MISE::get_grid_point_idx(Vector3D loc) {
  int resolution_1 = resolution + 1;
  auto p_idx = grid_point_hash.find(loc.x * resolution_1 * resolution_1 + loc.y * resolution_1 + loc.z);
  if (p_idx == grid_point_hash.end())
    return -1;

  int idx = p_idx->second;
  TORCH_CHECK(grid_points[idx].loc.x == loc.x);
  TORCH_CHECK(grid_points[idx].loc.y == loc.y);
  TORCH_CHECK(grid_points[idx].loc.z == loc.z);
  return idx;
}

int32_t MISE::get_resolution() {
  return resolution;
}

} // namespace kaolin
