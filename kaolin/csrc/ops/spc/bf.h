// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#pragma once

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> compactify_nodes(
  uint32_t num_nodes, 
  at::Tensor sum, 
  at::Tensor occ_ptr, 
  at::Tensor emp_ptr);

std::vector<at::Tensor> oracleB(
  at::Tensor Points, 
  uint32_t level, 
  float sigma, 
  at::Tensor cam, 
  at::Tensor dmap, 
  at::Tensor mipmap,
  int mip_levels);

std::vector<at::Tensor> oracleB_final(
  at::Tensor points,
  uint32_t level,
  float sigma,
  at::Tensor cam, 
  at::Tensor dmap);

std::vector<at::Tensor> process_final_voxels(
  uint32_t num_nodes, 
  uint32_t total_nodes, 
  at::Tensor state, 
  at::Tensor nvsum, 
  at::Tensor occup, 
  at::Tensor prev_state, 
  at::Tensor octree, 
  at::Tensor empty);

std::vector<at::Tensor> colorsB_final(
  at::Tensor points,
  uint32_t level,
  at::Tensor cam, 
  float sigma,
  at::Tensor im,
  at::Tensor dmap,
  at::Tensor probs);

std::vector<at::Tensor> merge_empty(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree0,
  at::Tensor octree1,  
  at::Tensor empty0,
  at::Tensor empty1,
  at::Tensor exsum0,
  at::Tensor exsum1);

std::vector<at::Tensor> bq_merge(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree0,
  at::Tensor octree1,  
  at::Tensor empty0,
  at::Tensor empty1,
  at::Tensor pyramid0,
  at::Tensor pyramid1,
  at::Tensor probs0,
  at::Tensor probs1,  
  at::Tensor colors0,
  at::Tensor colors1,
  at::Tensor normals0,
  at::Tensor normals1,
  at::Tensor exsum0,
  at::Tensor exsum1);

void bq_touch_extract(
  uint32_t num_nodes, 
  at::Tensor state, 
  at::Tensor nvsum, 
  at::Tensor prev_state);

std::vector<at::Tensor> bq_extract(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree, 
  at::Tensor empty,
  at::Tensor probs,
  at::Tensor pyramid,
  at::Tensor exsum);

std::vector<at::Tensor> bq_touch(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree, 
  at::Tensor empty,
  at::Tensor pyramid);

}  // namespace kaolin
