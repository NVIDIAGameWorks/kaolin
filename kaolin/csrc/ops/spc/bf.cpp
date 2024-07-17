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

#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <vector_types.h>
#include <iostream>

#include "../../spc_math.h"

namespace kaolin {

using namespace std;
using namespace at::indexing;

void CompactifyNodes_cuda(
  uint32_t num_nodes, 
  uint32_t* d_insum, 
  uchar* d_occ_ptr, 
  uchar* d_emp_ptr, 
  uchar* d_octree, 
  uchar* d_empty);

void OracleB_cuda(
  uint32_t num,
  point_data* points,
  float4x4 T,
  float one_over_sigma,
  float* Dmap,
  int32_t depth_height,
  int32_t depth_width,
  int32_t mip_levels,
  float2* mip,
  int32_t hw,
  uint32_t* occ,
  uint32_t* estate);

void OracleBFinal_cuda(
  int32_t num, 
  point_data* points, 
  float4x4 T, 
  float one_over_sigma,
  float* Dmap, 
  int32_t depth_height,
  int32_t depth_width,
  uint32_t* occ, 
  uint32_t* estate, 
  float* out_probs,
  cudaTextureObject_t		ProfileCurve);

void ProcessFinalVoxels_cuda(
  uint32_t num_nodes, 
  uint32_t* d_state, 
  uint32_t* d_nvsum, 
  uint32_t* d_occup,  
  uint32_t* d_prev_state, 
  uchar* d_octree, 
  uchar* d_empty);

void ColorsBFinal_cuda(
  int32_t num, 
  point_data* points, 
  float4x4 T, 
  float one_over_sigma,
  float3* image, 
  float* Dmap, 
  int32_t depth_height,
  int32_t depth_width,
  float* probs,
  uchar4* out_colors,
  float3* out_normals,
  cudaTextureObject_t		ProfileCurve);

void MergeEmpty_cuda(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uchar* d_octree0, 
  uchar* d_octree1, 
  uchar* d_empty0, 
  uchar* d_empty1, 
  int32_t* d_exsum0, 
  int32_t* d_exsum1, 
  uint32_t* occ,
  uint32_t* estate);

void BQMerge_cuda(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uchar* d_octree0, 
  uchar* d_octree1, 
  uchar* d_empty0, 
  uchar* d_empty1, 
  float* d_prob0,
  float* d_prob1,
  uchar4* d_color0,
  uchar4* d_color1,
  float3* d_normals0,
  float3* d_normals1,  
  int32_t* d_exsum0, 
  int32_t* d_exsum1, 
  uint32_t offset0,
  uint32_t offset1,
  uint32_t* occ,
  uint32_t* estate,
  float* d_out_probs,
  uchar4* d_out_colors);

void TouchExtract_cuda(
  uint32_t num_nodes, 
  uint32_t* state, 
  uint32_t* nvsum, 
  uint32_t* prev_state);

  void BQExtract_cuda(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uchar* d_octree, 
  uchar* d_empty, 
  float* d_probs,
  int32_t* d_exsum, 
  uint32_t offset,
  uint32_t* occ,
  uint32_t* estate);

void BQTouch_cuda(
  uint32_t num, 
  uchar* d_octree, 
  uchar* d_empty, 
  uint32_t* occ,
  uint32_t* estate);

cudaArray* SetupProfileCurve(cudaTextureObject_t* ProfileCurve)
{
  uint32_t num = 9;
 
  uint32_t BPSVals[] = {	
    0x02000000,
    0x10080402,
    0x30241810,
    0x4f483c30,
    0x5658564f,
    0x484e5456,
    0x383c4248,
    0x31323438,
    0x30303031  };

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, num);
  cudaMemcpyToArray(cuArray, 0, 0, BPSVals, num * sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaResourceDesc resDescr;
  memset(&resDescr, 0, sizeof(cudaResourceDesc));
  resDescr.resType = cudaResourceTypeArray;
  resDescr.res.array.array = cuArray;

  cudaTextureDesc             texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;
  texDescr.normalizedCoords = 0;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  cudaCreateTextureObject(ProfileCurve, &resDescr, &texDescr, NULL);

  return cuArray;
}

std::vector<at::Tensor> compactify_nodes(uint32_t num_nodes, at::Tensor insum, at::Tensor occ_ptr, at::Tensor emp_ptr)
{
  uint32_t pass = insum[-1].item<int32_t>();

  at::Tensor octree = at::zeros({pass}, insum.options().dtype(at::kByte));
  at::Tensor empty = at::zeros({pass}, insum.options().dtype(at::kByte));

  uint32_t* d_insum = reinterpret_cast<uint32_t*>(insum.data_ptr<int32_t>());
  uchar* d_occ_ptr = occ_ptr.data_ptr<uchar>();
  uchar* d_emp_ptr = emp_ptr.data_ptr<uchar>();

  uchar* d_octree = octree.data_ptr<uchar>();
  uchar* d_empty = empty.data_ptr<uchar>();

  CompactifyNodes_cuda(num_nodes, d_insum, d_occ_ptr, d_emp_ptr, d_octree, d_empty);

  return { octree, empty };
}

std::vector<at::Tensor>  oracleB(
  at::Tensor Points, 
  uint32_t level, 
  float sigma, 
  at::Tensor cam, 
  at::Tensor dmap, 
  at::Tensor mipmap,
  int32_t mip_levels)
{
  uint32_t num = Points.size(0);

  int32_t h = dmap.size(0);
  int32_t w = dmap.size(1);

  int32_t s = pow(2, mip_levels);
  int32_t h0 = h/s;
  int32_t w0 = w/s;

  at::Tensor occupancy = at::zeros({num}, Points.options().dtype(at::kInt));
  at::Tensor empty_state = at::zeros({num}, Points.options().dtype(at::kInt));

  uint32_t* occ = reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>());
  uint32_t* estate = reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>());
  point_data*  points = reinterpret_cast<point_data*>(Points.data_ptr<short>());
  float* Dmap = dmap.data_ptr<float>();
  float4x4* Cam = reinterpret_cast<float4x4*>(cam.data_ptr<float>());
  float2* d_mip = reinterpret_cast<float2*>(mipmap.data_ptr<float>());

  float scale = 2.0/powf(2.0, (float)level);
  float4x4 M = make_float4x4(scale, 0.0f, 0.0f, 0.0f,
                    0.0f, scale, 0.0f, 0.0f,
                    0.0f, 0.0f, scale, 0.0f,
                    -1.0f, -1.0f, -1.0f, 1.0f);

  float4x4 T = M*(*Cam);

  TORCH_CHECK(sigma != 0.0f, "Sigma can't be zero");

  if (level < 1)
  {
    occupancy[0] = 1;
  }
  else
  {
    OracleB_cuda(num, points, T, sigma, Dmap, h, w, mip_levels, d_mip, h0*w0, occ, estate);
  }

  return {occupancy, empty_state};
}

std::vector<at::Tensor> oracleB_final(
  at::Tensor points,
  uint32_t level,
  float sigma,
  at::Tensor cam, 
  at::Tensor dmap)
{
  uint32_t num = points.size(0);

  int32_t h = dmap.size(0);
  int32_t w = dmap.size(1);

  at::Tensor occupancy = at::zeros({num}, points.options().dtype(at::kInt));
  at::Tensor empty_state = at::zeros({num}, points.options().dtype(at::kInt));
  at::Tensor out_probs = at::zeros({num}, points.options().dtype(at::kFloat));

  uint32_t* occ = reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>());
  uint32_t* estate = reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>());
  float*  d_out_probs = out_probs.data_ptr<float>();
  point_data*  d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());
  float* Dmap = dmap.data_ptr<float>();  
  float4x4* Cam = reinterpret_cast<float4x4*>(cam.data_ptr<float>());

  float scale = 2.0/powf(2.0, (float)level);
  float4x4 M = make_float4x4(scale, 0.0f, 0.0f, 0.0f,
                    0.0f, scale, 0.0f, 0.0f,
                    0.0f, 0.0f, scale, 0.0f,
                    -1.0f, -1.0f, -1.0f, 1.0f);

  float4x4 T = M*(*Cam);

  TORCH_CHECK(sigma != 0.0f, "Sigma can't be zero");
  float one_over_sigma = 1.0f/sigma;

  if (level < 1)
  {
    occupancy[0] = 1;
  }
  else
  {
    cudaTextureObject_t		ProfileCurve;
    cudaArray *cuArray = SetupProfileCurve(&ProfileCurve);

    OracleBFinal_cuda(num, d_points, T, one_over_sigma, Dmap, h, w, occ, estate, d_out_probs, ProfileCurve);

    cudaFreeArray(cuArray);
  }

  return {occupancy, empty_state, out_probs};
}

std::vector<at::Tensor> process_final_voxels(
  uint32_t num_nodes, 
  uint32_t total_nodes,
  at::Tensor state, 
  at::Tensor nvsum, 
  at::Tensor occup, 
  at::Tensor prev_state, 
  at::Tensor octree, 
  at::Tensor empty)
  {
 
    uint32_t* d_state = reinterpret_cast<uint32_t*>(state.data_ptr<int32_t>());
    uint32_t* d_nvsum = reinterpret_cast<uint32_t*>(nvsum.data_ptr<int32_t>());
    uint32_t* d_prev_state = reinterpret_cast<uint32_t*>(prev_state.data_ptr<int32_t>());

    uint32_t size = octree.size(0);
    TORCH_CHECK(total_nodes <= size, "PROCESS FINAL VOXEL MEMORY ERROR");

    uchar* d_octree = octree.data_ptr<uchar>() + size - total_nodes;
    uchar* d_empty = empty.data_ptr<uchar>() + size - total_nodes;
    uint32_t* d_occup = reinterpret_cast<uint32_t*>(occup.data_ptr<int32_t>() + size - total_nodes);

    ProcessFinalVoxels_cuda(num_nodes, d_state, d_nvsum, d_occup, d_prev_state, d_octree, d_empty);

    return {octree, empty, occup};
  }

std::vector<at::Tensor> colorsB_final(
  at::Tensor points,
  uint32_t level,
  at::Tensor cam, 
  float sigma,
  at::Tensor im,
  at::Tensor dmap,
  at::Tensor probs)
{
  uint32_t num = points.size(0);

  int32_t h = dmap.size(0);
  int32_t w = dmap.size(1);

  at::Tensor out_colors = at::zeros({num, 4}, points.options().dtype(at::kByte));
  at::Tensor out_normals = at::zeros({num, 3}, points.options().dtype(at::kFloat));

  uchar4*  d_out_colors = reinterpret_cast<uchar4*>(out_colors.data_ptr<uchar>());
  float3*  d_out_normals = reinterpret_cast<float3*>(out_normals.data_ptr<float>());

  point_data*  d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());
  float3* image = reinterpret_cast<float3*>(im.data_ptr<float>());
  float* Dmap = dmap.data_ptr<float>();  
  float*  d_probs = probs.data_ptr<float>();
  float4x4* Cam = reinterpret_cast<float4x4*>(cam.data_ptr<float>());

  float scale = 2.0/powf(2.0, (float)level);
  float4x4 M = make_float4x4(scale, 0.0f, 0.0f, 0.0f,
                    0.0f, scale, 0.0f, 0.0f,
                    0.0f, 0.0f, scale, 0.0f,
                    -1.0f, -1.0f, -1.0f, 1.0f);

  float4x4 T = M*(*Cam);

  TORCH_CHECK(sigma != 0.0f, "Sigma can't be zero");
  float one_over_sigma = 1.0f/sigma;

  cudaTextureObject_t		ProfileCurve;
  cudaArray *cuArray = SetupProfileCurve(&ProfileCurve);

  ColorsBFinal_cuda(num, d_points, T, one_over_sigma, image, Dmap, h, w, d_probs, d_out_colors, d_out_normals, ProfileCurve);

  cudaFreeArray(cuArray);

  return {out_colors, out_normals};
}

std::vector<at::Tensor> merge_empty(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree0,
  at::Tensor octree1,  
  at::Tensor empty0,
  at::Tensor empty1,
  at::Tensor pyramid0,
  at::Tensor pyramid1,
  at::Tensor exsum0,
  at::Tensor exsum1)
{
  uint32_t num = points.size(0);

  at::Tensor occupancy = at::zeros({num}, points.options().dtype(at::kInt));
  uint32_t* occ = reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>());

  at::Tensor empty_state = at::zeros({num}, points.options().dtype(at::kInt));
  uint32_t* estate = reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>());

  point_data*  d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

  uchar* d_octree0 = octree0.data_ptr<uchar>();  
  uchar* d_empty0 = empty0.data_ptr<uchar>();
  int32_t*  d_exsum0 = exsum0.data_ptr<int32_t>();
  auto pyramid0_a = pyramid0.accessor<int32_t, 3>();
  uint32_t offset0 = pyramid0_a[0][1][level];

  uchar* d_octree1 = octree1.data_ptr<uchar>();
  uchar* d_empty1 = empty1.data_ptr<uchar>();
  int32_t*  d_exsum1 = exsum1.data_ptr<int32_t>();
  auto pyramid1_a = pyramid1.accessor<int32_t, 3>();
  uint32_t offset1 = pyramid1_a[0][1][level];

  MergeEmpty_cuda(num, d_points, level, d_octree0, d_octree1, d_empty0, d_empty1, d_exsum0, d_exsum1, occ, estate);

  return { occupancy, empty_state };
}

std::vector<at::Tensor> bq_merge(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree0,
  at::Tensor octree1,  
  at::Tensor empty0,
  at::Tensor empty1,
  at::Tensor probs0,
  at::Tensor probs1,  
  at::Tensor colors0,
  at::Tensor colors1,
  at::Tensor normals0,
  at::Tensor normals1,
  at::Tensor pyramid0,
  at::Tensor pyramid1,
  at::Tensor exsum0,
  at::Tensor exsum1)
{
  uint32_t num = points.size(0);

  at::Tensor occupancy = at::zeros({num}, points.options().dtype(at::kInt));
  uint32_t* occ = reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>());

  at::Tensor empty_state = at::zeros({num}, points.options().dtype(at::kInt));
  uint32_t* estate = reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>());

  at::Tensor out_colors = at::zeros({num, 4}, points.options().dtype(at::kByte));
  uchar4*  d_out_colors = reinterpret_cast<uchar4*>(out_colors.data_ptr<uchar>());

  at::Tensor out_probs = at::zeros({num}, points.options().dtype(at::kFloat));
  float*  d_out_probs = out_probs.data_ptr<float>();

  at::Tensor out_normals = at::zeros({num, 3}, points.options().dtype(at::kFloat));
  float3*  d_out_normals = reinterpret_cast<float3*>(out_normals.data_ptr<float>());

  point_data*  d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

  uchar* d_octree0 = octree0.data_ptr<uchar>();  
  uchar* d_empty0 = empty0.data_ptr<uchar>();
  int32_t*  d_exsum0 = exsum0.data_ptr<int32_t>();
  auto pyramid0_a = pyramid0.accessor<int32_t, 3>();
  uint32_t offset0 = pyramid0_a[0][1][level];
  uchar4*  d_colors0 = reinterpret_cast<uchar4*>(colors0.data_ptr<uchar>());
  float*  d_probs0 = probs0.data_ptr<float>();
  float3*  d_normals0 = reinterpret_cast<float3*>(normals0.data_ptr<float>());

  uchar* d_octree1 = octree1.data_ptr<uchar>();
  uchar* d_empty1 = empty1.data_ptr<uchar>();
  int32_t*  d_exsum1 = exsum1.data_ptr<int32_t>();
  auto pyramid1_a = pyramid1.accessor<int32_t, 3>();
  uint32_t offset1 = pyramid1_a[0][1][level];
  uchar4*  d_colors1 = reinterpret_cast<uchar4*>(colors1.data_ptr<uchar>());
  float*  d_probs1 = probs1.data_ptr<float>();
  float3*  d_normals1 = reinterpret_cast<float3*>(normals1.data_ptr<float>());

  BQMerge_cuda(num, d_points, level, d_octree0, d_octree1, d_empty0, d_empty1, d_probs0, d_probs1, d_colors0, d_colors1, 
  d_normals0, d_normals1, d_exsum0, d_exsum1, offset0, offset1, occ, estate, d_out_probs, d_out_colors);

  return { occupancy, empty_state, out_probs, out_colors, out_normals };
}

std::vector<at::Tensor> bq_extract(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree, 
  at::Tensor empty,
  at::Tensor probs,
  at::Tensor pyramid,
  at::Tensor exsum)
  {
  uint32_t num = points.size(0);

  at::Tensor occupancy = at::zeros({num}, points.options().dtype(at::kInt));
  uint32_t* occ = reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>());

  at::Tensor empty_state = at::zeros({num}, points.options().dtype(at::kInt));
  uint32_t* estate = reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>());

  point_data*  d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

  uchar* d_octree = octree.data_ptr<uchar>();  
  uchar* d_empty = empty.data_ptr<uchar>();
  int32_t*  d_exsum = exsum.data_ptr<int32_t>();
  auto pyramid_a = pyramid.accessor<int32_t, 3>();
  uint32_t offset = pyramid_a[0][1][level];
  float*  d_probs = probs.data_ptr<float>();

  BQExtract_cuda(num, d_points, level, d_octree, d_empty, d_probs, d_exsum, offset, occ, estate);

  return { occupancy, empty_state };
}

std::vector<at::Tensor> bq_touch(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree, 
  at::Tensor empty,
  at::Tensor pyramid)
{
  TORCH_CHECK(level > 0, "touch level too low");

  auto pyramid_a = pyramid.accessor<int32_t, 3>();
  uint32_t num = pyramid_a[0][0][level-1];
  uint32_t offset = pyramid_a[0][1][level-1];

  at::Tensor occupancy = at::zeros({8*num}, octree.options().dtype(at::kInt));
  uint32_t* occ = reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>());

  at::Tensor empty_state = at::zeros({8*num}, octree.options().dtype(at::kInt));
  uint32_t* estate = reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>());

  uchar* d_octree = octree.data_ptr<uchar>();  
  uchar* d_empty = empty.data_ptr<uchar>();

  BQTouch_cuda(num, d_octree+offset, d_empty+offset, occ, estate);

  return { occupancy, empty_state };
}

void bq_touch_extract(
  uint32_t num_nodes, 
  at::Tensor state, 
  at::Tensor nvsum, 
  at::Tensor prev_state)
{
  uint32_t* d_state = reinterpret_cast<uint32_t*>(state.data_ptr<int32_t>());
  uint32_t* d_nvsum = reinterpret_cast<uint32_t*>(nvsum.data_ptr<int32_t>());
  uint32_t* d_prev_state = reinterpret_cast<uint32_t*>(prev_state.data_ptr<int32_t>());

  TouchExtract_cuda(num_nodes, d_state, d_nvsum, d_prev_state);
}

}  // namespace kaolin
