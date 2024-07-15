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

#include "../../spc_math.h"

#include <iostream>

namespace kaolin {
using namespace std;
using namespace at::indexing;

// #ifdef WITH_CUDA

ulong GetTempSize(void* d_temp_storage, uint* d_M0, uint* d_M1, uint max_total_points);

void InclusiveSum_cuda(uint num, uint* inputs, uint* outputs, void* d_temp_storage, ulong temp_storage_bytes);
void BuildMip2D_cuda(float* img, 
                     uint width, uint miplevels, uint hw, 
                     float fx, float fy, float cx, float cy, float2* mipmap, float maxdepth, bool true_depth);

void Compactify_cuda(uint num, point_data* points, uint* insum, point_data* new_points);
void Subdivide_cuda(uint num, point_data* points, uint* insum, point_data* new_points);

void Subdivide2_cuda(uint num, point_data* points, uint* insum, point_data* new_points, uint* nvsum);
void Compactify2_cuda(uint num, point_data* points, uint* insum, point_data* new_points, int64_t* out_nvsum);

void scalar_to_rgb_cuda(uint num, cudaTextureObject_t ColorRamp, float* scalars, uchar4* colors);

void slice_image_cuda(
  uchar*    octree, 
  uint32_t  level, 
  int32_t*  sum, 
  uint32_t  offset, 
  uint32_t  axis, 
  uint32_t  voxel_slice, 
  uint32_t  pixel_cnt, 
  uchar4*   d_image);

void slice_image_empty_cuda(
  uchar* octree, 
  uchar* empty, 
  uint32_t level, 
  int32_t* sum, 
  uint32_t offset, 
  uint32_t axis, 
  uint32_t voxel_slice, 
  uint32_t pixel_cnt, 
  uchar* d_image);

at::Tensor inclusive_sum(at::Tensor Inputs)
{
  uint num = Inputs.size(0);

  at::Tensor Outputs = at::zeros_like(Inputs);

  uint* inputs = reinterpret_cast<uint*>(Inputs.data_ptr<int>()); 
  uint* outputs = reinterpret_cast<uint*>(Outputs.data_ptr<int>());

  // set up memory for DeviceScan and DeviceRadixSort calls
  void* d_temp_storage = NULL;
  ulong temp_storage_bytes = GetTempSize(d_temp_storage, inputs, outputs, num);

  at::Tensor temp_storage = at::zeros({(long)temp_storage_bytes}, Inputs.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  InclusiveSum_cuda(num, inputs, outputs, d_temp_storage, temp_storage_bytes);

  return Outputs;
}

at::Tensor build_mip2d(at::Tensor image, at::Tensor In, int mip_levels, float maxdepth, bool true_depth)
{
  int h = image.size(0);
  int w = image.size(1);

  int s = pow(2, mip_levels);
  int h0 = h/s;
  int w0 = w/s;
  int size = (int)(h0*w0*(pow(4, mip_levels)-1)/3);

  at::Tensor mipmap = at::empty({ size, 2 }, image.options());

  float* img = image.data_ptr<float>();
  float2* mip = reinterpret_cast<float2*>(mipmap.data_ptr<float>());
  float* in = In.data_ptr<float>();

  float fx = in[0];
  float fy = in[5];
  float cx = in[8];
  float cy = in[9];

  BuildMip2D_cuda(img, w, mip_levels, h0*w0, fx, fy, cx, cy, mip, maxdepth, true_depth);

  return mipmap;
}

at::Tensor compactify(at::Tensor Points, at::Tensor Insum)
{
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({pass, 3}, Points.options());

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());

  Compactify_cuda(num, points, insum, new_points);

  return NewPoints;
}

at::Tensor subdivide(at::Tensor Points, at::Tensor Insum)
{
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({8*pass, 3}, Points.options());

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());

  Subdivide_cuda(num, points, insum, new_points);

  return NewPoints;
}

std::vector<at::Tensor> subdivide2(at::Tensor Points, at::Tensor Insum)
{
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({8*pass, 3}, Points.options());
  at::Tensor NVSum = at::zeros({pass}, Points.options().dtype(at::kInt));

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());
  uint* nvsum = reinterpret_cast<uint*>(NVSum.data_ptr<int>());

  Subdivide2_cuda(num, points, insum, new_points, nvsum);

  return { NewPoints, NVSum };
}


std::vector<at::Tensor> compactify2(at::Tensor Points, at::Tensor Insum)
{
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({pass, 3}, Points.options());
  at::Tensor NVSum = at::zeros({pass}, Points.options().dtype(at::kLong));

  point_data*  points = reinterpret_cast<point_data*>(Points.data_ptr<short>());
  point_data*  new_points = reinterpret_cast<point_data*>(NewPoints.data_ptr<short>());

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());
  int64_t* nvsum = reinterpret_cast<int64_t*>(NVSum.data_ptr<int64_t>());

  Compactify2_cuda(num, points, insum, new_points, nvsum);

  return { NewPoints, NVSum };
}

void SetupColorRamp(cudaTextureObject_t	ColorRamp, uint* cData, at::Tensor tex_colors)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  cudaArray *cuArray = (cudaArray *) tex_colors.data_ptr<int>();

  cudaMemcpyToArray(cuArray, 0, 0, cData, 5 * sizeof(uint), cudaMemcpyHostToDevice);

  cudaResourceDesc            resDescr;
  memset(&resDescr, 0, sizeof(cudaResourceDesc));
  resDescr.resType = cudaResourceTypeArray;
  resDescr.res.array.array = cuArray;

  cudaTextureDesc             texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;
  texDescr.normalizedCoords = true;
  texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&ColorRamp, &resDescr, &texDescr, NULL);
}

at::Tensor scalar_to_rgb(at::Tensor scalars)
{
  uint num = scalars.size(0);

  at::Tensor out_colors = at::zeros({num, 4}, scalars.options().dtype(at::kByte));

  uchar4*  d_out_colors = reinterpret_cast<uchar4*>(out_colors.data_ptr<uchar>());
  float* d_scalars = scalars.data_ptr<float>();

  uint cData[] = { 0xff0000ff, 0xff00ffff, 0xff00ff00, 0xffffff00, 0xffff0000 };
  cudaTextureObject_t		ColorRamp;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, 5);

  cudaMemcpyToArray(cuArray, 0, 0, cData, 5 * sizeof(uint), cudaMemcpyHostToDevice);

  cudaResourceDesc            resDescr;
  memset(&resDescr, 0, sizeof(cudaResourceDesc));
  resDescr.resType = cudaResourceTypeArray;
  resDescr.res.array.array = cuArray;

  cudaTextureDesc             texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;
  texDescr.normalizedCoords = true;
  texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&ColorRamp, &resDescr, &texDescr, NULL);

  scalar_to_rgb_cuda(num, ColorRamp, d_scalars, d_out_colors);

  cudaFreeArray(cuArray);

  return out_colors;
}

at::Tensor slice_image(
    at::Tensor  octree,
    at::Tensor  points,
    uint32_t    level,
    at::Tensor  pyramid,
    at::Tensor  prefixsum,
    uint32_t    axes,
    uint32_t    val)
{
  uint32_t w = 0x1 << level;
  uint32_t pixel_cnt = w*w;

  at::Tensor Image = at::zeros({pixel_cnt, 4}, octree.options().dtype(at::kByte));

  uchar* d_octree = octree.data_ptr<uchar>();

  int32_t*  d_sum = prefixsum.data_ptr<int>();
  uchar4* d_image = reinterpret_cast<uchar4*>(Image.data_ptr<uchar>());

  auto pyramid_a = pyramid.accessor<int, 3>();
  uint32_t offset = pyramid_a[0][1][level];

  slice_image_cuda(d_octree, level, d_sum, offset, axes, val, pixel_cnt, d_image);

  return Image;
}

at::Tensor slice_image_empty(
    at::Tensor octree,
    at::Tensor empty,
    at::Tensor points,
    uint32_t   level,
    at::Tensor pyramid,
    at::Tensor prefixsum,
    uint32_t   axes,
    uint32_t   val)
{
  uint32_t w = 0x1 << level;
  uint32_t pixel_cnt = w*w;

  at::Tensor Image = at::zeros({pixel_cnt}, octree.options().dtype(at::kByte));

  uchar* d_octree = octree.data_ptr<uchar>();
  uchar* d_empty = empty.data_ptr<uchar>();
  int32_t*  d_sum = prefixsum.data_ptr<int>();
  uchar* d_image = Image.data_ptr<uchar>();

  auto pyramid_a = pyramid.accessor<int, 3>();
  uint32_t offset = pyramid_a[0][1][level];

  slice_image_empty_cuda(d_octree, d_empty, level, d_sum, offset, axes, val, pixel_cnt, d_image);

  return Image;
}

}  // namespace kaolin