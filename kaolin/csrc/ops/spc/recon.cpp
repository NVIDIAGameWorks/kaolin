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

#ifdef WITH_CUDA

ulong GetTempSize(void* d_temp_storage, uint* d_M0, uint* d_M1, uint max_total_points);

void InclusiveSum_cuda(uint num, uint* inputs, uint* outputs, void* d_temp_storage, ulong temp_storage_bytes);
void BuildMip2D_cuda(float* img, uint width, uint miplevels, uint hw, 
                     float fx, float fy, float cx, float cy, float2* mipmap, float maxdepth, bool true_depth);

void Subdivide_cuda(uint num, point_data* points, uint* insum, point_data* new_points, uint* nvsum);
void Compactify_cuda(uint num, point_data* points, uint* insum, point_data* new_points, int64_t* out_nvsum);

#endif // WITH_CUDA

at::Tensor inclusive_sum(at::Tensor Inputs)
{
#ifdef WITH_CUDA
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
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor build_mip2d(at::Tensor image, at::Tensor In, int mip_levels, float maxdepth, bool true_depth)
{
#ifdef WITH_CUDA
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
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

std::vector<at::Tensor> subdivide(at::Tensor Points, at::Tensor Insum)
{
#ifdef WITH_CUDA
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({8*pass, 3}, Points.options());
  at::Tensor NVSum = at::zeros({pass}, Points.options().dtype(at::kInt));

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());
  uint* nvsum = reinterpret_cast<uint*>(NVSum.data_ptr<int>());

  Subdivide_cuda(num, points, insum, new_points, nvsum);

  return { NewPoints, NVSum };
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}


std::vector<at::Tensor> compactify(at::Tensor Points, at::Tensor Insum)
{
#ifdef WITH_CUDA
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({pass, 3}, Points.options());
  at::Tensor NVSum = at::zeros({pass}, Points.options().dtype(at::kLong));

  point_data*  points = reinterpret_cast<point_data*>(Points.data_ptr<short>());
  point_data*  new_points = reinterpret_cast<point_data*>(NewPoints.data_ptr<short>());

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());
  int64_t* nvsum = reinterpret_cast<int64_t*>(NVSum.data_ptr<int64_t>());

  Compactify_cuda(num, points, insum, new_points, nvsum);

  return { NewPoints, NVSum };
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

}  // namespace kaolin