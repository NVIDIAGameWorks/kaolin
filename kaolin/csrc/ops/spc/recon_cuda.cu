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

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#include <torch/extension.h>
#include <stdio.h>
#include <math_constants.h>

#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../spc_math.h"
#include "../../spc_utils.cuh"

namespace kaolin {

using namespace cub;
using namespace std;

uint64_t GetTempSize(void* d_temp_storage, uint* d_M0, uint* d_M1, uint max_total_points)
{
    uint64_t    temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_M0, d_M1, max_total_points));
    return temp_storage_bytes;
}


at::Tensor inclusive_sum_cuda_x(at::Tensor Inputs)
{
  uint32_t num = Inputs.size(0);

  at::Tensor Outputs = at::zeros_like(Inputs);

  uint32_t* inputs = reinterpret_cast<uint32_t*>(Inputs.data_ptr<int>()); 
  uint32_t* outputs = reinterpret_cast<uint32_t*>(Outputs.data_ptr<int>());

  // set up memory for DeviceScan and DeviceRadixSort calls
  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetTempSize(d_temp_storage, inputs, outputs, num);

  at::Tensor temp_storage = at::zeros({(long)temp_storage_bytes}, Inputs.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, inputs, outputs, num);
  CubDebugExit(cudaGetLastError());

  return Outputs;
}



__global__ void d_TrueToZDepth (
    const uint32_t num, float* dmap, 
    const uint32_t height, const uint32_t width,
    const float fx, const float fy, const float cx, const float cy, 
    float maxdepth)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= num) return;

  uint x = tidx % width;
  uint y = tidx / width;

  float u = (x - cx)/fx;
  float v = (y - cy)/fy;

  float d = dmap[y*width+x];
  float l = rsqrtf(u*u + v*v + 1.0f);
  
  dmap[y*width+x] = d*l;
}


__global__ void d_MinMaxDM (
    const uint32_t num, float* img, 
    const uint32_t height, const uint32_t width,
    float2* out)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    int x1 = tidx % width;
    int y1 = tidx / width;
    int x0 = max(x1-1, 0);
    int y0 = max(y1-1, 0);
    int x2 = min(x1+1, width-1);
    int y2 = min(y1+1, height-1);

    float d00  = img[y0*width+x0];
    float d01  = img[y0*width+x1];
    float d02  = img[y0*width+x2];
    float d10  = img[y1*width+x0];
    float d11  = img[y1*width+x1];
    float d12  = img[y1*width+x2];
    float d20  = img[y2*width+x0];
    float d21  = img[y2*width+x1];
    float d22  = img[y2*width+x2];

    float z0 = fmin(fmin(fmin(fmin(d00, d01), fmin(d02, d10)),fmin(fmin(d11, d12), fmin(d20, d21))),d22);
    float z1 = fmax(fmax(fmax(fmax(d00, d01), fmax(d02, d10)),fmax(fmax(d11, d12), fmax(d20, d21))),d22);

    out[tidx] = make_float2(z0, z1);
  }
}


__global__ void d_MiddleMip2D (
    const uint num, const float2* mipin, const uint width, float2* mipout, float maxdepth)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    uint x = tidx % width;
    uint y = tidx / width;

    x *= 2; y *= 2;
    uint w = 2*width;

    float2 d00 = mipin[y*w+x];
    float2 d01 = mipin[y*w+x+1];
    float2 d10 = mipin[(y+1)*w+x];
    float2 d11 = mipin[(y+1)*w+x+1];

    float z0 = fmin(fmin(d00.x, d10.x), fmin(d01.x, d11.x));
    float z1 = fmax(fmax(d00.y, d10.y), fmax(d01.y, d11.y));

    mipout[tidx] = make_float2(z0, z1);  
  }
}


at::Tensor  build_mip2d_cuda(at::Tensor image, at::Tensor In, int mip_levels, float maxdepth, bool true_depth)
{
  int height = image.size(0);
  int width = image.size(1);

  int s = pow(2, mip_levels-1);
  int h0 = height/s;
  int w0 = width/s;
  int hw = h0*w0;
  int size = (int)(hw*(pow(4, mip_levels)-1)/3); 

  at::Tensor mipmap = at::empty({ size, 2 }, image.options());

  float* img = image.data_ptr<float>();
  float2* mip = reinterpret_cast<float2*>(mipmap.data_ptr<float>());
  float* in = In.data_ptr<float>();
  float2* mmmip = mip + (int)(hw*(pow(4, mip_levels-1)-1)/3);

  if (true_depth) // turn true depth into z-buffer depth
  {
    float fx = in[0];
    float fy = in[5];
    float cx = in[8];
    float cy = in[9];

    d_TrueToZDepth <<< (height*width + 1023)/1024, 1024 >>> 
    ( height*width,
      img, 
      height, width,
      fx, fy, cx, cy,
      maxdepth);
  }

  d_MinMaxDM <<< (height*width + 1023)/1024, 1024 >>> 
  ( height*width,
    img, 
    height, width,
    mmmip);

  for (int l = mip_levels-2; l >= 0; l--)
  {
    uint num_threads = hw*pow(4, l);
    uint64_t offset0 = (uint64_t)(hw*(pow(4, l+1)-1)/3);
    uint64_t offset1 = (uint64_t)(hw*(pow(4, l)-1)/3);

    width /= 2;
    d_MiddleMip2D <<< (num_threads + 1023)/1024, 1024 >>> 
    (
      num_threads,
      mip+offset0, 
      width,
      mip+offset1,
      maxdepth);

    AT_CUDA_CHECK(cudaGetLastError());
  }

  return mipmap.contiguous();
}


__global__ void d_Subdivide(uint num, point_data* voxelDataIn, point_data* voxelDataOut, uint* InSum, uint* out_nvsum)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    uint IdxOut = (tidx == 0u) ? 0 : InSum[tidx-1];
		if (IdxOut != InSum[tidx])
		{
      out_nvsum[IdxOut] = tidx;
			point_data Vin = voxelDataIn[tidx];
			point_data Vout = make_point_data(2 * Vin.x, 2 * Vin.y, 2 * Vin.z);

			uint IdxBase = 8 * IdxOut;

			for (uint i = 0; i < 8; i++)
			{
				voxelDataOut[IdxBase + i] = make_point_data(Vout.x + (i >> 2), Vout.y + ((i >> 1) & 0x1), Vout.z + (i & 0x1));
			}
		}
	}
}

std::vector<at::Tensor>  subdivide_cuda(at::Tensor Points, at::Tensor Insum)
{
  uint32_t num = Points.size(0);
  uint32_t pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({8*pass, 3}, Points.options());
  at::Tensor NVSum = at::zeros({pass}, Points.options().dtype(at::kInt));

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint32_t* insum = reinterpret_cast<uint32_t*>(Insum.data_ptr<int>());
  uint32_t* nvsum = reinterpret_cast<uint32_t*>(NVSum.data_ptr<int>());

  if (num != 0u)
	  d_Subdivide << <(num + 1023) / 1024, 1024 >> >(num, points, new_points, insum, nvsum);

  return { NewPoints, NVSum };
}

__global__ void d_Compactify(uint num, point_data* voxelDataIn, point_data* voxelDataOut, uint* InSum, int64_t* out_nvsum)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    uint IdxOut = (tidx == 0u) ? 0 : InSum[tidx-1];
		if (IdxOut != InSum[tidx])
		{
			voxelDataOut[IdxOut] = voxelDataIn[tidx];
      out_nvsum[IdxOut] = tidx;
		}
	}
}

std::vector<at::Tensor>  compactify_cuda(at::Tensor Points, at::Tensor Insum)
{
  uint32_t num = Points.size(0);
  uint32_t pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({pass, 3}, Points.options());
  at::Tensor NVSum = at::zeros({pass}, Points.options().dtype(at::kLong));

  point_data*  points = reinterpret_cast<point_data*>(Points.data_ptr<short>());
  point_data*  new_points = reinterpret_cast<point_data*>(NewPoints.data_ptr<short>());

  uint32_t* insum = reinterpret_cast<uint32_t*>(Insum.data_ptr<int>());
  int64_t* nvsum = reinterpret_cast<int64_t*>(NVSum.data_ptr<int64_t>());

  if (num != 0u)
	  d_Compactify << <(num + 1023) / 1024, 1024 >> >(num, points, new_points, insum, nvsum);

  return { NewPoints, NVSum };
}

}  // namespace kaolin

