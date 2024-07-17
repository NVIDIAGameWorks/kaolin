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

ulong GetTempSize(void* d_temp_storage, uint* d_M0, uint* d_M1, uint max_total_points)
{
    ulong    temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_M0, d_M1, max_total_points));
    return temp_storage_bytes;
}


void InclusiveSum_cuda(uint num, uint* inputs, uint* outputs, void* d_temp_storage, ulong temp_storage_bytes)
{
  DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, inputs, outputs, num);
  CubDebugExit(cudaGetLastError());
}

__global__ void d_FinalMip2D (
    const uint num, float* img, 
    const uint width,
    const float fx, const float fy, const float cx, const float cy, 
    float2* mip, float maxdepth, bool true_depth)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    uint x = tidx % width;
    uint y = tidx / width;

    x *= 2; y *= 2;
    uint w = 2*width;

    float u0 = (x - cx)/fx;
    float v0 = (y - cy)/fy;
    float u1 = (x+1 - cx)/fx;
    float v1 = (y+1 - cy)/fy;

    float d00 = img[y*w+x];
    float d01 = img[y*w+x+1];
    float d10 = img[(y+1)*w+x];
    float d11 = img[(y+1)*w+x+1];

    if (true_depth)
    {
      float l00 = rsqrtf(u0*u0 + v0*v0 + 1.0f);
      float l01 = rsqrtf(u1*u1 + v0*v0 + 1.0f);
      float l10 = rsqrtf(u0*u0 + v1*v1 + 1.0f);
      float l11 = rsqrtf(u1*u1 + v1*v1 + 1.0f);

      d00 *= d00 == maxdepth ? 1.0 : l00;
      d01 *= d01 == maxdepth ? 1.0 : l01;
      d10 *= d10 == maxdepth ? 1.0 : l10;
      d11 *= d11 == maxdepth ? 1.0 : l11;
    }

    img[y*w+x] = d00;
    img[y*w+x+1] = d01;
    img[(y+1)*w+x] = d10;
    img[(y+1)*w+x+1] = d11;

    float z0 = fmin(fmin(d00, d10), fmin(d01, d11));
    // float z1 = altmax(altmax(d00, d10, maxdepth), altmax(d01, d11, maxdepth), maxdepth);
    float z1 = fmax(fmax(d00, d10), fmax(d01, d11));

    mip[tidx] = make_float2(z0, z1);
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
    // float z1 = altmax(altmax(d00.y, d10.y, maxdepth), altmax(d01.y, d11.y, maxdepth), maxdepth);
    float z1 = fmax(fmax(d00.y, d10.y), fmax(d01.y, d11.y));

    mipout[tidx] = make_float2(z0, z1);  
  }
}

void BuildMip2D_cuda(float* img, 
                     uint width, uint mip_levels, uint hw,
                     float fx, float fy, float cx, float cy, 
                     float2* mipmap, float maxdepth, bool true_depth)
{
  uint num_threads = hw*pow(4, mip_levels-1);
  uint64_t offset = (uint64_t)(hw*(pow(4, mip_levels-1)-1)/3);

  width /= 2;
  d_FinalMip2D <<< (num_threads + 1023)/1024, 1024 >>> 
  (
    num_threads,
    img, 
    width,
    fx, fy, cx, cy,
    mipmap+offset,
    maxdepth,
    true_depth);

  AT_CUDA_CHECK(cudaGetLastError());

  for (int l = mip_levels-2; l >= 0; l--)
  {
    num_threads = hw*pow(4, l);
    uint64_t offset0 = (uint64_t)(hw*(pow(4, l+1)-1)/3);
    uint64_t offset1 = (uint64_t)(hw*(pow(4, l)-1)/3);

    width /= 2;
    d_MiddleMip2D <<< (num_threads + 1023)/1024, 1024 >>> 
    (
      num_threads,
      mipmap+offset0, 
      width,
      mipmap+offset1,
      maxdepth);

    AT_CUDA_CHECK(cudaGetLastError());
  }
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

void Subdivide_cuda(uint num, point_data* points, uint* exsum, point_data* new_points, uint* out_nvsum)
{
	if (num == 0u) return;

	d_Subdivide << <(num + 1023) / 1024, 1024 >> >(num, points, new_points, exsum, out_nvsum);
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

void Compactify_cuda(uint num, point_data* points, uint* insum, point_data* new_points, int64_t* out_nvsum)
{
	if (num == 0u) return;

	d_Compactify << <(num + 1023) / 1024, 1024 >> >(num, points, new_points, insum, out_nvsum);
}

}  // namespace kaolin

