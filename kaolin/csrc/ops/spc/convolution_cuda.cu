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


#include "../../utils.h"
#include "convolution.cuh"

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_scan.cuh>

namespace kaolin {

#define THREADS_PER_BLOCK 64

namespace minkowski {

  template <typename Dtype, typename Itype>
  void ConvolutionForwardKernelGPU(const Dtype *d_in_feat, int in_nchannel,
      Dtype *d_out_feat, int out_nchannel,
      const Dtype *d_kernel,
      const pInOutMaps<Itype> &in_map,
      const pInOutMaps<Itype> &out_map,
      int out_nrows, cublasHandle_t cuhandle,
      cudaStream_t stream);

  template <typename Dtype, typename Itype>
  void ConvolutionBackwardKernelGPU(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
      int in_nchannel, const Dtype *d_grad_out_feat,
      int out_nchannel, const Dtype *d_kernel,
      Dtype *d_grad_kernel,
      const pInOutMaps<Itype> &in_map,
      const pInOutMaps<Itype> &out_map,
      int out_nrows, cublasHandle_t cuhandle,
      cudaStream_t stream);

} //end namespace minkowski

uint GetPyramid(uint* Pyramid, int batch, int k, int level, int olevel) {
  return Pyramid[(2 * batch + k) * (olevel + 2) + level];
}

uint64_t GetStorageBytesX(void* d_temp_storage, uint* d_Info,
                       uint* d_PrefixSum, uint max_total_points) {
  uint64_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum, max_total_points));
  return temp_storage_bytes;
}

__device__ int Identify(
    const point_data   k,
    const uint       Level,
    uint*         PrefixSum,
    uchar*         Oroot,
    uint         offset) {
  int maxval = (0x1 << Level) - 1; // seems you could do this better using Morton codes
  if (k.x < 0 || k.y < 0 || k.z < 0 || k.x > maxval || k.y > maxval || k.z > maxval)
    return -1;

  int ord = 0;
  for (uint l = 0; l < Level; l++) {
    uint depth = Level - l - 1;
    uint mask = (0x1 << depth);
    uint child_idx = ((mask&k.x) << 2 | (mask&k.y) << 1 | (mask&k.z)) >> depth;
    uchar bits = Oroot[ord];

    // if bit set, keep going
    if (bits&(0x1 << child_idx)) {
      // count set bits up to child - inclusive sum
      uint cnt = __popc(bits&((0x2 << child_idx) - 1));
      ord = PrefixSum[ord] + cnt;

      if (depth == 0)
        return ord - offset;
    }
    else
      return -1;
  }

  return ord; // only if called with Level=0
}

__global__ void GenerateKernelMap(
    const uint num,
    const point_data* Pdata,
    int*  Inmap,
    int*  Outmap,
    uint*  Info,
    const uint K, const point_data* Kvec,
    const int scale,
    uchar* Oroot, uint* PrefixSum,
    uint level, uint offset) {
  int o_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (o_idx < num)
  {
    point_data V = mul_point_data(scale, Pdata[o_idx]);
    Outmap[o_idx] = o_idx;

    for (int k = 0; k < K; k++)
    {
      int i_idx = Identify(add_point_data(V, Kvec[k]), level, PrefixSum, Oroot, offset);
      Inmap[k*num + o_idx] = i_idx;
      Info[k*num + o_idx] = i_idx == -1 ? 0 : 1;
    }
  }
}


__global__ void GenerateKernelMapTrans(
    const uint num,
    const point_data* Pdata,
    int*  Inmap,
    int*  Outmap,
    uint*  Info,
    const uint K, const point_data* Kvec,
    const int scale,
    uchar* Oroot, uint* PrefixSum,
    uint level, uint offset) {
  int o_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (o_idx < num) {
    point_data V = Pdata[o_idx];
    Outmap[o_idx] = o_idx;

    for (int k = 0; k < K; k++) {
      point_data U = sub_point_data(V, Kvec[k]);

      if (U.x%scale == 0 && U.y%scale == 0 && U.z%scale == 0) {
        int i_idx = Identify(div_point_data(U, scale), level, PrefixSum, Oroot, offset);
        Inmap[k*num + o_idx] = i_idx;
        Info[k*num + o_idx] = i_idx == -1 ? 0 : 1;
      } else {
        Info[k*num + o_idx] = 0;
      }
    }
  }
}

__global__ void CompactifyMaps(
    const uint Psize,
    const uint num,
    const int *Inmap,
    const int *Outmap,
    int *InmapX,
    int *OutmapX,
    const uint *Info,
    const uint *PrefixSum) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < Psize)	{
		if (Info[tidx] != 0) {
			uint IdxOut = PrefixSum[tidx] - 1;
			InmapX[IdxOut] = Inmap[tidx];
			OutmapX[IdxOut] = Outmap[tidx % num];
		}
	}
}

void ProcessKernelMaps(
    uint K,
    uint Cnt,
    pInOutMaps<int32_t> &in_map,
    pInOutMaps<int32_t> &out_map,
    uint* Info,
    uint* PSum,
    void* d_temp_storageA,
    size_t temp_storage_bytesA,
    int* Inmap,
    int* Outmap,
    int* InmapX,
    int* OutmapX) {
  cub::DeviceScan::InclusiveSum(d_temp_storageA, temp_storage_bytesA,
                           Info, PSum, K*Cnt);
  AT_CUDA_CHECK(cudaGetLastError());

  CompactifyMaps<<<(K*Cnt + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                   THREADS_PER_BLOCK>>>(
      K*Cnt,
      Cnt,
      Inmap,
      Outmap,
      InmapX,
      OutmapX,
      Info,
      PSum);
  AT_CUDA_CHECK(cudaGetLastError());

  in_map.clear();
  out_map.clear();
  int currSum, prevSum = 0;
  int size = 0;
  int* Ix = InmapX;
  int* Ox = OutmapX;
  for (int k = 0; k < K; k++) {
    cudaMemcpy(&currSum, PSum + (k + 1)*Cnt - 1, sizeof(int),
               cudaMemcpyDeviceToHost);

    size = currSum - prevSum;
    in_map.push_back(pVector<int>(Ix, size));
    out_map.push_back(pVector<int>(Ox, size));
    prevSum = currSum;
    Ix += size;
    Ox += size;
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

void Conv3d_forward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint*     dP,
    float*     Input, int N,
    float*     Output, int M,
    float*     Params,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     Qlevel,
    int     Olevel,
    int     BatchSize,
    uint*    Pyramid,
    uint*    d_Info,
    uint*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX) {

  pInOutMaps<int32_t>     d_inmap;
  pInOutMaps<int32_t>     d_outmap;

  float* X = Input;
  float* Y = Output;

  int Plevel = Qlevel - Jump;
  uint scale_factor = 0x1 << (Qlevel - Plevel);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint Psize = GetPyramid(Pyramid, batch, 0, Plevel, Olevel);
    uint Qsize = GetPyramid(Pyramid, batch, 0, Qlevel, Olevel);
    uint offset = GetPyramid(Pyramid, batch, 1, Qlevel, Olevel);

    GenerateKernelMap<<<(Psize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                        THREADS_PER_BLOCK>>>(
        Psize,
        d_Proot + GetPyramid(Pyramid, batch, 1, Plevel, Olevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dP, Qlevel, offset);

    AT_CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        Psize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    AT_CUDA_CHECK(cudaGetLastError());

    cublasHandle_t handle = NULL; //TODO: get from Pytorch (and stream)

    minkowski::ConvolutionForwardKernelGPU<float, int32_t>(
        X, N,// input
        Y, M,
        Params, d_inmap, d_outmap, Psize,
        handle, 0);

    AT_CUDA_CHECK(cudaGetLastError());

    X += N * Qsize;
    Y += M * Psize;

    d_Proot += GetPyramid(Pyramid, batch, 1, Olevel + 1, Olevel);
    dO += GetPyramid(Pyramid, batch, 1, Olevel, Olevel);
    dP += GetPyramid(Pyramid, batch, 1, Olevel, Olevel) + 1;
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

void Conv3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint*     dP,
    float*     Input, int N,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int M,
    float*     Params, float* Grad_Params,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     Plevel,
    int     Olevel,
    int     BatchSize,
    uint*    Pyramid,
    uint*    d_Info,
    uint*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX) {
  pInOutMaps<int32_t> d_inmap;
  pInOutMaps<int32_t> d_outmap;

  float* X = Input;

  int Qlevel = Plevel + Jump;
  TORCH_CHECK(Qlevel <= Olevel,
              "Level + jump must be lower or equal than the depth of the octree.");
  uint scale_factor = 0x1 << (Qlevel - Plevel);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint Qsize = GetPyramid(Pyramid, batch, 0, Qlevel, Olevel);
    uint Psize = GetPyramid(Pyramid, batch, 0, Plevel, Olevel);
    uint offset = GetPyramid(Pyramid, batch, 1, Plevel, Olevel);

    GenerateKernelMapTrans<<<(Qsize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                             THREADS_PER_BLOCK>>>(
        Qsize,
        d_Proot + GetPyramid(Pyramid, batch, 1, Qlevel, Olevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dP, Plevel, offset);

    AT_CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        Qsize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    cublasHandle_t handle = NULL; //TODO: get from Pytorch (and stream)

    minkowski::ConvolutionBackwardKernelGPU<float, int32_t>(
        X, Grad_Inputs, N,
        Grad_Outputs, M,
        Params, Grad_Params,
        d_outmap, d_inmap, Psize, // note the swapping of i/o maps
        handle, 0);
    AT_CUDA_CHECK(cudaGetLastError());

    X += N * Qsize;
    Grad_Inputs += N * Qsize;
    Grad_Outputs += M * Psize;
    d_Proot += GetPyramid(Pyramid, batch, 1, Olevel + 1, Olevel);
    dO += GetPyramid(Pyramid, batch, 1, Olevel, Olevel);
    dP += GetPyramid(Pyramid, batch, 1, Olevel, Olevel) + 1;
  }
}

void ConvTranspose3d_forward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint*     dP,
    float*     Input, int N,
    float*     Output, int M,
    float*     Params,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     Qlevel,
    int     Olevel,
    int     BatchSize,
    uint*    Pyramid,
    uint*    d_Info,
    uint*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX) {
  pInOutMaps<int32_t>     d_inmap;
  pInOutMaps<int32_t>     d_outmap;

  float* X = Input;
  float* Y = Output;

  int Plevel = Qlevel + Jump;
  TORCH_CHECK(Plevel <= Olevel,
              "Level + jump must be lower or equal than the depth of the octree.");

  uint scale_factor = 0x1 << (Plevel - Qlevel);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint Qsize = GetPyramid(Pyramid, batch, 0, Qlevel, Olevel);
    uint Psize = GetPyramid(Pyramid, batch, 0, Plevel, Olevel);
    uint offset = GetPyramid(Pyramid, batch, 1, Qlevel, Olevel);

    GenerateKernelMapTrans<<<(Psize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                             THREADS_PER_BLOCK>>>(
        Psize,
        d_Proot + GetPyramid(Pyramid, batch, 1, Plevel, Olevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dP, Qlevel, offset);
    AT_CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        Psize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    cublasHandle_t handle = NULL; //TODO: get from Pytorch (and stream)

    minkowski::ConvolutionForwardKernelGPU<float, int32_t>(
        X, N,// input
        Y, M,
        Params, d_inmap, d_outmap, Psize,
        handle, 0);
    AT_CUDA_CHECK(cudaGetLastError());

    d_Proot += GetPyramid(Pyramid, batch, 1, Olevel + 1, Olevel);
    X += N * Qsize;
    Y += M * Psize;
    dO += GetPyramid(Pyramid, batch, 1, Olevel, Olevel);
    dP += GetPyramid(Pyramid, batch, 1, Olevel, Olevel) + 1;
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void ConvTranspose3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint*     dP,
    float*     Input, int N,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int M,
    float*     Params, float* Grad_Params,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     Plevel,
    int     Olevel,
    int     BatchSize,
    uint*    Pyramid,
    uint*    d_Info,
    uint*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX) {
  pInOutMaps<int32_t>     d_inmap;
  pInOutMaps<int32_t>     d_outmap;

  float* X = Input;

  int Qlevel = Plevel - Jump;
  TORCH_CHECK(Qlevel >= 0,
              "level - jump must be positive");

  uint scale_factor = 0x1 << (Plevel - Qlevel);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint Qsize = GetPyramid(Pyramid, batch, 0, Qlevel, Olevel);
    uint Psize = GetPyramid(Pyramid, batch, 0, Plevel, Olevel);
    uint offset = GetPyramid(Pyramid, batch, 1, Plevel, Olevel);

    GenerateKernelMap<<<(Qsize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                        THREADS_PER_BLOCK>>>(
        Qsize,
        d_Proot + GetPyramid(Pyramid, batch, 1, Qlevel, Olevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dP, Plevel, offset);
    AT_CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        Qsize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    cublasHandle_t handle = NULL; //TODO: get from Pytorch (and stream)

    minkowski::ConvolutionBackwardKernelGPU<float, int32_t>(
        X, Grad_Inputs, N,
        Grad_Outputs, M,
        Params, Grad_Params,
        d_outmap, d_inmap, Psize,
        handle, 0);
    AT_CUDA_CHECK(cudaGetLastError());

    d_Proot += GetPyramid(Pyramid, batch, 1, Olevel + 1, Olevel);
    X += N * Qsize;
    Grad_Inputs += N * Qsize;
    Grad_Outputs += M * Psize;
    dO += GetPyramid(Pyramid, batch, 1, Olevel, Olevel);
    dP += GetPyramid(Pyramid, batch, 1, Olevel, Olevel) + 1;
  }
}

}  // namespace kaolin
