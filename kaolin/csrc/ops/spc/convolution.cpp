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


#include "../../check.h"

#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

#include <ATen/ATen.h>

namespace kaolin {

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, #x " is not Nx3")

#define CHECK_OCTREES(x) CHECK_BYTE(x); CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_POINTS(x) CHECK_SHORT(x); CHECK_TRIPLE(x); CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace at::indexing;

#ifdef WITH_CUDA
uint64_t GetStorageBytesX(void* d_temp_storage, uint32_t* d_Info, uint32_t* d_PrefixSum, uint32_t max_total_points);

void Conv3d_forward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint32_t*     dP,
    float*     Input, int N,
    float*     Output, int M,
    float*     Params,
    point_data* Kvec, uint32_t Ksize,
    int     Jump,
    int     Qlevel,
    int     Olevel,
    int     BatchSize,
    uint32_t*    Pyramid,
    uint32_t*    d_Info,
    uint32_t*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX);

void Conv3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint32_t*     dP,
    float*     Input, int N,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int M,
    float*     Params, float* Grad_Params,
    point_data* Kvec, uint32_t Ksize,
    int     Jump,
    int     Plevel,
    int     Olevel,
    int     BatchSize,
    uint32_t*    Pyramid,
    uint32_t*    d_Info,
    uint32_t*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX);

void ConvTranspose3d_forward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint32_t*     dP,
    float*     Input, int N,
    float*     Output, int M,
    float*     Params,
    point_data* Kvec, uint32_t Ksize,
    int     Jump,
    int     Qlevel,
    int     Olevel,
    int     BatchSize,
    uint32_t*    Pyramid,
    uint32_t*    d_Info,
    uint32_t*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX);

void ConvTranspose3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uint32_t*     dP,
    float*     Input, int N,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int M,
    float*     Params, float* Grad_Params,
    point_data* Kvec, uint32_t Ksize,
    int     Jump,
    int     Plevel,
    int     Olevel,
    int     BatchSize,
    uint32_t*    Pyramid,
    uint32_t*    d_Info,
    uint32_t*    d_PSum,
    void*    d_temp_storageA,
    int64_t    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX);

#endif

std::tuple<at::Tensor, int> Conv3d_forward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump) {
#ifdef WITH_CUDA
  CHECK_OCTREES(octree);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);

  uint32_t kernel_vectors_size = params.size(0);
  assert(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint32_t N = params.size(1);
  assert(N == inputs.size(1));

  uint32_t M = params.size(2);

  int BatchSize = pyramid.size(0);
  uint32_t* Pyramid = reinterpret_cast<uint32_t*>(pyramid.data_ptr<int>());

  int Qlevel = level;
  int Plevel = Qlevel - jump;
  int Olevel = pyramid.size(2)-2;
  assert(PLevel >= 0);

  uint32_t psize = pyramid.index({ Slice(None), 0, Plevel }).sum().item<int>();
  int pmax = pyramid.index({ Slice(None), 0, Plevel }).max().item<int>();

  at::Tensor outputs = at::zeros({ psize, M}, octree.options().dtype(at::kFloat));

  float* Params = params.data_ptr<float>();
  float*   Y = outputs.data_ptr<float>();
  float*   X = inputs.data_ptr<float>();

  //intermediate storage
  int scan_size = kernel_vectors_size * pmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size },
                              octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ pmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  // get tensor data pointers
  uint32_t*  d_Info = reinterpret_cast<uint32_t*>(Info.data_ptr<int>());
  uint32_t*  d_PrefixSum = reinterpret_cast<uint32_t*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytesX(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (int64_t)temp_storage_bytes },
                                      octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data* d_Proot = reinterpret_cast<point_data*>(points.data_ptr<short>());
  uchar* dO = octree.data_ptr<uchar>();
  uint32_t* dEx = reinterpret_cast<uint32_t*>(exsum.data_ptr<int>());

  Conv3d_forward_cuda(
    d_Proot, dO, dEx,
    X, N, Y, M, Params, Kvec, kernel_vectors_size, jump,
    Qlevel, Olevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX);

  return std::tuple<at::Tensor, int>{outputs, Plevel};
#else
  AT_ERROR("Conv3d_forward not built with CUDA");
#endif
}


std::vector<at::Tensor> Conv3d_backward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor grad_outputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump) {
#ifdef WITH_CUDA
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);

  uint32_t kernel_vectors_size = params.size(0);
  assert(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint32_t N = params.size(1);
  assert(N == inputs.size(1));

  uint32_t M = params.size(2);

  int BatchSize = pyramid.size(0);
  int Olevel = pyramid.size(2)-2;
  uint32_t* Pyramid = reinterpret_cast<uint32_t*>(pyramid.data_ptr<int>());

  int Plevel = level;
  int Qlevel = Plevel + jump;

  at::Tensor grad_inputs = at::zeros_like(inputs);
  at::Tensor grad_params = at::zeros_like(params);

  float* Params = params.data_ptr<float>();
  float* grad_Params = grad_params.data_ptr<float>();

  float* Grad_Outputs = grad_outputs.data_ptr<float>();
  float* Grad_Inputs = grad_inputs.data_ptr<float>();
  float* X = inputs.data_ptr<float>();

  int pmax = pyramid.index({ Slice(None), 0, Qlevel }).max().item<int>();

  //intermediate storage
  int scan_size = kernel_vectors_size * pmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ pmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  // get tensor data pointers
  uint32_t*  d_Info = reinterpret_cast<uint32_t*>(Info.data_ptr<int>());
  uint32_t*  d_PrefixSum = reinterpret_cast<uint32_t*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytesX(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (int64_t)temp_storage_bytes }, octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data*  d_Proot = (point_data*)points.data_ptr<short>();
  uchar*     dO = octree.data_ptr<uchar>();
  uint32_t*     dEx = reinterpret_cast<uint32_t*>(exsum.data_ptr<int>());

  Conv3d_backward_cuda(
    d_Proot, dO, dEx,
    X, N, Grad_Inputs, Grad_Outputs, M, Params, grad_Params, Kvec, kernel_vectors_size, jump,
    Plevel, Olevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX);

  return {grad_inputs, grad_params};
#else
  AT_ERROR("Conv3d_backward not built with CUDA");
#endif
}

std::tuple<at::Tensor, int> ConvTranspose3d_forward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump) {
#if WITH_CUDA
  CHECK_OCTREES(octree);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);

  uint32_t kernel_vectors_size = params.size(0);
  assert(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint32_t N = params.size(1);
  assert(N == inputs.size(1));

  uint32_t M = params.size(2);

  // int jump = jump[0].item<int>();
  int BatchSize = pyramid.size(0);
  uint32_t* Pyramid = reinterpret_cast<uint32_t*>(pyramid.data_ptr<int>());

  int Qlevel = level;
  int Plevel = Qlevel + jump;
  int Olevel = pyramid.size(2)-2;
  assert(PLevel <= Olevel);

  uint32_t psize = pyramid.index({ Slice(None), 0, Plevel }).sum().item<int>();
  int pmax = pyramid.index({ Slice(None), 0, Plevel }).max().item<int>();

  at::Tensor outputs = at::zeros({ psize, M}, octree.options().dtype(at::kFloat));

  float* Params = params.data_ptr<float>();
  float*   Y = outputs.data_ptr<float>();
  float*   X = inputs.data_ptr<float>();

  //intermediate storage
  int scan_size = kernel_vectors_size * pmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ pmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  // get tensor data pointers
  uint32_t*  d_Info = reinterpret_cast<uint32_t*>(Info.data_ptr<int>());
  uint32_t*  d_PrefixSum = reinterpret_cast<uint32_t*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytesX(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (int64_t)temp_storage_bytes }, octree.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data*  d_Proot = (point_data*)points.data_ptr<short>();
  uchar*     dO = octree.data_ptr<uchar>();
  uint32_t*     dEx = reinterpret_cast<uint32_t*>(exsum.data_ptr<int>());

  ConvTranspose3d_forward_cuda(
    d_Proot, dO, dEx,
    X, N, Y, M, Params, Kvec, kernel_vectors_size, jump,
    Qlevel, Olevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX);

  return {outputs, Plevel};
#else
  AT_ERROR("ConvTranspose3d_forward not built with CUDA");
#endif
}

std::vector<at::Tensor>  ConvTranspose3d_backward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor grad_outputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump) {
#if WITH_CUDA
  CHECK_OCTREES(octree);
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);

  uint32_t kernel_vectors_size = params.size(0);
  assert(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint32_t N = params.size(1);
  assert(N == inputs.size(1));

  uint32_t M = params.size(2);
  int BatchSize = pyramid.size(0);
  int Olevel = pyramid.size(2)-2;
  uint32_t* Pyramid = reinterpret_cast<uint32_t*>(pyramid.data_ptr<int>());

  int Plevel = level;
  int Qlevel = Plevel - jump;

  at::Tensor grad_inputs = at::zeros_like(inputs);
  at::Tensor grad_params = at::zeros_like(params);

  float* Params = params.data_ptr<float>();
  float* grad_Params = grad_params.data_ptr<float>();

  float* Grad_Outputs = grad_outputs.data_ptr<float>();
  float* Grad_Inputs = grad_inputs.data_ptr<float>();
  float* X = inputs.data_ptr<float>();

  int pmax = pyramid.index({ Slice(None), 0, Qlevel }).max().item<int>();

  //intermediate storage
  int scan_size = kernel_vectors_size * pmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ pmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  // get tensor data pointers
  uint32_t*  d_Info = reinterpret_cast<uint32_t*>(Info.data_ptr<int>());
  uint32_t*  d_PrefixSum = reinterpret_cast<uint32_t*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytesX(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (int64_t)temp_storage_bytes }, octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data*  d_Proot = (point_data*)points.data_ptr<short>();
  uchar*     dO = octree.data_ptr<uchar>();
  uint32_t*     dEx = reinterpret_cast<uint32_t*>(exsum.data_ptr<int>());

  ConvTranspose3d_backward_cuda(
    d_Proot, dO, dEx,
    X, N, Grad_Inputs, Grad_Outputs, M, Params, grad_Params, Kvec, kernel_vectors_size, jump,
    Plevel, Olevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX);

  return {grad_inputs, grad_params};
#else
  AT_ERROR("ConvTranspose3d_backward not built with CUDA");
#endif
}

}  // namespace kaolin
