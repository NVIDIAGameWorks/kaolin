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

#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>

#include "../../../check.h"
#ifdef WITH_CUDA
#include "../../../spc_math.h"
#endif

namespace kaolin {

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, "input is not Nx3")
#define CHECK_PACKED_FLOAT3(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x) CHECK_TRIPLE(x)
#define CHECK_PACKED_LONG3(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x) CHECK_TRIPLE(x)

using namespace std;
using namespace at::indexing;

#ifdef WITH_CUDA
extern ulong GetStorageBytes(void* d_temp_storageA, morton_code* d_M0, morton_code* d_M1, uint max_total_points);

uint VoxelizeGPU(uint npnts, float3* Pnts, uint ntris, long3* Tris, uint Level,
  point_data* d_P, morton_code* d_M0, morton_code* d_M1,
  uint* d_Info, uint* d_PrefixSum, uint* d_info, uint* d_psum,
  float3* d_l0, float3* d_l1, float3* d_l2, float3* d_F,
  uchar* d_axis, ushort* d_W, ushort2* d_pmin,
  void* d_temp_storageA, ulong temp_storage_bytesA, uchar* d_Odata, int* d_Pyramid);

uint PointToOctree(point_data* d_points, morton_code* d_morton, uint* d_info, uint* d_psum, 
    void* d_temp_storage, ulong temp_storage_bytes, uchar* d_octree, int* h_pyramid, 
    uint psize, uint level);
#endif

at::Tensor points_to_octree(
    at::Tensor points,
    uint level) {
#ifdef WITH_CUDA
    uint psize = points.size(0);
    at::Tensor morton = at::zeros({KAOLIN_SPC_MAX_POINTS}, points.options().dtype(at::kLong));
    at::Tensor info = at::zeros({KAOLIN_SPC_MAX_POINTS}, points.options().dtype(at::kInt));
    at::Tensor psum = at::zeros({KAOLIN_SPC_MAX_POINTS}, points.options().dtype(at::kInt));
    at::Tensor octree = at::zeros({KAOLIN_SPC_MAX_OCTREE}, points.options().dtype(at::kByte));
    at::Tensor pyramid = at::zeros({2, level+2}, at::device(at::kCPU).dtype(at::kInt));
  
    point_data* d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());
    morton_code* d_morton = reinterpret_cast<morton_code*>(morton.data_ptr<long>());
    uint*  d_info = reinterpret_cast<uint*>(info.data_ptr<int>());
    uint*  d_psum = reinterpret_cast<uint*>(psum.data_ptr<int>());
    uchar* d_octree = octree.data_ptr<uchar>();
    int*  h_pyramid = pyramid.data_ptr<int>();
    void* d_temp_storage = NULL;
    ulong temp_storage_bytes = GetStorageBytes(d_temp_storage, d_morton, d_morton, KAOLIN_SPC_MAX_POINTS);
    at::Tensor temp_storage = at::zeros({(long)temp_storage_bytes}, points.options().dtype(at::kByte));
    d_temp_storage = (void*)temp_storage.data_ptr<uchar>();
    
    uint osize = PointToOctree(d_points, d_morton, d_info, d_psum, d_temp_storage, temp_storage_bytes,
            d_octree, h_pyramid, psize, level);

    return octree.index({Slice(KAOLIN_SPC_MAX_OCTREE - osize, None)});
#else
  AT_ERROR("points_to_octree not built with CUDA");
#endif
}

at::Tensor mesh_to_spc(
    at::Tensor vertices,
    at::Tensor triangles,
    uint Level) {
#ifdef WITH_CUDA
  CHECK_PACKED_FLOAT3(vertices);
  CHECK_PACKED_LONG3(triangles);

  uint npnts = vertices.size(0);
  uint ntris = triangles.size(0);

  // allocate local GPU storage
  at::Tensor P = at::zeros({KAOLIN_SPC_MAX_POINTS, 3}, vertices.options().dtype(at::kShort));

  at::Tensor M0 = at::zeros({KAOLIN_SPC_MAX_POINTS}, vertices.options().dtype(at::kLong));
  at::Tensor M1 = at::zeros({KAOLIN_SPC_MAX_POINTS}, vertices.options().dtype(at::kLong));

  at::Tensor Info = at::zeros({KAOLIN_SPC_MAX_POINTS}, vertices.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({KAOLIN_SPC_MAX_POINTS}, vertices.options().dtype(at::kInt));
  at::Tensor info = at::zeros({ntris}, vertices.options().dtype(at::kInt));
  at::Tensor psum = at::zeros({ntris}, vertices.options().dtype(at::kInt));

  at::Tensor l0 = at::zeros({3*ntris}, vertices.options().dtype(at::kFloat));
  at::Tensor l1 = at::zeros({3*ntris}, vertices.options().dtype(at::kFloat));
  at::Tensor l2 = at::zeros({3*ntris}, vertices.options().dtype(at::kFloat));
  at::Tensor F = at::zeros({3*ntris}, vertices.options().dtype(at::kFloat));

  at::Tensor axis = at::zeros({ntris}, vertices.options().dtype(at::kByte));
  at::Tensor W = at::zeros({ntris}, vertices.options().dtype(at::kShort));
  at::Tensor pmin = at::zeros({2*ntris}, vertices.options().dtype(at::kShort));

  at::Tensor Odata = at::zeros({KAOLIN_SPC_MAX_OCTREE}, vertices.options().dtype(at::kByte));
  at::Tensor Pyramid = at::zeros({2, Level+2}, at::device(at::kCPU).dtype(at::kInt));

  // get tensor data pointers
  point_data* d_P = reinterpret_cast<point_data*>(P.data_ptr<short>());
  morton_code* d_M0 = reinterpret_cast<morton_code*>(M0.data_ptr<long>());
  morton_code* d_M1 = reinterpret_cast<morton_code*>(M1.data_ptr<long>());

  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());
  uint*  d_info = reinterpret_cast<uint*>(info.data_ptr<int>());
  uint*  d_psum = reinterpret_cast<uint*>(psum.data_ptr<int>());

  float3* d_Pnts = reinterpret_cast<float3*>(vertices.data_ptr<float>());
  tri_index* d_Tris = reinterpret_cast<tri_index*>(triangles.data_ptr<long>());

  float3* d_l0 = reinterpret_cast<float3*>(l0.data_ptr<float>());
  float3* d_l1 = reinterpret_cast<float3*>(l1.data_ptr<float>());
  float3* d_l2 = reinterpret_cast<float3*>(l2.data_ptr<float>());
  float3* d_F = reinterpret_cast<float3*>(F.data_ptr<float>());

  uchar*  d_axis = axis.data_ptr<uchar>();
  ushort* d_W = reinterpret_cast<ushort*>(W.data_ptr<short>());
  ushort2* d_pmin = reinterpret_cast<ushort2*>(pmin.data_ptr<short>());

  uchar* d_Odata = Odata.data_ptr<uchar>();
  int*  h_Pyramid = Pyramid.data_ptr<int>();

  // set up memory for DeviceScan and DeviceRadixSort calls
  void* d_temp_storageA = NULL;
  ulong temp_storage_bytesA = GetStorageBytes(d_temp_storageA, d_M0, d_M1, KAOLIN_SPC_MAX_POINTS);

  at::Tensor temp_storageA = at::zeros({(long)temp_storage_bytesA}, vertices.options().dtype(at::kByte));
  d_temp_storageA = (void*)temp_storageA.data_ptr<uchar>();

  // do cuda
  uint Osize = VoxelizeGPU(npnts, d_Pnts, ntris, d_Tris, Level,
              d_P, d_M0, d_M1, d_Info, d_PrefixSum, d_info, d_psum,
              d_l0, d_l1, d_l2, d_F,
              d_axis, d_W, d_pmin, d_temp_storageA, temp_storage_bytesA, d_Odata, h_Pyramid);

  return Odata.index({Slice(KAOLIN_SPC_MAX_OCTREE - Osize, None)});
#else
  AT_ERROR("mesh_to_spc not built with CUDA");
#endif

}

}  // namespace kaolin
