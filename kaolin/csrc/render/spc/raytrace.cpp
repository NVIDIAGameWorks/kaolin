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

#include <ATen/ATen.h>

#include "../../check.h"

#ifdef WITH_CUDA
#include "../../utils.h"
#include "../../spc_math.h"
#endif

namespace kaolin {

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 1 && x.size(0) == 3, #x " must be a triplet")
#define CHECK_CPU_COORDS(x) CHECK_CONTIGUOUS(x); CHECK_CPU(x); CHECK_FLOAT(x); CHECK_TRIPLE(x)

using namespace std;
using namespace at::indexing;

#ifdef WITH_CUDA

uint64_t GetStorageBytes(
  void* d_temp_storage,
  uint* d_Info,
  uint* d_PrefixSum,
  uint max_total_points);

void generate_primary_rays_cuda(
    uint imageW,
    uint imageH,
    float4x4& nM,
    float3* d_org,
    float3* d_dir);

uint spc_raytrace_cuda(
    uchar* d_octree,
    uint Level,
    uint targetLevel,
    point_data* d_points,
    uint* h_pyramid,
    uint*  d_exsum,
    uint num,
    float3* d_Org,
    float3* d_Dir,
    uint2* d_Nuggets,
    uint*  d_Info,
    uint*  d_PrefixSum,
    void* d_temp_storage,
    uint64_t temp_storage_bytes);

uint remove_duplicate_rays_cuda(
  uint num,
  uint2* d_Nuggets0,
  uint2* d_Nuggets1,
  uint* d_Info,
  uint* d_PrefixSum,
  void* d_temp_storage,
  uint64_t temp_storage_bytes);

void mark_first_hit_cuda(
  uint num,
  uint2* d_Nuggets,
  uint* d_Info);

uint generate_shadow_rays_cuda(
  uint num,
  float3* org,
  float3* dir,
  float3* src,
  float3* dst,
  uint* map,
  float3& light,
  float4& plane,
  uint* info,
  uint* prefixSum,
  void* d_temp_storage,
  uint64_t temp_storage_bytes);


void ray_aabb_cuda(
    const float3* query,     // ray query array
    const float3* ray_d,     // ray direction array
    const float3* ray_inv,   // inverse ray direction array
    const int2*  nuggets,    // nugget array (ray-aabb correspondences)
    const float3* points,    // 3d coord array
    const int* info,         // binary array denoting beginning of nugget group
    const int* info_idxes,   // array of active nugget indices
    const float r,           // radius of aabb
    const bool init,         // first run?
    float* d,                // distance
    bool* cond,              // true if hit
    int* pidx,               // index of 3d coord array
    const int num_nuggets,   // # of nugget indices
    const int n);            // # of active nugget indices

#endif

std::vector<at::Tensor> generate_primary_rays(
    uint imageH, uint imageW, at::Tensor Eye, at::Tensor At,
    at::Tensor Up, float fov, at::Tensor World) {
#ifdef WITH_CUDA
  CHECK_CPU_COORDS(Eye);
  CHECK_CPU_COORDS(At);
  CHECK_CPU_COORDS(Up);
  CHECK_CONTIGUOUS(World);
  CHECK_CPU(World);
  CHECK_SIZES(World, 4, 4);

  uint num = imageW * imageH;
  at::Tensor Org = at::zeros({num, 3}, at::device(at::kCUDA).dtype(at::kFloat));
  at::Tensor Dir = at::zeros({num, 3}, at::device(at::kCUDA).dtype(at::kFloat));
  float3* d_org = reinterpret_cast<float3*>(Org.data_ptr<float>());
  float3* d_dir = reinterpret_cast<float3*>(Dir.data_ptr<float>());

  float3 eye = *reinterpret_cast<float3*>(Eye.data_ptr<float>());
  float3 at = *reinterpret_cast<float3*>(At.data_ptr<float>());
  float3 up = *reinterpret_cast<float3*>(Up.data_ptr<float>());

  float4x4 world = *reinterpret_cast<float4x4*>(World.data_ptr<float>());

  float4x4 mWorldInv = transpose(world);

  float ar = (float)imageW / (float)imageH;
  float tanHalfFov = tanf(0.5f * fov);

  // version where pixel origin is upper left
  float4x4 mPvpInv = make_float4x4(
      2.0f * ar * tanHalfFov / imageW, 0.0f, 0.0f, 0.0f,
      0.0f, -2.0f * tanHalfFov / imageH, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f,
      ar * tanHalfFov * (1.0f - imageW) / imageW, tanHalfFov * (imageH - 1.0f) / imageH, -1.0f, 0.0f);

  float3 z = normalize(at - eye);
  float3 x = normalize(crs3(z, up));
  float3 y = crs3(x, z);

  float4x4 mViewInv = make_float4x4(
    x.x, x.y, x.z, 0.0f,
    y.x, y.y, y.z, 0.0f,
    -z.x, -z.y, -z.z, 0.0f,
    eye.x, eye.y, eye.z, 1.0f);

  float4x4 mCubeInv = make_float4x4(0.5f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.5f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.5f, 0.0f,
                                    0.5f, 0.5f, 0.5f, 1.0f);

  float4x4 mWVPInv = mPvpInv * mViewInv * mWorldInv * mCubeInv;

  generate_primary_rays_cuda(imageW, imageH, mWVPInv, d_org, d_dir);

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  return {Org, Dir};
#else
  AT_ERROR("generate_primary_rays not built with CUDA");
#endif
}

at::Tensor spc_raytrace(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor Org,
    at::Tensor Dir,
    uint targetLevel) {
#if WITH_CUDA
  CHECK_CUDA(octree);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(exsum);
  CHECK_CUDA(Org);
  CHECK_CUDA(Dir);
  CHECK_CONTIGUOUS(octree);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(exsum);
  CHECK_CONTIGUOUS(Org);
  CHECK_CONTIGUOUS(Dir);
  CHECK_SHORT(points);
  CHECK_DIMS(points, 2);
  CHECK_SIZE(points, 1, 3);
  TORCH_CHECK(pyramid.dim() == 2, "bad spc table0");
  TORCH_CHECK(pyramid.size(0) == 2, "bad spc table1");
  uint Level = pyramid.size(1)-2;
  TORCH_CHECK(Level < KAOLIN_SPC_MAX_LEVELS, "bad spc table2");

  uint* h_pyramid = (uint*)pyramid.data_ptr<int>();
  uint osize = h_pyramid[2*Level+2];
  uint psize = h_pyramid[2*Level+3];
  TORCH_CHECK(octree.size(0) == osize, "bad spc octree size");
  TORCH_CHECK(points.size(0) == psize, "bad spc points size");
  TORCH_CHECK(h_pyramid[Level+1] == 0 && h_pyramid[Level+2] == 0, "bad spc table3");

  //check Org and Dir better... for now
  uint num = Org.size(0);

  // allocate local GPU storage
  at::Tensor Nuggets = at::zeros({2 * KAOLIN_SPC_MAX_POINTS, 2}, octree.options().dtype(at::kInt));
  at::Tensor Info = at::zeros({KAOLIN_SPC_MAX_POINTS}, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({KAOLIN_SPC_MAX_POINTS}, octree.options().dtype(at::kInt));

  // get tensor data pointers
  float3* d_org = reinterpret_cast<float3*>(Org.data_ptr<float>());
  float3* d_dir = reinterpret_cast<float3*>(Dir.data_ptr<float>());

  uint2*  d_Nuggets = reinterpret_cast<uint2*>(Nuggets.data_ptr<int>());
  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());
  // uint*  d_D = reinterpret_cast<uint*>(D.data_ptr<int>());
  uint*  d_S = reinterpret_cast<uint*>(exsum.data_ptr<int>());
  uchar*  d_octree = octree.data_ptr<uchar>();
  point_data* d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

  // set up memory for DeviceScan calls
  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytes(
      d_temp_storage, d_Info, d_PrefixSum, KAOLIN_SPC_MAX_POINTS);
  at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  // do cuda
  num = spc_raytrace_cuda(d_octree, Level, targetLevel, d_points, h_pyramid, d_S,
                          num, d_org, d_dir, d_Nuggets, d_Info, d_PrefixSum,
                          d_temp_storage, temp_storage_bytes);


  uint pad = ((targetLevel + 1) % 2) * KAOLIN_SPC_MAX_POINTS;

  return Nuggets.index({Slice(pad, pad+num)}).contiguous();
#else
  AT_ERROR("spc_raytrace not built with CUDA");
#endif  // WITH_CUDA
}


at::Tensor remove_duplicate_rays(
    at::Tensor nuggets) {
#ifdef WITH_CUDA
  int num = nuggets.size(0);
  uint2*  d_Nuggets0 = reinterpret_cast<uint2*>(nuggets.data_ptr<int>());

  // allocate local GPU storage
  at::Tensor Nuggets = at::zeros({num, 2}, nuggets.options().dtype(at::kInt));
  at::Tensor Info = at::zeros({num}, nuggets.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({num}, nuggets.options().dtype(at::kInt));

  uint2*  d_Nuggets1 = reinterpret_cast<uint2*>(Nuggets.data_ptr<int>());
  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, num);
  at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, nuggets.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();


  uint cnt = remove_duplicate_rays_cuda(num, d_Nuggets0, d_Nuggets1, d_Info, d_PrefixSum, d_temp_storage, temp_storage_bytes);

  return Nuggets.index({Slice(None, cnt)}).contiguous();
#else
  AT_ERROR("remove_duplicate_rays not built with CUDA");
#endif  // WITH_CUDA
}

at::Tensor mark_first_hit(
    at::Tensor nuggets) {
#ifdef WITH_CUDA
  int num_nuggets = nuggets.size(0);
  at::Tensor info = at::zeros({num_nuggets}, nuggets.options().dtype(at::kInt));
  mark_first_hit_cuda(
    num_nuggets,   
    reinterpret_cast<uint2*>(nuggets.data_ptr<int>()),
    reinterpret_cast<uint*>(info.data_ptr<int>()));
  return info;
#else
  AT_ERROR("mark_first_hit not built with CUDA");
#endif  // WITH_CUDA
}


std::vector<at::Tensor> generate_shadow_rays(
    at::Tensor Org,
    at::Tensor Dir,
    at::Tensor Light,
    at::Tensor Plane) {
#ifdef WITH_CUDA
  // do some tensor hecks
  uint num = Dir.size(0);
  // allocate local GPU storage
  at::Tensor Src = at::zeros({num, 3}, Org.options().dtype(at::kFloat));
  at::Tensor Dst = at::zeros({num, 3}, Org.options().dtype(at::kFloat));
  at::Tensor Map = at::zeros({num}, Org.options().dtype(at::kInt));
  at::Tensor Info = at::zeros({num}, Org.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({num}, Org.options().dtype(at::kInt));

  float3* d_org = reinterpret_cast<float3*>(Org.data_ptr<float>());
  float3* d_dir = reinterpret_cast<float3*>(Dir.data_ptr<float>());

  float3 h_light = *reinterpret_cast<float3*>(Light.data_ptr<float>());
  float4 h_plane = *reinterpret_cast<float4*>(Plane.data_ptr<float>());

  float3* d_src = reinterpret_cast<float3*>(Src.data_ptr<float>());
  float3* d_dst = reinterpret_cast<float3*>(Dst.data_ptr<float>());
  uint* d_map = reinterpret_cast<uint*>(Map.data_ptr<int>());

  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());

  // set up memory for DeviceScan calls
  void* d_temp_storage = NULL;
  uint64_t temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, num);
  at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, Org.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();


  float3 light = make_float3(0.5f * (h_light.x + 1.0f), 0.5f * (h_light.y + 1.0f), 0.5f * (h_light.z + 1.0f));
  float4 plane = make_float4(2.0f * h_plane.x, 2.0f * h_plane.y, 2.0f * h_plane.z,
                             h_plane.w - h_plane.x - h_plane.y - h_plane.z);


  // do cuda
  uint cnt = generate_shadow_rays_cuda(num, d_org, d_dir, d_src, d_dst, d_map, light, plane, d_Info,
                                       d_PrefixSum, d_temp_storage, temp_storage_bytes);


  // assemble output tensors
  std::vector<at::Tensor> result;
  result.push_back(Src.index({Slice(None, cnt)}));
  result.push_back(Dst.index({Slice(None, cnt)}));
  result.push_back(Map.index({Slice(None, cnt)}));

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  return result;
#else
  AT_ERROR("generate_shadow_rays not built with CUDA");
#endif  // WITH_CUDA
}

std::vector<at::Tensor> spc_ray_aabb(
    at::Tensor nuggets,
    at::Tensor points,
    at::Tensor ray_query,
    at::Tensor ray_d,
    uint targetLevel,
    at::Tensor info,
    at::Tensor info_idxes,
    at::Tensor cond,
    bool init) {
#ifdef WITH_CUDA
    int nr = ray_query.size(0); // # rays
    int nn = nuggets.size(0);

    at::Tensor fpoints = points.to(at::kFloat);

    int n_iidx = info_idxes.size(0);
    at::Tensor ray_inv = 1.0 / ray_d;

    auto f_opt = at::TensorOptions().dtype(at::kFloat).device(ray_query.device());
    at::Tensor d = at::zeros({ nr, 1 }, f_opt);

    auto i_opt = at::TensorOptions().dtype(at::kInt).device(ray_query.device());
    at::Tensor pidx = at::zeros({ nr }, i_opt) - 1;
    
    int voxel_res = pow(2, targetLevel);
    float voxel_radius = (1.0 / voxel_res);

    ray_aabb_cuda(
        reinterpret_cast<float3*>(ray_query.data_ptr<float>()),
        reinterpret_cast<float3*>(ray_d.data_ptr<float>()),
        reinterpret_cast<float3*>(ray_inv.data_ptr<float>()),
        reinterpret_cast<int2*>(nuggets.data_ptr<int>()),
        reinterpret_cast<float3*>(fpoints.data_ptr<float>()),
        info.data_ptr<int>(),
        info_idxes.data_ptr<int>(),
        voxel_radius,
        init,
        d.data_ptr<float>(),
        cond.data_ptr<bool>(),
        pidx.data_ptr<int>(),
        nn,
        n_iidx);

    return { d, pidx, cond };
#else
  AT_ERROR("ray_aabb not built with CUDA");
#endif  // WITH_CUDA
}

}  // namespace kaolin
