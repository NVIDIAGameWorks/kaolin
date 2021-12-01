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
#include <vector>

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

uint raytrace_cuda_impl(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exclusive_sum,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor nugget_buffers,
    at::Tensor depth_buffers,
    uint max_level,
    uint target_level,
    bool return_depth,
    bool with_exit);

void mark_pack_boundary_cuda_impl(
    at::Tensor pack_ids,
    at::Tensor boundaries);

void generate_primary_rays_cuda_impl(
    uint width,
    uint height,
    float4x4& tf,
    float3* ray_o,
    float3* ray_d);

uint generate_shadow_rays_cuda_impl(
  uint num,
  float3* ray_o,
  float3* ray_d,
  float3* src,
  float3* dst,
  uint* map,
  float3& light,
  float4& plane,
  uint* info,
  uint* prefix_sum);

#endif

std::vector<at::Tensor> generate_primary_rays_cuda(
    uint height, 
    uint width, 
    at::Tensor Eye, 
    at::Tensor At,
    at::Tensor Up, 
    float fov, 
    at::Tensor World) {
#ifdef WITH_CUDA
  CHECK_CPU_COORDS(Eye);
  CHECK_CPU_COORDS(At);
  CHECK_CPU_COORDS(Up);
  CHECK_CONTIGUOUS(World);
  CHECK_CPU(World);
  CHECK_SIZES(World, 4, 4);

  uint num = width * height;
  at::Tensor Org = at::zeros({num, 3}, at::device(at::kCUDA).dtype(at::kFloat));
  at::Tensor Dir = at::zeros({num, 3}, at::device(at::kCUDA).dtype(at::kFloat));
  float3* d_org = reinterpret_cast<float3*>(Org.data_ptr<float>());
  float3* d_dir = reinterpret_cast<float3*>(Dir.data_ptr<float>());

  float3 eye = *reinterpret_cast<float3*>(Eye.data_ptr<float>());
  float3 at = *reinterpret_cast<float3*>(At.data_ptr<float>());
  float3 up = *reinterpret_cast<float3*>(Up.data_ptr<float>());

  float4x4 world = *reinterpret_cast<float4x4*>(World.data_ptr<float>());

  float4x4 mWorldInv = transpose(world);

  float ar = (float)width / (float)height;
  float tanHalfFov = tanf(0.5f * fov);

  float4x4 mPvpInv = make_float4x4(
      2.0f * ar * tanHalfFov / width, 0.0f, 0.0f, 0.0f,
      0.0f, 2.0f * tanHalfFov / height, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f,
      ar * tanHalfFov * (1.0f - width) / width, tanHalfFov * (1.0f - height) / height, -1.0f, 0.0f);

  float3 z = normalize(at - eye);
  float3 x = normalize(crs3(z, up));
  float3 y = crs3(x, z);

  float4x4 mViewInv = make_float4x4(
    x.x, x.y, x.z, 0.0f,
    y.x, y.y, y.z, 0.0f,
    -z.x, -z.y, -z.z, 0.0f,
    eye.x, eye.y, eye.z, 1.0f);

  float4x4 mWVPInv = mPvpInv * mViewInv * mWorldInv;

  generate_primary_rays_cuda_impl(width, height, mWVPInv, d_org, d_dir);

  CUDA_CHECK(cudaGetLastError());

  return {Org, Dir};
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
}

std::vector<at::Tensor> raytrace_cuda(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exclusive_sum,
    at::Tensor ray_o,
    at::Tensor ray_d,
    uint target_level,
    bool return_depth,
    bool with_exit) {
#ifdef WITH_CUDA
  at::TensorArg octree_arg{octree, "octree", 1};
  at::TensorArg points_arg{points, "points", 2};
  at::TensorArg pyramid_arg{pyramid, "pyramid", 3};
  at::TensorArg exclusive_sum_arg{exclusive_sum, "exclusive_sum", 4};
  at::TensorArg ray_o_arg{ray_o, "ray_o", 5};
  at::TensorArg ray_d_arg{ray_d, "ray_d", 6};
  at::checkAllSameGPU(__func__, {octree_arg, points_arg, exclusive_sum_arg, ray_o_arg, ray_d_arg});
  at::checkAllContiguous(__func__,  {octree_arg, points_arg, exclusive_sum_arg, ray_o_arg, ray_d_arg});
  at::checkDeviceType(__func__, {pyramid}, at::DeviceType::CPU);
  
  CHECK_SHORT(points);
  at::checkDim(__func__, points_arg, 2);
  at::checkSize(__func__, points_arg, 1, 3);
  at::checkDim(__func__, pyramid_arg, 2);
  at::checkSize(__func__, pyramid_arg, 0, 2);
  uint max_level = pyramid.size(1)-2;
  TORCH_CHECK(max_level < KAOLIN_SPC_MAX_LEVELS, "SPC pyramid too big");

  uint* pyramid_ptr = (uint*)pyramid.data_ptr<int>();
  uint osize = pyramid_ptr[2*max_level+2];
  uint psize = pyramid_ptr[2*max_level+3];
  at::checkSize(__func__, octree_arg, 0, osize);
  at::checkSize(__func__, points_arg, 0, psize);
  TORCH_CHECK(pyramid_ptr[max_level+1] == 0 && pyramid_ptr[max_level+2] == 0, 
              "SPC pyramid corrupt, check if the SPC pyramid has been sliced");

  // allocate local GPU storage
  at::Tensor nuggets = at::zeros({2 * KAOLIN_SPC_MAX_POINTS, 2}, octree.options().dtype(at::kInt));

  uint depth_dim = with_exit ? 2 : 1;
  at::Tensor depth = at::zeros({2 * KAOLIN_SPC_MAX_POINTS, depth_dim}, octree.options().dtype(at::kFloat));

  // do cuda
  uint num = raytrace_cuda_impl(octree, points, pyramid, exclusive_sum, ray_o, ray_d, nuggets, depth, 
                                max_level, target_level, return_depth, with_exit);

  uint pad = ((target_level + 1) % 2) * KAOLIN_SPC_MAX_POINTS;
 
  if (return_depth) {
    return { nuggets.index({Slice(pad, pad+num)}).contiguous(),
             depth.index({Slice(pad, pad+num)}).contiguous() };
  } else {
    return { nuggets.index({Slice(pad, pad+num)}).contiguous() };
  }
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor mark_pack_boundary_cuda(
    at::Tensor pack_ids) {
#ifdef WITH_CUDA
  at::TensorArg pack_ids_arg{pack_ids, "pack_ids", 1};
  at::checkDim(__func__, pack_ids_arg, 1);
  at::checkAllSameGPU(__func__, {pack_ids_arg});
  at::checkAllContiguous(__func__,  {pack_ids_arg});
  at::checkScalarTypes(__func__, pack_ids_arg, {at::kByte, at::kChar, at::kInt, at::kLong, at::kShort});
  int num_ids = pack_ids.size(0);
  at::Tensor boundaries = at::zeros({num_ids}, pack_ids.options().dtype(at::kInt));
  mark_pack_boundary_cuda_impl(pack_ids, boundaries);
  return boundaries;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}


std::vector<at::Tensor> generate_shadow_rays_cuda(
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor light,
    at::Tensor plane) {
#ifdef WITH_CUDA
  // do some tensor hecks
  uint num = ray_d.size(0);
  // allocate local GPU storage
  at::Tensor Src = at::zeros({num, 3}, ray_o.options().dtype(at::kFloat));
  at::Tensor Dst = at::zeros({num, 3}, ray_o.options().dtype(at::kFloat));
  at::Tensor Map = at::zeros({num}, ray_o.options().dtype(at::kInt));
  at::Tensor Info = at::zeros({num}, ray_o.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({num}, ray_o.options().dtype(at::kInt));

  float3* d_org = reinterpret_cast<float3*>(ray_o.data_ptr<float>());
  float3* d_dir = reinterpret_cast<float3*>(ray_d.data_ptr<float>());

  float3 h_light = *reinterpret_cast<float3*>(light.data_ptr<float>());
  float4 h_plane = *reinterpret_cast<float4*>(plane.data_ptr<float>());

  float3* d_src = reinterpret_cast<float3*>(Src.data_ptr<float>());
  float3* d_dst = reinterpret_cast<float3*>(Dst.data_ptr<float>());
  uint* d_map = reinterpret_cast<uint*>(Map.data_ptr<int>());

  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());



  float3 light_ = make_float3(0.5f * (h_light.x + 1.0f), 0.5f * (h_light.y + 1.0f), 0.5f * (h_light.z + 1.0f));
  float4 plane_ = make_float4(2.0f * h_plane.x, 2.0f * h_plane.y, 2.0f * h_plane.z,
                              h_plane.w - h_plane.x - h_plane.y - h_plane.z);


  // do cuda
  uint cnt = generate_shadow_rays_cuda_impl(num, d_org, d_dir, d_src, d_dst, d_map, 
          light_, plane_, d_Info, d_PrefixSum);

  // assemble output tensors
  std::vector<at::Tensor> result;
  result.push_back(Src.index({Slice(None, cnt)}));
  result.push_back(Dst.index({Slice(None, cnt)}));
  result.push_back(Map.index({Slice(None, cnt)}));

  CUDA_CHECK(cudaGetLastError());

  return result;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

}  // namespace kaolin
