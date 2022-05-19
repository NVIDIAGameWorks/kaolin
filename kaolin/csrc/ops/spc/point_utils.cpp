// Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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
#include "../../spc_math.h"

namespace kaolin {

#ifdef WITH_CUDA

void morton_to_points_cuda_impl(at::Tensor morton_codes, at::Tensor points);
void points_to_morton_cuda_impl(at::Tensor points, at::Tensor morton_codes);
void points_to_corners_cuda_impl(at::Tensor points, at::Tensor corners);
void interpolate_trilinear_cuda_impl(at::Tensor coords, at::Tensor pidx,
                                     at::Tensor points, at::Tensor trinkets,
                                     at::Tensor feats_in, at::Tensor feats_out, int32_t level);
void coords_to_trilinear_cuda_impl(at::Tensor coord, at::Tensor points, at::Tensor coeffs);
//void coord_to_trilinear_jacobian_cuda_impl(at::Tensor coord);

#endif // WITH_CUDA

at::Tensor morton_to_points_cuda(at::Tensor morton_codes) {
#ifdef WITH_CUDA
  at::TensorArg morton_codes_arg{morton_codes, "morton_codes", 1};
  at::checkAllSameGPU(__func__, {morton_codes_arg});
  at::checkAllContiguous(__func__, {morton_codes_arg});
  at::checkScalarType(__func__, morton_codes_arg, at::kLong);

  int64_t num_points = morton_codes.size(0);
  at::Tensor points = at::zeros({num_points, 3}, at::device(at::kCUDA).dtype(at::kShort));
  morton_to_points_cuda_impl(morton_codes, points);
  return points;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor points_to_morton_cuda(at::Tensor points) {
#ifdef WITH_CUDA
  at::TensorArg points_arg{points, "points", 1};
  at::checkAllSameGPU(__func__, {points_arg});
  at::checkAllContiguous(__func__, {points_arg});
  at::checkScalarType(__func__, points_arg, at::kShort);
  
  int64_t num_points = points.size(0);
  at::Tensor morton_codes = at::zeros({num_points}, at::device(at::kCUDA).dtype(at::kLong));
  points_to_morton_cuda_impl(points, morton_codes);
  return morton_codes;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor points_to_corners_cuda(at::Tensor points) {
#ifdef WITH_CUDA
  at::TensorArg points_arg{points, "points", 1};
  at::checkAllSameGPU(__func__, {points_arg});
  at::checkAllContiguous(__func__, {points_arg});
  at::checkScalarType(__func__, points_arg, at::kShort);
  
  int64_t num = points.size(0);
  at::Tensor corners = at::zeros({num, 8, 3}, at::device(at::kCUDA).dtype(at::kShort));
  points_to_corners_cuda_impl(points, corners);
  return corners;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor interpolate_trilinear_cuda(
        at::Tensor coords, 
        at::Tensor pidx, 
        at::Tensor points, 
        at::Tensor trinkets, 
        at::Tensor feats,
        int32_t level) {
#ifdef WITH_CUDA
  at::TensorArg coords_arg{coords, "coords", 1};
  at::TensorArg pidx_arg{pidx, "pidx", 2};
  at::TensorArg points_arg{points, "points", 3};
  at::TensorArg trinkets_arg{trinkets, "trinkets", 4};
  at::TensorArg feats_arg{feats, "feats", 5};
  at::checkAllSameGPU(__func__, {coords_arg, pidx_arg, points_arg, trinkets_arg, feats_arg});
  at::checkAllContiguous(__func__, {coords_arg, pidx_arg, points_arg, trinkets_arg, feats_arg});
  at::checkScalarType(__func__, coords_arg, at::kFloat);
  at::checkScalarType(__func__, pidx_arg, at::kInt);
  at::checkScalarType(__func__, points_arg, at::kShort);
  at::checkScalarType(__func__, trinkets_arg, at::kInt);
  at::checkScalarTypes(__func__, feats_arg, {at::kHalf, at::kFloat, at::kDouble});
  at::checkDim(__func__, coords_arg, 3);
  at::checkDim(__func__, pidx_arg, 1);
  at::checkDim(__func__, points_arg, 2);
  at::checkDim(__func__, trinkets_arg, 2);
  at::checkDim(__func__, feats_arg, 2);
  
  int64_t num_voxels = coords.size(0);
  int64_t num_samples = coords.size(1);
  int64_t feat_dim = feats.size(1);
  at::Tensor feats_out = at::zeros({num_voxels, num_samples, feat_dim}, feats.options());
  interpolate_trilinear_cuda_impl(coords, pidx, points, trinkets, feats, feats_out, level);
  return feats_out;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor coords_to_trilinear_cuda(at::Tensor coords, at::Tensor points) {
#ifdef WITH_CUDA
  at::TensorArg coords_arg{coords, "coords", 1};
  at::TensorArg points_arg{points, "points", 2};
  at::checkAllSameGPU(__func__, {coords_arg, points_arg});
  at::checkAllContiguous(__func__, {coords_arg, points_arg});
  at::checkScalarType(__func__, coords_arg, at::kFloat);
  at::checkScalarType(__func__, points_arg, at::kShort);
  
  int64_t num = coords.size(0);
  at::Tensor coeffs = at::zeros({num, 8}, at::device(at::kCUDA).dtype(at::kFloat));
  coords_to_trilinear_cuda_impl(coords, points, coeffs);
  return coeffs;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor coords_to_trilinear_jacobian_cuda(at::Tensor coords) {
#ifdef WITH_CUDA
  // TODO(ttakikawa): To implement when we implement backprop
  AT_ERROR("coords_to_trilinear_jacobian_cuda is not implemented yet");
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

} // namespace kaolin

