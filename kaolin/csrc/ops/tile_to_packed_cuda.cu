// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

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

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "../utils.h"
#include "../packed_base.cuh"

/*
 * This solution is largely inspired by
 * https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh.
 */

// this limit is due to CUDA kernel parameters limit (4KB)
// currently based on sizeof(long), TODO(cfujitsang): make it variable to sizeof(idx_t)
#define NB_CHUNK_LIMIT 448
#define CHUNK_SIZE 32768
#define BLOCK_SIZE 1024

namespace kaolin {

template<typename scalar_t, typename out_scalar_t, typename idx_t>
__global__ void tile_to_packed_cuda_kernel(
    const scalar_t* __restrict__ values_tensor,
    PackedSimpleKernelMetadata<unsigned char, NB_CHUNK_LIMIT, 256, 1, 0, false, false> param,
    out_scalar_t* output) {
  const idx_t chunk_start_idx = param.chunk_to_idx[blockIdx.x];
  const idx_t chunk_end_idx = param.chunk_to_idx[blockIdx.x + 1];
  const unsigned char tensor_idx = param.chunk_to_tensor_idx[blockIdx.x];

  out_scalar_t val = static_cast<out_scalar_t>(values_tensor[tensor_idx + param.tensor_offset]);

  for (idx_t i = chunk_start_idx + threadIdx.x; i < chunk_end_idx; i += blockDim.x) {
    output[i] = val;
  }
}

#define DISPATCH_INOUT_TYPES(IN_TORCH_TYPE, OUT_TORCH_TYPE, IN_TYPE_NAME, OUT_TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Half, at::ScalarType::Half, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             at::Half, at::Half, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Float, at::ScalarType::Float, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             float, float, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Double, at::ScalarType::Double, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             double, double, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Bool, at::ScalarType::Bool, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             bool, bool, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Byte, at::ScalarType::Byte, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             uint8_t, uint8_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Short, at::ScalarType::Short, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int16_t, int16_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Int, at::ScalarType::Int, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int32_t, int32_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Long, at::ScalarType::Long, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int64_t, int64_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Long, at::ScalarType::Bool, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int64_t, bool, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Long, at::ScalarType::Byte, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int64_t, uint8_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Long, at::ScalarType::Short, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int64_t, int16_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Long, at::ScalarType::Int, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             int64_t, int32_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    PRIVATE_CASE_INOUT_TYPES(at::ScalarType::Float, at::ScalarType::Half, IN_TORCH_TYPE, OUT_TORCH_TYPE, \
                             float, at::Half, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
    { \
      AT_ERROR(#SCOPE_NAME, " not implemented for inputs as '", toString(IN_TORCH_TYPE), " and ", toString(OUT_TORCH_TYPE), "'"); \
    } \
  }()

/*
 * CUDA function for tiling values to a packed tensor of last_dim = 1
 */
void tile_to_packed_cuda_kernel_launcher(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output) {
  const int batch_size = shape_per_tensor.size(0);
  assert(shape_per_tensor.scalar_type() == at::ScalarType::Long);
  DISPATCH_INOUT_TYPES(values_tensor.scalar_type(), output.scalar_type(), scalar_t, out_scalar_t, "tile_to_packed", [&] {
    auto input_ptr = values_tensor.data_ptr<scalar_t>();
    auto output_ptr = output.data_ptr<out_scalar_t>();
    packed_simple_cuda_launcher<BLOCK_SIZE, NB_CHUNK_LIMIT, 256, 1, false, false, unsigned char, 0>(
        input_ptr,
        shape_per_tensor.data_ptr<int64_t>(),
        batch_size,
        tile_to_packed_cuda_kernel<scalar_t, out_scalar_t, unsigned int>,
        CHUNK_SIZE,
        std::array<int64_t*, 0>(),
        output_ptr);
  });
  return;
}

#undef DISPATCH_INOUT_TYPES

}  // namespace kaolin
