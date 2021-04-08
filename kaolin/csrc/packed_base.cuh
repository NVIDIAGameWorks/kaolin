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

#ifndef KAOLIN_PACKED_BASE_CUH_
#define KAOLIN_PACKED_BASE_CUH_

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>


namespace kaolin {

/*
 * This solution is largely inspired by
 * https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh.
 */

// max_nb_tensor and max_nb_chunk are due to CUDA kernel parameters limit (4KB)
template<typename T, int max_nb_chunk, int max_nb_tensor, int tensor_ndim,
	 int nb_cpu_data, bool use_shape, bool use_numel>
struct PackedSimpleKernelMetadata {
  unsigned int chunk_to_idx[max_nb_chunk + 1];  // chunk first idx in packed tensor
  T chunk_to_tensor_idx[max_nb_chunk];  // idx of the tensor of each chunk (with the offset)
  // We set arrays to 1 because we can't set it to 0, ugly but avoid code duplication with template specialization
  unsigned int tensor_to_shape[use_shape ? max_nb_tensor : 1][use_shape ? tensor_ndim : 1];  // shape of corresponding tensor
  unsigned int tensor_to_numel[use_numel ? max_nb_tensor : 1];  // numel of corresponding tensor
  unsigned int tensor_to_cpu_data[nb_cpu_data > 0 ? max_nb_tensor : 1][nb_cpu_data > 0 ? nb_cpu_data : 1];  // additional data on cpu
  unsigned int tensor_offset;
  unsigned int batch_size;
};

template<int max_nb_chunk, int max_nb_tensor, int tensor_ndim, int nb_cpu_data,
         bool use_shape, bool use_numel, typename ParamIdxT>
unsigned int prepare_param(
    PackedSimpleKernelMetadata<ParamIdxT, max_nb_chunk, max_nb_tensor,
                               tensor_ndim, nb_cpu_data, use_shape, use_numel>& param,
    const int64_t* shape_per_tensor,
    const std::array<int64_t*, nb_cpu_data>& cpu_data,
    const unsigned int tensor_idx) {
  unsigned int cur_numel = 1;
  if (use_shape) {
#pragma unroll
    for (int dim = 0; dim < tensor_ndim; dim++) {
      param.tensor_to_shape[tensor_idx][dim] = \
          shape_per_tensor[(tensor_idx + param.tensor_offset) * tensor_ndim + dim];
    }
  }
#pragma unroll
  for (int dim = 0; dim < tensor_ndim; dim++) {
    cur_numel *= shape_per_tensor[(tensor_idx + param.tensor_offset) * tensor_ndim + dim];
  }
  if (use_numel) {
    param.tensor_to_numel[tensor_idx] = cur_numel;
  }
#pragma unroll
  for (int i = 0; i < nb_cpu_data; i++) {
    param.tensor_to_cpu_data[tensor_idx][i] = \
        static_cast<unsigned int>(cpu_data[i][tensor_idx + param.tensor_offset]);
  }
  return cur_numel;
}

/*
 * This template function can be used to create operation over packed_tensors.
 *
 * It splits each sub tensor into chunk of similar sizes to avoid computational unbalance between threads of a cuda kernel.
 *
 * To use it you must design a cuda kernel with the following signature:
 * __global__ void cudakernel(packed_tensor, PackedSimpleKernelMetadata, Args ...args);
 * the kernel launch parameters will be `max_nb_chunk` blocks and `block_size` threads per block
 * `PackedSimpleKernelMetadata` template parameters must match `packed_tensor_launcher`'s ones
 *
 * Template arguments:
 *   - block_size: The cuda launch parameter, number of thread per block
 *   - max_nb_chunk: The number of chunk processed by a single kernel mainly limited by the cuda argument 4KB limit.
 *                     It's also the number of blocks
 *   - max_nb_tensor: The number of sub-tensor that can be processed by a single kernel,
 *                      mainly limited by the cuda argument 4KB limit, but also by the max value of ParamIdxT
 *                      to store the tensor index in `chunk_to_tensor_idx`.
 *                      Warning: Avoid having max_nb_tensor > max_value(ParamIdxT).
 *   - tensor_ndim: The number of dimension of each sub_tensor (without the last dimension).
 *   - use_shape: True if using shape of each subtensor in the cuda kernel.
 *   - use_numel: True if using the number of element of each subtensor in the cuda kernel.
 *   - ParamIdxT: Type used to store the tensor index. Warning: Avoid having max_nb_tensor > max_value(ParamIdxT).
 *   - nb_cpu_data: Number of additional cpu tensor to be shared with the cuda kernel as arrays in the param struct.
 *
 * Function arguments:
 *   - packed_tensor: Pointer on the main packed tensor which will be processed, must be on gpu.
 *   - shape_per_tensor: The associated shape_per_tensor, pointer to int64_t cpu tensor.
 *   - batch_size: The batch size.
 *   - cuda_kernel: The cuda kernel used.
 *   - chunk_size: The chunk size used.
 *   - cpu_data: An array of pointer on cpu tensor for additional data.
 *               Those data will be distributed with an offset in param.tensor_to_cpu_data.
 *   - ...args: All the additional arguments will be forwarded to the cuda kernel as is.
 *
 * For implementation example look at ./packed_sum_cuda_kernel.cu or ./tile_to_packed_kernel.cu
 */
template<int block_size, int max_nb_chunk, int max_nb_tensor,
	 int tensor_ndim, bool use_shape, bool use_numel, typename ParamIdxT,
	 int nb_cpu_data, typename ScalarT, typename F, typename ...Args>
void packed_simple_cuda_launcher(
    ScalarT* packed_tensor,
    const int64_t* shape_per_tensor,
    const unsigned int batch_size,
    F cuda_kernel,
    const unsigned int chunk_size,
    const std::array<int64_t*, nb_cpu_data>& cpu_data,  // pytorch doesn't have uint32_t type
    Args ...args) {
  using IdxT = unsigned int;
  // TODO(cfujitsang): probably make an assertion to check that sizeof(ParamIdxT) and max_nb_tensor are not incoherent
  // (for instance ParamIdx is char and max_nb_tensor is 257
  auto stream = at::cuda::getCurrentCUDAStream();
  // this is the structure containing the chunks information to be processed
  PackedSimpleKernelMetadata<ParamIdxT, max_nb_chunk, max_nb_tensor, tensor_ndim,
	                     nb_cpu_data, use_shape, use_numel> param;
  param.batch_size = batch_size;
  param.tensor_offset = 0;
  unsigned int tensor_idx = 0;
  IdxT chunk_idx = 0;

  unsigned int cur_numel = \
      prepare_param<max_nb_chunk, max_nb_tensor, tensor_ndim, nb_cpu_data, use_shape,
                    use_numel, ParamIdxT>(param, shape_per_tensor, cpu_data, 0U);
  IdxT next_first_idx = cur_numel;
  unsigned int param_idx = 0; // index in param.chunk_to*
  // We divide each subtensor into similar size chunks (as much as we can)
  // this allow each block to process the same amount of computation
  // regardless of disparity of subtensor sizes
  while (tensor_idx < batch_size - param.tensor_offset) {
    // if we filled param or we reached the tensor_idx maximum value of param.chunk_to_tensor_idx
    // we launch a kernel for the given param and prepare another one.
    if (param_idx == max_nb_chunk || tensor_idx == max_nb_tensor) {
      param.chunk_to_idx[param_idx] = chunk_idx;
      cuda_kernel<<<param_idx, block_size, 0, stream>>>(packed_tensor, param, args...);
      param_idx = 0;
      param.tensor_offset += tensor_idx;
      tensor_idx = 0;
      cur_numel = prepare_param<max_nb_chunk, max_nb_tensor, tensor_ndim, nb_cpu_data, use_shape,
                                use_numel, ParamIdxT>(param, shape_per_tensor, cpu_data, tensor_idx); 
      next_first_idx = chunk_idx + cur_numel;
    }
    param.chunk_to_idx[param_idx] = chunk_idx;
    param.chunk_to_tensor_idx[param_idx] = static_cast<ParamIdxT>(tensor_idx);
    
    chunk_idx = chunk_idx + chunk_size;
    if (chunk_idx >= next_first_idx) {
      // If the remaining part of the current subtensor is smaller than chunk_size,
      // we move the chunk index to the end of the current subtensor instead.
      // Otherwise, it would overflow on the next subtensor
      tensor_idx++;
      chunk_idx = next_first_idx;
      if (tensor_idx < max_nb_tensor) {
  	cur_numel = prepare_param<max_nb_chunk, max_nb_tensor, tensor_ndim, nb_cpu_data, use_shape,
                                  use_numel, ParamIdxT>(param, shape_per_tensor, cpu_data, tensor_idx);
        next_first_idx = chunk_idx + cur_numel;
      }
    }
    param_idx++;
  }
  param.chunk_to_idx[param_idx] = chunk_idx;
  cuda_kernel<<<param_idx, block_size, 0, stream>>>(packed_tensor, param, args...);
}

}  // namespace kaolin

#endif  // KAOLIN_PACKED_BASE_CUH_
