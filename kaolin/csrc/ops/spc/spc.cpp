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
#include <stdlib.h>
#include <stdio.h>

#include "../../check.h"
#include "../../spc_math.h"

namespace kaolin {

#ifdef WITH_CUDA

at::Tensor morton_to_octree_cuda_impl(at::Tensor mortons, uint32_t level);
at::Tensor points_to_octree_cuda_impl(at::Tensor points, uint32_t level);

int scan_octrees_cuda_impl(
    at::Tensor octrees,
    at::Tensor lengths,
    at::Tensor num_childrens_per_node,
    at::Tensor prefix_sum,
    at::Tensor pyramid);

void generate_points_cuda_impl(
    at::Tensor octrees,
    at::Tensor points,
    at::Tensor morton,
    at::Tensor pyramid,
    at::Tensor prefix_sum);

#endif  // WITH_CUDA

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, #x " must be Nx3")
#define CHECK_OCTREES(x) CHECK_BYTE(x); CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_POINTS(x) CHECK_SHORT(x); CHECK_TRIPLE(x); CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace at::indexing;


at::Tensor morton_to_octree(
    at::Tensor mortons,
    uint32_t level) {
#ifdef WITH_CUDA
    
    return morton_to_octree_cuda_impl(mortons, level);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
}

at::Tensor points_to_octree(
    at::Tensor points,
    uint32_t level) {
#ifdef WITH_CUDA
    
    return points_to_octree_cuda_impl(points, level);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
}

std::tuple<int, at::Tensor, at::Tensor> scan_octrees_cuda(
    at::Tensor octrees,
    at::Tensor lengths) {
#ifdef WITH_CUDA
  CHECK_OCTREES(octrees);
  CHECK_CPU(lengths);
  CHECK_CONTIGUOUS(lengths);
  int batch_size = lengths.size(0);

  int max_num_nodes = at::max(lengths).item<int>();
  int total_num_nodes = at::sum(lengths).item<int>();

  // allocate local GPU storage
  at::Tensor num_childrens_per_node = at::zeros({ max_num_nodes },
                                                 octrees.options().dtype(at::kInt));
  at::Tensor prefix_sum = at::zeros({ total_num_nodes + batch_size },
                                    octrees.options().dtype(at::kInt));
  at::Tensor pyramid = at::zeros({ batch_size, 2, KAOLIN_SPC_MAX_LEVELS + 2 },
                                 at::device(at::kCPU).dtype(at::kInt));

  int level = scan_octrees_cuda_impl(octrees, lengths, num_childrens_per_node,
                                     prefix_sum, pyramid);
  return {level,
          pyramid.index({ Slice(None), Slice(None), Slice(None, level + 2) }).contiguous(),
          prefix_sum};
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor generate_points_cuda(
    at::Tensor octrees,
    at::Tensor pyramid,
    at::Tensor exsum) {
#ifdef WITH_CUDA
  CHECK_OCTREES(octrees);
  CHECK_CPU(pyramid);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CUDA(exsum);
  CHECK_CONTIGUOUS(exsum);

  int level = pyramid.size(2) - 2;

  int psum = pyramid.index({ Slice(None), 1, level + 1 }).sum().item<int>();
  int pmax = pyramid.index({ Slice(None), 1, level + 1 }).max().item<int>();

  // allocate local GPU storage
  at::Tensor points = at::zeros({ psum, 3 }, octrees.options().dtype(at::kShort));
  at::Tensor morton = at::zeros({ pmax }, octrees.options().dtype(at::kLong));

  generate_points_cuda_impl(octrees, points, morton, pyramid, exsum);
  return points;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

}  // namespace kaolin
