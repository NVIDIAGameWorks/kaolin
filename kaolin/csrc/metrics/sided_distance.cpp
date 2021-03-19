// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// pyTorchChamferDistance components
// https://github.com/chrdiller/pyTorchChamferDistance
// 
// MIT License
// 
// Copyright (c) 2018 Christian Diller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
// SOFTWARE.

#include <torch/extension.h>

#include "../check.h"

namespace kaolin {

#ifdef WITH_CUDA
void sided_distance_forward_cuda_kernel_launcher(
    const at::Tensor p1,
    const at::Tensor p2,
    const at::Tensor dist,
    const at::Tensor idx);

void sided_distance_backward_cuda_kernel_launcher(
    torch::Tensor grad_output,
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor idx,
    torch::Tensor grad_input1,
    torch::Tensor grad_input2);

#endif  // WITH_CUDA


void sided_distance_forward_cuda(
    const at::Tensor p1,
    const at::Tensor p2,
    const at::Tensor dist,
    const at::Tensor idx) {
  CHECK_CUDA(p1);
  CHECK_CUDA(p2);
  TORCH_CHECK(p1.dim() == 3, "p1 must have a dimension of 3.");
  TORCH_CHECK(p2.dim() == 3, "p2 must have a dimension of 3.");

  TORCH_CHECK(p1.size(0) == p2.size(0), "p1 and p2's batch size must be the same.");

  TORCH_CHECK(p1.size(2) == 3, "p1's last dimension must be 3.");
  TORCH_CHECK(p2.size(2) == 3, "p2's last dimension must be 3.");

  TORCH_CHECK(p1.dtype() == p2.dtype(), "p1 and p2's dtype must be the same.");

#ifdef WITH_CUDA
  sided_distance_forward_cuda_kernel_launcher(p1, p2, dist, idx);
#else
  AT_ERROR("sided_distance_forward not built with CUDA");
#endif
}

void sided_distance_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor idx,
    torch::Tensor grad_input1,
    torch::Tensor grad_input2) {
  CHECK_CUDA(p1);
  CHECK_CUDA(p2);
#ifdef WITH_CUDA
  sided_distance_backward_cuda_kernel_launcher(grad_output, p1, p2, idx, grad_input1, grad_input2);
#else
  AT_ERROR("sided_distance_backward not built with CUDA");
#endif
}

}  // namespace kaolin
