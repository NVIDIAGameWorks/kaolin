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

#include <torch/extension.h>

#ifdef WITH_CUDA
// CUDA forward declarations
void triangle_distance_cuda_forward(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor dist1,
    const at::Tensor idx1,
    const at::Tensor type1);

void triangle_distance_forward(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor dist1,
    const at::Tensor idx1,
    const at::Tensor type1)
{
    triangle_distance_cuda_forward(points, verts_1, verts_2, verts_3, dist1, idx1, type1);
}

void triangle_distance_cuda_backward(
    const at::Tensor grad_output,
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor idx,
    const at::Tensor dist_type,
    const at::Tensor grad_input_p,
    const at::Tensor grad_input_v1,
    const at::Tensor grad_input_v2,
    const at::Tensor grad_input_v3);

void triangle_distance_backward(
    const at::Tensor grad_output,
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor idx,
    const at::Tensor dist_type,
    const at::Tensor grad_input_p,
    const at::Tensor grad_input_v1,
    const at::Tensor grad_input_v2,
    const at::Tensor grad_input_v3)
{
  triangle_distance_cuda_backward(grad_output, points, verts_1, verts_2, verts_3, idx, dist_type,
                                  grad_input_p, grad_input_v1, grad_input_v2, grad_input_v3);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  m.def("forward", &triangle_distance_forward, "TriangleDistance forward ");
  m.def("backward", &triangle_distance_backward, "TriangleDistance backward");
#endif
}
