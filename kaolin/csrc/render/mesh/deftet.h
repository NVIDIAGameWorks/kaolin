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

#ifndef KAOLIN_RENDER_MESH_DEFTET_H_
#define KAOLIN_RENDER_MESH_DEFTET_H_

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> deftet_sparse_render_forward_cuda(
    const at::Tensor points3d,
    const at::Tensor points2d,
    const at::Tensor pointsbbox,
    const at::Tensor imcoords,
    const at::Tensor imdeprange,
    const int knum,
    const float eps);

std::vector<at::Tensor> deftet_sparse_render_backward_cuda(
    const at::Tensor grad_interpolated_features,
    const at::Tensor face_idx,
    const at::Tensor weights,
    const at::Tensor face_vertices_image,
    const at::Tensor face_features,
    const float eps);

}

#endif // KAOLIN_RENDER_MESH_DEFTET_H_

