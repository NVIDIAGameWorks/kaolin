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

#ifndef KAOLIN_RENDER_MESH_DEFTET_H_
#define KAOLIN_RENDER_MESH_DEFTET_H_

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> deftet_sparse_render_forward_cuda(
    at::Tensor points3d,
    at::Tensor points2d,
    at::Tensor pointsbbox,
    at::Tensor imcoords,
    at::Tensor imdeprange,
    int knum);

std::vector<at::Tensor> deftet_sparse_render_backward_cuda(
    at::Tensor grad_interpolated_features,
    at::Tensor face_idx,
    at::Tensor weights,
    at::Tensor face_vertices_image,
    at::Tensor face_features);

}

#endif // KAOLIN_RENDER_MESH_DEFTET_H_

