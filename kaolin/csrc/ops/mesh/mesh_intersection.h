// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_OPS_MESH_MESH_INTERSECTION_H_
#define KAOLIN_OPS_MESH_MESH_INTERSECTION_H_

#include <ATen/ATen.h>

namespace kaolin {

at::Tensor unbatched_mesh_intersection_cuda(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3);

}  // namespace kaolin

#endif  // KAOLIN_OPS_MESH_MESH_INTERSECTION_H_
