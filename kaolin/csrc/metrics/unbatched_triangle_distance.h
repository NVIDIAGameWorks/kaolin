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

#ifndef KAOLIN_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_
#define KAOLIN_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_

#include <ATen/ATen.h>

namespace kaolin {

void unbatched_triangle_distance_forward_cuda(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor face_idx,
    at::Tensor dist_type);

void unbatched_triangle_distance_backward_cuda(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points,
    at::Tensor grad_face_vertices);

}  // namespace kaolin

#endif // KAOLIN_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_
