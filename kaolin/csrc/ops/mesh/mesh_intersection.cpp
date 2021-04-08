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

#include "../../check.h"

namespace kaolin {

#ifdef WITH_CUDA
void UnbatchedMeshIntersectionKernelLauncher(
    const float* points,
    const float* verts_1,
    const float* verts_2,
    const float* verts_3,
    const int n,
    const int m,
    float* result);
#endif


void unbatched_mesh_intersection_cuda(
    const at::Tensor points, 
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor ints) {   
    CHECK_CUDA(points);
    CHECK_CUDA(verts_1);
    CHECK_CUDA(verts_2);
    CHECK_CUDA(verts_3);
    CHECK_CUDA(ints);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(verts_1);
    CHECK_CONTIGUOUS(verts_2);
    CHECK_CONTIGUOUS(verts_3);
    CHECK_CONTIGUOUS(ints);

    TORCH_CHECK(verts_1.size(0) == verts_2.size(0), "vert_1 and verts_2 must have the same number of points.");
    TORCH_CHECK(verts_1.size(0) == verts_3.size(0), "vert_1 and verts_3 must have the same number of points.");
    TORCH_CHECK(ints.size(0) == points.size(0), "ints and points must have the same number of points.");

    TORCH_CHECK(verts_1.dim() == 2, "verts_1 must have a dimension of 2.");
    TORCH_CHECK(verts_2.dim() == 2, "verts_2 must have a dimension of 2.");
    TORCH_CHECK(verts_3.dim() == 2, "verts_3 must have a dimension of 2.");
    TORCH_CHECK(points.dim() == 2, "points must have a dimension of 2.");
    TORCH_CHECK(ints.dim() == 1, "ints must have a dimension of 1.");

    TORCH_CHECK(verts_1.size(1) == 3, "verts_1's last dimension must be 3.");
    TORCH_CHECK(verts_2.size(1) == 3, "verts_2's last dimension must be 3.");
    TORCH_CHECK(verts_3.size(1) == 3, "verts_3's last dimension must be 3.");
    TORCH_CHECK(points.size(1) == 3, "points's last dimension must be 3.");

    TORCH_CHECK(verts_1.dtype() == verts_2.dtype(), "verts_1 and verts_2's dtype must be the same.");
    TORCH_CHECK(verts_1.dtype() == verts_3.dtype(), "verts_1 and verts_3's dtype must be the same.");
    TORCH_CHECK(verts_1.dtype() == points.dtype(), "verts_1 and points's dtype must be the same.");

#ifdef WITH_CUDA    
    UnbatchedMeshIntersectionKernelLauncher(points.data<float>(), verts_1.data<float>(),
                                            verts_2.data<float>(),verts_3.data<float>(), points.size(0), 
                                            verts_1.size(0),
                                            ints.data<float>());
#else
    AT_ERROR("unbatched_mesh_intersection is not built with CUDA");
#endif 
}

}  // namespace kaolin
