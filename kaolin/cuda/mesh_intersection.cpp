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

#include <torch/torch.h>
#include <iostream>
using namespace std;

// CUDA forward declarations
void MeshIntersectionKernelLauncher(
    const float* points,
    const float* verts_1,
    const float* verts_2,
    const float* verts_3,
    const int b, const int n,
    const int m,
    float* result);



void mesh_intersection_forward_cuda(
    const at::Tensor points, 
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor ints) 
{   
    
    MeshIntersectionKernelLauncher(points.data<float>(), verts_1.data<float>(),
                                            verts_2.data<float>(),verts_3.data<float>(),
                                            points.size(0), points.size(1), 
                                            verts_1.size(1),
                                            ints.data<float>());
 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &mesh_intersection_forward_cuda, "MeshIntersection forward (CUDA)");
}
