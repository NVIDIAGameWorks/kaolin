// Copyright (c) 2021,2023 NVIDIA CORPORATION & AFFILIATES.
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

#include "../../../check.h"

namespace kaolin {

#ifdef WITH_CUDA
std::vector<at::Tensor> mesh_to_spc_cuda_impl(at::Tensor face_vertices, uint32_t Level);
#endif


std::vector<at::Tensor> mesh_to_spc_cuda(
    at::Tensor face_vertices,
    uint32_t Level) {
#ifdef WITH_CUDA
  CHECK_CUDA(face_vertices);
  CHECK_CONTIGUOUS(face_vertices);
  CHECK_DIMS(face_vertices, 3);
  CHECK_SIZE(face_vertices, 1, 3);
  CHECK_SIZE(face_vertices, 2, 3);

  return mesh_to_spc_cuda_impl(face_vertices, Level);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif

}



}  // namespace kaolin
