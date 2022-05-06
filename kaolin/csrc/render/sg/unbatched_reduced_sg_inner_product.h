// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef KAOLIN_RENDER_SG_UNBATCHED_REDUCED_SG_INNER_PRODUCT_H_
#define KAOLIN_RENDER_SG_UNBATCHED_REDUCED_SG_INNER_PRODUCT_H_

#include <ATen/ATen.h>

namespace kaolin {

at::Tensor unbatched_reduced_sg_inner_product_forward_cuda(
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness);

std::vector<at::Tensor> unbatched_reduced_sg_inner_product_backward_cuda(
    at::Tensor grad_out,
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness);

}  // namespace kaolin

#endif  // KAOLIN_RENDER_SG_UNBATCHED_REDUCED_SG_INNER_PRODUCT_H_
