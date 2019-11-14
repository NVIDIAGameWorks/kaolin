// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// #
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// #
//     http://www.apache.org/licenses/LICENSE-2.0
// #
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Soft Rasterizer (SoftRas)

// Copyright (c) 2017 Hiroharu Kato
// Copyright (c) 2018 Nikos Kolotouros
// Copyright (c) 2019 Shichen Liu

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// #
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// #
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <torch/torch.h>

// CUDA forward declarations

at::Tensor create_texture_image_cuda(
        at::Tensor vertices_all,
        at::Tensor textures,
        at::Tensor image,
        float eps);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor create_texture_image(
        at::Tensor vertices_all,
        at::Tensor textures,
        at::Tensor image,
        float eps) {

    CHECK_INPUT(vertices_all);
    CHECK_INPUT(textures);
    CHECK_INPUT(image);
    
    return create_texture_image_cuda(vertices_all, textures, image, eps);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(create_texture_image, m) {
    m.def("create_texture_image", &create_texture_image, "CREATE_TEXTURE_IMAGE (CUDA)");
}
