// MIT License

// Copyright (c) 2017 Hiroharu Kato
// Copyright (c) 2018 Nikos Kolotouros
// A PyTorch implementation of Neural 3D Mesh Renderer (https://github.com/hiroharu-kato/neural_renderer)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> forward_face_index_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor face_inv_map,
        at::Tensor faces_inv,
        int image_size,
        float near,
        float far,
        int return_rgb,
        int return_alpha,
        int return_depth);

std::vector<at::Tensor> forward_texture_sampling_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        int image_size,
        float eps);

at::Tensor backward_pixel_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor rgb_map,
        at::Tensor alpha_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int image_size,
        float eps,
        int return_rgb,
        int return_alpha);

at::Tensor backward_textures_cuda(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces);

at::Tensor backward_depth_map_cuda(
        at::Tensor faces,
        at::Tensor depth_map,
        at::Tensor face_index_map,
        at::Tensor face_inv_map,
        at::Tensor weight_map,
        at::Tensor grad_depth_map,
        at::Tensor grad_faces,
        int image_size);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> forward_face_index_map(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor face_inv_map,
        at::Tensor faces_inv,
        int image_size,
        float near,
        float far,
        int return_rgb,
        int return_alpha,
        int return_depth) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(face_inv_map);
    CHECK_INPUT(faces_inv);

    return forward_face_index_map_cuda(faces, face_index_map, weight_map,
                                       depth_map, face_inv_map, faces_inv,
                                       image_size, near, far,
                                       return_rgb, return_alpha, return_depth);
}

std::vector<at::Tensor> forward_texture_sampling(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        int image_size,
        float eps) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(rgb_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(sampling_weight_map);

    return forward_texture_sampling_cuda(faces, textures, face_index_map,
                                    weight_map, depth_map, rgb_map,
                                    sampling_index_map, sampling_weight_map,
                                    image_size, eps);
}

at::Tensor backward_pixel_map(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor rgb_map,
        at::Tensor alpha_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int image_size,
        float eps,
        int return_rgb,
        int return_alpha) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(rgb_map);
    CHECK_INPUT(alpha_map);
    CHECK_INPUT(grad_rgb_map);
    CHECK_INPUT(grad_alpha_map);
    CHECK_INPUT(grad_faces);

    return backward_pixel_map_cuda(faces, face_index_map, rgb_map, alpha_map,
                                   grad_rgb_map, grad_alpha_map, grad_faces,
                                   image_size, eps, return_rgb, return_alpha);
}

at::Tensor backward_textures(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(sampling_weight_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(grad_rgb_map);
    CHECK_INPUT(grad_textures);

    return backward_textures_cuda(face_index_map, sampling_weight_map,
                                  sampling_index_map, grad_rgb_map,
                                  grad_textures, num_faces);
}

at::Tensor backward_depth_map(
        at::Tensor faces,
        at::Tensor depth_map,
        at::Tensor face_index_map,
        at::Tensor face_inv_map,
        at::Tensor weight_map,
        at::Tensor grad_depth_map,
        at::Tensor grad_faces,
        int image_size) {

    CHECK_INPUT(faces);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(face_inv_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(grad_depth_map);
    CHECK_INPUT(grad_faces);

    return backward_depth_map_cuda(faces, depth_map, face_index_map,
                                   face_inv_map, weight_map,
                                   grad_depth_map, grad_faces,
                                   image_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_face_index_map", &forward_face_index_map, "FORWARD_FACE_INDEX_MAP (CUDA)");
    m.def("forward_texture_sampling", &forward_texture_sampling, "FORWARD_TEXTURE_SAMPLING (CUDA)");
    m.def("backward_pixel_map", &backward_pixel_map, "BACKWARD_PIXEL_MAP (CUDA)");
    m.def("backward_textures", &backward_textures, "BACKWARD_TEXTURES (CUDA)");
    m.def("backward_depth_map", &backward_depth_map, "BACKWARD_DEPTH_MAP (CUDA)");
}
