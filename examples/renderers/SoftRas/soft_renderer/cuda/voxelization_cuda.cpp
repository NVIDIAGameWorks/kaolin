// Copyright (c) 2017 Hiroharu Kato
// Copyright (c) 2018 Nikos Kolotouros
// Copyright (c) 2019 Shichen Liu

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

std::vector<at::Tensor> voxelize_sub1_cuda(
        at::Tensor faces,
        at::Tensor voxels);


std::vector<at::Tensor> voxelize_sub2_cuda(
        at::Tensor faces,
        at::Tensor voxels);


std::vector<at::Tensor> voxelize_sub3_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);


std::vector<at::Tensor> voxelize_sub4_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);



// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> voxelize_sub1(
        at::Tensor faces,
        at::Tensor voxels) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);

    return voxelize_sub1_cuda(faces, voxels);
}

std::vector<at::Tensor> voxelize_sub2(
        at::Tensor faces,
        at::Tensor voxels) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);

    return voxelize_sub2_cuda(faces, voxels);
}

std::vector<at::Tensor> voxelize_sub3(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);

    return voxelize_sub3_cuda(faces, voxels, visible);
}

std::vector<at::Tensor> voxelize_sub4(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);

    return voxelize_sub4_cuda(faces, voxels, visible);
}


PYBIND11_MODULE(voxelization, m) {
    m.def("voxelize_sub1", &voxelize_sub1, "VOXELIZE_SUB1 (CUDA)");
    m.def("voxelize_sub2", &voxelize_sub2, "VOXELIZE_SUB2 (CUDA)");
    m.def("voxelize_sub3", &voxelize_sub3, "VOXELIZE_SUB3 (CUDA)");
    m.def("voxelize_sub4", &voxelize_sub4, "VOXELIZE_SUB4 (CUDA)");
}
