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
