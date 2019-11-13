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
