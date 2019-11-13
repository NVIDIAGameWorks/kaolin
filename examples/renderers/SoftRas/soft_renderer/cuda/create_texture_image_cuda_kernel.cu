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

#include <ATen/ATen.h>

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
template<typename scalar_t>
__global__ void create_texture_image_cuda_kernel(
        const scalar_t* __restrict__ faces,
        const scalar_t* __restrict__ textures,
        scalar_t* __restrict__ image,
        size_t image_size,
        size_t num_faces,
        size_t texture_res_in,
        size_t texture_res_out,
        size_t tile_width,
        scalar_t eps) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = i % (tile_width * texture_res_out);
    const int y = i / (tile_width * texture_res_out);
    const int row = x / texture_res_out;
    const int column = y / texture_res_out;
    const int fn = row + column * tile_width;
    const int R = texture_res_in;

    // if (fn >= num_faces || (y % texture_res_out) < (x % texture_res_out))
    if (fn >= num_faces)
        return;

    const scalar_t* texture = &textures[fn * R * R * 3];
    const scalar_t* vertices = &faces[fn * 3 * 2];
    const scalar_t* p0 = &vertices[2 * 0];
    const scalar_t* p1 = &vertices[2 * 1];
    const scalar_t* p2 = &vertices[2 * 2];

    /* compute face_inv */
    scalar_t face_inv[9] = {
        p1[1] - p2[1], p2[0] - p1[0], p1[0] * p2[1] - p2[0] * p1[1],
        p2[1] - p0[1], p0[0] - p2[0], p2[0] * p0[1] - p0[0] * p2[1],
        p0[1] - p1[1], p1[0] - p0[0], p0[0] * p1[1] - p1[0] * p0[1]};
    scalar_t face_inv_denominator = (
        p2[0] * (p0[1] - p1[1]) +
        p0[0] * (p1[1] - p2[1]) +
        p1[0] * (p2[1] - p0[1]));
    for (int k = 0; k < 9; k++) face_inv[k] /= (face_inv_denominator + eps);

    /* compute w = face_inv * p */
    scalar_t w[3];
    scalar_t w_sum = 0;
    for (int k = 0; k < 3; k++) {
        w[k] = face_inv[3 * k + 0] * x + face_inv[3 * k + 1] * y + face_inv[3 * k + 2];
        w[k] = max(min(w[k], 1.), 0.);
        w_sum += w[k];
    }
    for (int k = 0; k < 3; k++)
        w[k] /= (w_sum + eps);

    const int w_x = w[0] * R;
    const int w_y = w[1] * R;
    if ((w[0] + w[1]) * R - w_x - w_y <= 1) {
        for (int k = 0; k < 3; k++)
            image[i * 3 + k] = texture[(w_y * R + w_x) * 3 + k];
    } else {
        for (int k = 0; k < 3; k++)
            image[i * 3 + k] = texture[((R - 1 - w_y) * R + (R - 1 - w_x)) * 3 + k];
    }

    // /* get texture index (scalar_t) */
    // scalar_t texture_index_scalar_t[3];
    // for (int k = 0; k < 3; k++) {
    //     scalar_t tif = w[k] * (tsi - 1);
    //     tif = max(tif, 0.);
    //     tif = min(tif, tsi - 1 - eps);
    //     texture_index_scalar_t[k] = tif;
    // }

    // /* blend */
    // scalar_t new_pixel[3] = {0, 0, 0};
    // for (int pn = 0; pn < 8; pn++) {
    //     scalar_t w = 1;                         // weight
    //     int texture_index_int[3];            // index in source (int)
    //     for (int k = 0; k < 3; k++) {
    //         if ((pn >> k) % 2 == 0) {
    //             w *= 1 - (texture_index_scalar_t[k] - (int)texture_index_scalar_t[k]);
    //             texture_index_int[k] = (int)texture_index_scalar_t[k];
    //         }
    //         else {
    //             w *= texture_index_scalar_t[k] - (int)texture_index_scalar_t[k];
    //             texture_index_int[k] = (int)texture_index_scalar_t[k] + 1;
    //         }
    //     }
    //     int isc = texture_index_int[0] * tsi * tsi + texture_index_int[1] * tsi + texture_index_int[2];
    //     for (int k = 0; k < 3; k++)
    //         new_pixel[k] += w * texture[isc * 3 + k];
    // }
    // for (int k = 0; k < 3; k++)
    //     image[i * 3 + k] = new_pixel[k];
}

// didn't really look to see if we fuse the 2 kernels
// probably not because of synchronization issues
// template<typename scalar_t>
// __global__ void create_texture_image_boundary_cuda_kernel(
//         scalar_t* image,
//         size_t image_size,
//         size_t texture_res_out,
//         size_t tile_width) {

//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= image_size / 3) {
//         return;
//     }

//     const int x = i % (tile_width * texture_res_out);
//     const int y = i / (tile_width * texture_res_out);
//     if ((y % texture_res_out + 1) == (x % texture_res_out)) {
//       for (int k = 0; k < 3; k++)
//           image[i * 3 + k] = 
//               image[(y * tile_width * texture_res_out + (x - 1))  * 3 + k];
//     }
// }

}

at::Tensor create_texture_image_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor image,
        float eps) {

    const auto num_faces = textures.size(0);
    const auto texture_res_in = int(sqrt(textures.size(1)));
    const auto tile_width = int(sqrt(num_faces - 1)) + 1;
    const auto texture_res_out = image.size(1) / tile_width;

    const int threads = 1024;
    const int image_size = image.numel();
    const dim3 blocks ((image_size / 3 - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "create_texture_image_cuda", ([&] {
      create_texture_image_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          image.data<scalar_t>(),
          image_size,
          num_faces,
          texture_res_in,
          texture_res_out,
          tile_width,
          (scalar_t) eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in create_texture_image: %s\n", cudaGetErrorString(err));

    return image;
}
