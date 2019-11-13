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

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
static __inline__ __device__ scalar_t mod(scalar_t x, scalar_t y) {
    if (x > 0) {
        return fmod(x,y);
    }
    else {
        return y + fmod(x,y);
    }
}

namespace {

const int REPEAT = 0;
const int MIRRORED_REPEAT = 1;
const int CLAMP_TO_EDGE = 2;
const int CLAMP_TO_BORDER = 3;

template <typename scalar_t>
__global__ void load_textures_cuda_kernel(
    const scalar_t* image,
    const int32_t* is_update,
    scalar_t* faces,
    scalar_t* __restrict__ textures, 
    int textures_size,
    int texture_size,
    int image_height,
    int image_width,
    int texture_wrapping,
    bool use_bilinear) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= textures_size / 3) {
      return;
  }
  const int ts = texture_size;
  const int fn = i / (ts * ts * ts);
  scalar_t dim0 = ((i / (ts * ts)) % ts) / (ts - 1.) ;
  scalar_t dim1 = ((i / ts) % ts) / (ts - 1.);
  scalar_t dim2 = (i % ts) / (ts - 1.);
  if (0 < dim0 + dim1 + dim2) {
      float sum = dim0 + dim1 + dim2;
      dim0 /= sum;
      dim1 /= sum;
      dim2 /= sum;
  }
  scalar_t* face = &faces[fn * 3 * 2];
  scalar_t* texture_ = &textures[i * 3];

  if (is_update[fn] != 0) {
    if (texture_wrapping == REPEAT) {
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            face[i] = mod(face[i], (scalar_t)1.);
        }
    }
    else if (texture_wrapping == MIRRORED_REPEAT) {
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            if (mod(face[i], (scalar_t)2) < 1) {
                face[i] = mod(face[i], (scalar_t)1.);
            }
            else {
                face[i] = 1 - mod(face[i], (scalar_t)1.);
            }
        }
    }
    else if (texture_wrapping == CLAMP_TO_EDGE) {
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            face[i] = max(min(face[i], (scalar_t) 1), (scalar_t) 0);
        }
    }
    const scalar_t pos_x = (
        (face[2 * 0 + 0] * dim0 + face[2 * 1 + 0] * dim1 + face[2 * 2 + 0] * dim2) * (image_width - 1));
    const scalar_t pos_y = (
        (face[2 * 0 + 1] * dim0 + face[2 * 1 + 1] * dim1 + face[2 * 2 + 1] * dim2) * (image_height - 1));
    if (use_bilinear) {
        /* bilinear sampling */
        const scalar_t weight_x1 = pos_x - (int)pos_x;
        const scalar_t weight_x0 = 1 - weight_x1;
        const scalar_t weight_y1 = pos_y - (int)pos_y;
        const scalar_t weight_y0 = 1 - weight_y1;
        for (int k = 0; k < 3; k++) {
            if (texture_wrapping != CLAMP_TO_BORDER) {
                scalar_t c = 0;
                c += image[(int)pos_y * image_width * 3 + (int)pos_x * 3 + k] * (weight_x0 * weight_y0);
                c += image[min((int)(pos_y + 1), image_height-1) * image_width * 3 + (int)pos_x * 3 + k] * (weight_x0 * weight_y1);
                c += image[(int)pos_y * image_width * 3 + min((int)pos_x + 1, image_width-1) * 3 + k] * (weight_x1 * weight_y0);
                c += image[min((int)(pos_y + 1), image_height-1) * image_width * 3 + min((int)pos_x + 1, image_width-1) * 3 + k] * (weight_x1 * weight_y1);
                texture_[k] = c;
            }
            else {
                texture_[k] = 0;
            }
        }
    } else {
        /* nearest neighbor */
        const int pos_xi = round(pos_x);
        const int pos_yi = round(pos_y);
        for (int k = 0; k < 3; k++) {
            if (texture_wrapping != CLAMP_TO_BORDER) {
                texture_[k] = image[pos_yi * image_width * 3 + pos_xi * 3 + k];
            }
            else {
                texture_[k] = 0;
            }
        }
    }
  }
}
}

at::Tensor load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update,
        int texture_wrapping,
        int use_bilinear) {
    // textures_size = size of the textures tensor
    const auto textures_size = textures.numel();
    // notice that texture_size != texture_size
    const auto texture_size = textures.size(1);
    const auto image_height = image.size(0);
    const auto image_width = image.size(1);
    
    const int threads = 1024;
    const dim3 blocks ((textures_size / 3 - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "load_textures_cuda", ([&] {
      load_textures_cuda_kernel<scalar_t><<<blocks, threads>>>(
          image.data<scalar_t>(),
          is_update.data<int32_t>(),
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          textures_size,
          texture_size,
          image_height,
          image_width,
          texture_wrapping,
          use_bilinear);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in load_textures: %s\n", cudaGetErrorString(err));
    return textures;
}
