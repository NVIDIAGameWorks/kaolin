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


#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.h"

#define eps 1e-10

namespace kaolin {

template<typename scalar_t>
__global__ void dr_cuda_backword_color_batch(
    const scalar_t* __restrict__ grad_im_bxhxwxd,
    const scalar_t* __restrict__ im_bxhxwxd,
    const int64_t* __restrict__ imidx_bxhxwx1,
    const scalar_t* __restrict__ imwei_bxhxwx3,
    const scalar_t* __restrict__ points2d_bxfx6,
    const scalar_t* __restrict__ features_bxfx3d,
    scalar_t* __restrict__ grad_points2d_bxfx6,
    scalar_t* __restrict__ grad_features_bxfx3d, int bnum, int height,
    int width, int fnum, int dnum, int multiplier) {

  int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
  int wididx = presentthread % width;
  presentthread = (presentthread - wididx) / width;
  int heiidx = presentthread % height;
  int bidx = (presentthread - heiidx) / height;

  if (bidx >= bnum || heiidx >= height || wididx >= width)
    return;

  // which pixel it belongs to
  const int totalidx1 = bidx * height * width + heiidx * width + wididx;
  const int totalidx3 = totalidx1 * 3;
  const int totalidxd = totalidx1 * dnum;

  // coordinates
  scalar_t x0 = 1.0 * multiplier / width * (2 * wididx + 1 - width);
  scalar_t y0 = 1.0 * multiplier / height * (height - 2 * heiidx - 1);

  // which face it belongs to?
  int fidxint = imidx_bxhxwx1[totalidx1];

  // visible faces
  if (fidxint >= 0) {
    const int shift1 = bidx * fnum + fidxint;
    const int shift6 = shift1 * 6;
    const int shift3d = shift1 * 3 * dnum;

    // the imaging model is:
    // I(x, y) = w0 * c0 + w1 * c1 + w2 * c2

    // gradient of colors
    // 3 points in one face
    for (int i = 0; i < 3; i++) {

      // directly use opengl weights
      scalar_t w = imwei_bxhxwx3[totalidx3 + i];
      int pointshift = shift3d + i * dnum;

      // rgb value
      for (int rgb = 0; rgb < dnum; rgb++) {
        int colorshift = pointshift + rgb;

        // this should be atomic operation
        scalar_t * addr = grad_features_bxfx3d + colorshift;
        scalar_t val = grad_im_bxhxwxd[totalidxd + rgb] * w;
        atomicAdd(addr, val);
      }
    }

    // gradient of points
    // here, we calculate dl/dp
    // dl/dp = dldI * dI/dp
    // dI/dp = c0 * dw0 / dp + c1 * dw1 / dp + c2 * dw2 / dp
    // first
    // 4 coorinates
    scalar_t ax = points2d_bxfx6[shift6 + 0];
    scalar_t ay = points2d_bxfx6[shift6 + 1];
    scalar_t bx = points2d_bxfx6[shift6 + 2];
    scalar_t by = points2d_bxfx6[shift6 + 3];
    scalar_t cx = points2d_bxfx6[shift6 + 4];
    scalar_t cy = points2d_bxfx6[shift6 + 5];

    // replace with other variables
    scalar_t m = bx - ax;
    scalar_t p = by - ay;

    scalar_t n = cx - ax;
    scalar_t q = cy - ay;

    scalar_t s = x0 - ax;
    scalar_t t = y0 - ay;

    // m* w1 + n * w2 = s
    // p * w1 + q * w2 = t
    // w1 = (sq - nt) / (mq - np)
    // w2 = (mt - sp) / (mq - np)
    scalar_t k1 = s * q - n * t;
    scalar_t k2 = m * t - s * p;
    scalar_t k3 = m * q - n * p;

    scalar_t dk1dm = 0;
    scalar_t dk1dn = -t;
    scalar_t dk1dp = 0;
    scalar_t dk1dq = s;
    scalar_t dk1ds = q;
    scalar_t dk1dt = -n;

    scalar_t dk2dm = t;
    scalar_t dk2dn = 0;
    scalar_t dk2dp = -s;
    scalar_t dk2dq = 0;
    scalar_t dk2ds = -p;
    scalar_t dk2dt = m;

    scalar_t dk3dm = q;
    scalar_t dk3dn = -p;
    scalar_t dk3dp = -n;
    scalar_t dk3dq = m;
    scalar_t dk3ds = 0;
    scalar_t dk3dt = 0;

    // w1 = k1 / k3
    // w2 = k2 / k3
    // remember we need divide k3 ^ 2
    scalar_t dw1dm = dk1dm * k3 - dk3dm * k1;
    scalar_t dw1dn = dk1dn * k3 - dk3dn * k1;
    scalar_t dw1dp = dk1dp * k3 - dk3dp * k1;
    scalar_t dw1dq = dk1dq * k3 - dk3dq * k1;
    scalar_t dw1ds = dk1ds * k3 - dk3ds * k1;
    scalar_t dw1dt = dk1dt * k3 - dk3dt * k1;

    scalar_t dw2dm = dk2dm * k3 - dk3dm * k2;
    scalar_t dw2dn = dk2dn * k3 - dk3dn * k2;
    scalar_t dw2dp = dk2dp * k3 - dk3dp * k2;
    scalar_t dw2dq = dk2dq * k3 - dk3dq * k2;
    scalar_t dw2ds = dk2ds * k3 - dk3ds * k2;
    scalar_t dw2dt = dk2dt * k3 - dk3dt * k2;

    scalar_t dw1dax = -(dw1dm + dw1dn + dw1ds);
    scalar_t dw1day = -(dw1dp + dw1dq + dw1dt);
    scalar_t dw1dbx = dw1dm;
    scalar_t dw1dby = dw1dp;
    scalar_t dw1dcx = dw1dn;
    scalar_t dw1dcy = dw1dq;

    scalar_t dw2dax = -(dw2dm + dw2dn + dw2ds);
    scalar_t dw2day = -(dw2dp + dw2dq + dw2dt);
    scalar_t dw2dbx = dw2dm;
    scalar_t dw2dby = dw2dp;
    scalar_t dw2dcx = dw2dn;
    scalar_t dw2dcy = dw2dq;

    for (int rgb = 0; rgb < dnum; rgb++) {
      // the same color for 3 points
      // thus we can simplify it
      scalar_t c0 = features_bxfx3d[shift3d + rgb];
      scalar_t c1 = features_bxfx3d[shift3d + dnum + rgb];
      scalar_t c2 = features_bxfx3d[shift3d + dnum + dnum + rgb];

      scalar_t dIdax = (c1 - c0) * dw1dax + (c2 - c0) * dw2dax;
      scalar_t dIday = (c1 - c0) * dw1day + (c2 - c0) * dw2day;
      scalar_t dIdbx = (c1 - c0) * dw1dbx + (c2 - c0) * dw2dbx;
      scalar_t dIdby = (c1 - c0) * dw1dby + (c2 - c0) * dw2dby;
      scalar_t dIdcx = (c1 - c0) * dw1dcx + (c2 - c0) * dw2dcx;
      scalar_t dIdcy = (c1 - c0) * dw1dcy + (c2 - c0) * dw2dcy;

      scalar_t dldI = multiplier * grad_im_bxhxwxd[totalidxd + rgb]
          / (k3 * k3 + eps);

      atomicAdd(grad_points2d_bxfx6 + shift6 + 0, dldI * dIdax);
      atomicAdd(grad_points2d_bxfx6 + shift6 + 1, dldI * dIday);

      atomicAdd(grad_points2d_bxfx6 + shift6 + 2, dldI * dIdbx);
      atomicAdd(grad_points2d_bxfx6 + shift6 + 3, dldI * dIdby);

      atomicAdd(grad_points2d_bxfx6 + shift6 + 4, dldI * dIdcx);
      atomicAdd(grad_points2d_bxfx6 + shift6 + 5, dldI * dIdcy);
    }
  }
}

template<typename scalar_t>
__global__ void dr_cuda_backword_prob_batch(
    const scalar_t* __restrict__ grad_improb_bxhxwx1,
    const scalar_t* __restrict__ improb_bxhxwx1,
    const int64_t* __restrict__ imidx_bxhxwx1,
    const scalar_t* __restrict__ probface_bxhxwxk,
    const scalar_t* __restrict__ probcase_bxhxwxk,
    const scalar_t* __restrict__ probdis_bxhxwxk,
    const scalar_t* __restrict__ points2d_bxfx6,
    scalar_t* __restrict__ grad_points2dprob_bxfx6, int bnum, int height,
    int width, int fnum, int knum, int multiplier, int sigmainv) {

  int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

  int wididx = presentthread % width;
  presentthread = (presentthread - wididx) / width;

  int heiidx = presentthread % height;
  int bidx = (presentthread - heiidx) / height;

  if (bidx >= bnum || heiidx >= height || wididx >= width)
    return;

  // which pixel it belongs to
  const int totalidx1 = bidx * height * width + heiidx * width + wididx;
  const int totalidxk = totalidx1 * knum;

  // coordinates
  scalar_t x0 = 1.0 * multiplier / width * (2 * wididx + 1 - width);
  scalar_t y0 = 1.0 * multiplier / height * (height - 2 * heiidx - 1);

  // which face it belongs to?
  int fidxint = imidx_bxhxwx1[totalidx1];

  // not covered by any faces
  if (fidxint < 0) {

    scalar_t dLdp = grad_improb_bxhxwx1[totalidx1];
    scalar_t allprob = improb_bxhxwx1[totalidx1];

    for (int kid = 0; kid < knum; kid++) {

      scalar_t fidx = probface_bxhxwxk[totalidxk + kid];

      // face begins from 1
      // convert it into int, use round!
      int fidxint = static_cast<int>(fidx + 0.5) - 1;
      if (fidxint < 0)
        break;

      const int shift1 = bidx * fnum + fidxint;
      const int shift6 = shift1 * 6;

      scalar_t prob = probdis_bxhxwxk[totalidxk + kid];

      scalar_t dLdz = -1.0 * sigmainv * dLdp * (1.0 - allprob)
          / (1.0 - prob + eps) * prob;

      scalar_t edgecase = probcase_bxhxwxk[totalidxk + kid];
      int edgeid = static_cast<int>(edgecase + 0.5) - 1;

      if (edgeid >= 3) {

        // point distance
        int pshift = shift6 + (edgeid - 3) * 2;
        scalar_t x1 = points2d_bxfx6[pshift + 0];
        scalar_t y1 = points2d_bxfx6[pshift + 1];

        scalar_t dLdx1 = dLdz * 2 * (x1 - x0);
        scalar_t dLdy1 = dLdz * 2 * (y1 - y0);

        atomicAdd(grad_points2dprob_bxfx6 + pshift + 0,
            dLdx1 / multiplier);
        atomicAdd(grad_points2dprob_bxfx6 + pshift + 1,
            dLdy1 / multiplier);

      } else {

        // perpendicular distance

        int pshift = shift6 + edgeid * 2;
        scalar_t x1 = points2d_bxfx6[pshift + 0];
        scalar_t y1 = points2d_bxfx6[pshift + 1];

        int pshift2 = shift6 + ((edgeid + 1) % 3) * 2;
        scalar_t x2 = points2d_bxfx6[pshift2 + 0];
        scalar_t y2 = points2d_bxfx6[pshift2 + 1];

        // ax + by + c = 0
        scalar_t A = y2 - y1;
        scalar_t B = x1 - x2;
        scalar_t C = x2 * y1 - x1 * y2;

        // dissquare = d^2 = (ax+by+c)^2 / (a^2+b^2)
        // up = ax + by + c
        // down = a^2 + b^2
        // dissquare = up^2 / down
        scalar_t up = A * x0 + B * y0 + C;
        scalar_t down = A * A + B * B;
        scalar_t dissquare = up * up / (down + eps);

        scalar_t dzdA = 2 * (x0 * up - dissquare * A) / (down + eps);
        scalar_t dzdB = 2 * (y0 * up - dissquare * B) / (down + eps);
        scalar_t dzdC = 2 * up / (down + eps);

        scalar_t dLdx1 = dLdz * (dzdB - y2 * dzdC);
        scalar_t dLdy1 = dLdz * (x2 * dzdC - dzdA);

        scalar_t dLdx2 = dLdz * (y1 * dzdC - dzdB);
        scalar_t dLdy2 = dLdz * (dzdA - x1 * dzdC);

        atomicAdd(grad_points2dprob_bxfx6 + pshift + 0,
            dLdx1 / multiplier);
        atomicAdd(grad_points2dprob_bxfx6 + pshift + 1,
            dLdy1 / multiplier);

        atomicAdd(grad_points2dprob_bxfx6 + pshift2 + 0,
            dLdx2 / multiplier);
        atomicAdd(grad_points2dprob_bxfx6 + pshift2 + 1,
            dLdy2 / multiplier);
      }
    }
  }

  return;
}

void rasterize_backward_cuda_kernel_launcher(at::Tensor grad_image_bxhxwxd,
    at::Tensor grad_improb_bxhxwx1, at::Tensor image_bxhxwxd,
    at::Tensor improb_bxhxwx1, at::Tensor imidx_bxhxwx1,
    at::Tensor imwei_bxhxwx3, at::Tensor probface_bxhxwxk,
    at::Tensor probcase_bxhxwxk, at::Tensor probdis_bxhxwxk,
    at::Tensor points2d_bxfx6, at::Tensor colors_bxfx3d,
    at::Tensor grad_points2d_bxfx6, at::Tensor grad_colors_bxfx3d,
    at::Tensor grad_points2dprob_bxfx6, int multiplier, int sigmainv) {

  int bnum = grad_image_bxhxwxd.size(0);
  int height = grad_image_bxhxwxd.size(1);
  int width = grad_image_bxhxwxd.size(2);
  int dnum = grad_image_bxhxwxd.size(3);
  int fnum = grad_points2d_bxfx6.size(1);
  int knum = probface_bxhxwxk.size(3);

  // for bxhxw image size
  const int threadnum = 512;
  const int totalthread = bnum * height * width;
  const int blocknum = totalthread / threadnum + 1;

  const dim3 threads(threadnum, 1, 1);
  const dim3 blocks(blocknum, 1, 1);

  // we exchange block and thread!
  AT_DISPATCH_FLOATING_TYPES(grad_image_bxhxwxd.scalar_type(),
      "dr_cuda_backward_color_batch", ([&] {
        dr_cuda_backword_color_batch<scalar_t><<<blocks, threads>>>(
            grad_image_bxhxwxd.data_ptr<scalar_t>(),
            image_bxhxwxd.data_ptr<scalar_t>(),
            imidx_bxhxwx1.data_ptr<int64_t>(),
            imwei_bxhxwx3.data_ptr<scalar_t>(),
            points2d_bxfx6.data_ptr<scalar_t>(),
            colors_bxfx3d.data_ptr<scalar_t>(),
            grad_points2d_bxfx6.data_ptr<scalar_t>(),
            grad_colors_bxfx3d.data_ptr<scalar_t>(),
            bnum, height, width, fnum, dnum, multiplier);
      }));

  AT_DISPATCH_FLOATING_TYPES(grad_image_bxhxwxd.scalar_type(),
      "dr_cuda_backward_prob_batch", ([&] {
        dr_cuda_backword_prob_batch<scalar_t><<<blocks, threads>>>(
            grad_improb_bxhxwx1.data_ptr<scalar_t>(),
            improb_bxhxwx1.data_ptr<scalar_t>(),
            imidx_bxhxwx1.data_ptr<int64_t>(),
            probface_bxhxwxk.data_ptr<scalar_t>(),
            probcase_bxhxwxk.data_ptr<scalar_t>(),
            probdis_bxhxwxk.data_ptr<scalar_t>(),
            points2d_bxfx6.data_ptr<scalar_t>(),
            grad_points2dprob_bxfx6.data_ptr<scalar_t>(),
            bnum, height, width, fnum, knum, multiplier, sigmainv);
      }));
}

}  // namespace kaolin
