// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define eps 1e-15

template<typename scalar_t>
__global__ void dr_cuda_forward_render_batch(
		const scalar_t* __restrict__ points3d_bxfx9,
		const scalar_t* __restrict__ points2d_bxfx6,
		const scalar_t* __restrict__ pointsdirect_bxfx1,
		const scalar_t* __restrict__ pointsbbox_bxfx4,
		const scalar_t* __restrict__ features_bxfx3d,
		scalar_t* __restrict__ imidx_bxhxwx1,
		scalar_t* __restrict__ imdep_bxhxwx1,
		scalar_t* __restrict__ imwei_bxhxwx3, scalar_t* __restrict__ im_bxhxwxd,
		int bnum, int height, int width, int fnum, int dnum, int multiplier) {

	// bidx * height * width + heiidx * width + wididx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

	int wididx = presentthread % width;
	presentthread = (presentthread - wididx) / width;

	int heiidx = presentthread % height;
	int bidx = (presentthread - heiidx) / height;

	if (bidx >= bnum || heiidx >= height || wididx >= width) {
		return;
	}

	/////////////////////////////////////////////////////////////////
	// which pixel it belongs to
	const int totalidx1 = bidx * height * width + heiidx * width + wididx;
	const int totalidx3 = totalidx1 * 3;
	const int totalidxd = totalidx1 * dnum;

	// pixel coordinate
	// scalar_t x0 = 1.0 * (wididx + 0.5) / width * 2 - 1;
	// scalar_t y0 = -(1.0 * (heiidx + 0.5) / height * 2 - 1);
	scalar_t x0 = 1.0 * multiplier / width * (2 * wididx + 1 - width);
	scalar_t y0 = 1.0 * multiplier / height * (height - 2 * heiidx - 1);

	////////////////////////////////////////////////////////////////////////
	for (int fidxint = 0; fidxint < fnum; fidxint++) {

		// which face it belongs to
		const int shift1 = bidx * fnum + fidxint;
		const int shift4 = shift1 * 4;
		const int shift6 = shift1 * 6;
		const int shift9 = shift1 * 9;
		const int shift3d = shift1 * 3 * dnum;

		// is this face visible?
		scalar_t direction = pointsdirect_bxfx1[shift1];
		if (direction < 0) {
			continue;
		}

		///////////////////////////////////////////////////////////////
		// will this pixel is influenced by this face?
		scalar_t xmin = pointsbbox_bxfx4[shift4 + 0];
		scalar_t ymin = pointsbbox_bxfx4[shift4 + 1];
		scalar_t xmax = pointsbbox_bxfx4[shift4 + 2];
		scalar_t ymax = pointsbbox_bxfx4[shift4 + 3];

		// not covered by this face!
		if (x0 < xmin || x0 >= xmax || y0 < ymin || y0 >= ymax) {
			continue;
		}

		//////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		// if this pixel is covered by this face, then we check its depth and weights
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

		scalar_t w1 = k1 / (k3 + eps);
		scalar_t w2 = k2 / (k3 + eps);
		scalar_t w0 = 1 - w1 - w2;

		// not lie in the triangle
		if (w0 < 0 || w1 < 0 || w2 < 0) {
			continue;
		}

		//////////////////////////////////////////////////////////////////////////////////////
		// if it is perspective, then this way has a little error
		// because face plane may not be parallel to the image plane
		// but let's ignore it first
		scalar_t az = points3d_bxfx9[shift9 + 2];
		scalar_t bz = points3d_bxfx9[shift9 + 5];
		scalar_t cz = points3d_bxfx9[shift9 + 8];

		scalar_t z0 = w0 * az + w1 * bz + w2 * cz;
		scalar_t znow = imdep_bxhxwx1[totalidx1];

		// it will be filled by a nearer face
		if (z0 <= znow) {
			continue;
		}

		///////////////////////////////////////////////////////////
		// update it!
		// depth
		imdep_bxhxwx1[totalidx1] = z0;

		// index
		imidx_bxhxwx1[totalidx1] = fidxint + 1.0;

		// wei
		imwei_bxhxwx3[totalidx3 + 0] = w0;
		imwei_bxhxwx3[totalidx3 + 1] = w1;
		imwei_bxhxwx3[totalidx3 + 2] = w2;

		// color
		for (int d = 0; d < dnum; d++) {
			scalar_t r0 = features_bxfx3d[shift3d + d];
			scalar_t r1 = features_bxfx3d[shift3d + dnum + d];
			scalar_t r2 = features_bxfx3d[shift3d + dnum + dnum + d];
			im_bxhxwxd[totalidxd + d] = w0 * r0 + w1 * r1 + w2 * r2;
		}

		// care about first triangle
		// break;
		// calculate all the faces!!!
	}
}

template<typename scalar_t>
__global__ void dr_cuda_forward_prob_batch(
		const scalar_t* __restrict__ points2d_bxfx6,
		const scalar_t* __restrict__ pointsbbox2_bxfx4,
		const scalar_t* __restrict__ pointsdep_bxfx1,
		const scalar_t* __restrict__ imidx_bxhxwx1,
		scalar_t* __restrict__ probface_bxhxwxk,
		scalar_t* __restrict__ probcase_bxhxwxk,
		scalar_t* __restrict__ probdis_bxhxwxk,
		scalar_t* __restrict__ probdep_bxhxwxk,
		scalar_t* __restrict__ probacc_bxhxwxk,
		scalar_t* __restrict__ improb_bxhxwx1, int bnum, int height, int width,
		int fnum, int knum, int multiplier, int sigmainv) {

	// bidx * height * width + heiidx * width + wididx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

	int wididx = presentthread % width;
	presentthread = (presentthread - wididx) / width;

	int heiidx = presentthread % height;
	int bidx = (presentthread - heiidx) / height;

	if (bidx >= bnum || heiidx >= height || wididx >= width) {
		return;
	}

	/////////////////////////////////////////////////////////////////
	// which pixel it belongs to
	const int totalidx1 = bidx * height * width + heiidx * width + wididx;
	const int totalidxk = totalidx1 * knum;

	/////////////////////////////////////////////////////////
	// which face it belongs to?
	scalar_t fidx = imidx_bxhxwx1[totalidx1];

	// face begins from 1
	// convert it into int, use round!
	int fidxint = static_cast<int>(fidx + 0.5) - 1;

	// not covered by any faces
	// maybe we can search its neighbour
	if (fidxint >= 0) {
		improb_bxhxwx1[totalidx1] = 1.0;
	}
	////////////////////////////////////////////////////////////////////////
	//  pixels not covered by any faces
	else {

		// pixel coordinate
		// scalar_t x0 = 1.0 * (wididx + 0.5) / width * 2 - 1;
		// scalar_t y0 = -(1.0 * (heiidx + 0.5) / height * 2 - 1);
		scalar_t x0 = 1.0 * multiplier / width * (2 * wididx + 1 - width);
		scalar_t y0 = 1.0 * multiplier / height * (height - 2 * heiidx - 1);

		int fidxcover = fidxint;

		int kid = 0;

		for (int fidxint = 0; fidxint < fnum; fidxint++) {

			// which face it belongs to
			const int shift1 = bidx * fnum + fidxint;
			const int shift4 = shift1 * 4;
			const int shift6 = shift1 * 6;

			///////////////////////////////////////////////////////////////
			// will this pixel is influenced by this face?
			scalar_t xmin = pointsbbox2_bxfx4[shift4 + 0];
			scalar_t ymin = pointsbbox2_bxfx4[shift4 + 1];
			scalar_t xmax = pointsbbox2_bxfx4[shift4 + 2];
			scalar_t ymax = pointsbbox2_bxfx4[shift4 + 3];

			// not covered by this face!
			if (x0 < xmin || x0 >= xmax || y0 < ymin || y0 >= ymax) {
				continue;
			}

			//////////////////////////////////////////////////////////
			scalar_t pdis[6];

			// perdis
			for (int i = 0; i < 3; i++) {

				int pshift = shift6 + i * 2;
				scalar_t x1 = points2d_bxfx6[pshift + 0];
				scalar_t y1 = points2d_bxfx6[pshift + 1];

				int pshift2 = shift6 + ((i + 1) % 3) * 2;
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

				// is it a bad triangle?
				scalar_t x3 = B * B * x0 - A * B * y0 - A * C;
				scalar_t y3 = A * A * y0 - A * B * x0 - B * C;
				x3 = x3 / (down + eps);
				y3 = y3 / (down + eps);

				scalar_t direct = (x3 - x1) * (x3 - x2) + (y3 - y1) * (y3 - y2);

				if (direct > 0) {
					// bad triangle
					pdis[i] = 4 * multiplier * multiplier;
				} else {
					// perpendicular  distance
					pdis[i] = up * up / (down + eps);
				}
			}

			////////////////////////////////////////////////////////////
			// point distance
			for (int i = 0; i < 3; i++) {
				int pshift = shift6 + i * 2;
				scalar_t x1 = points2d_bxfx6[pshift + 0];
				scalar_t y1 = points2d_bxfx6[pshift + 1];
				pdis[i + 3] = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);
			}

			///////////////////////////////////////////////////////////////
			int edgeid = 0;
			scalar_t dissquare = pdis[0];

			for (int i = 1; i < 6; i++) {
				if (dissquare > pdis[i]) {
					dissquare = pdis[i];
					edgeid = i;
				}
			}

			//////////////////////////////////////////////////
			scalar_t z = sigmainv * dissquare / multiplier / multiplier;

			scalar_t prob = exp(-z);

			//////////////////////////////////////////////////
			probface_bxhxwxk[totalidxk + kid] = fidxint + 1.0;
			probcase_bxhxwxk[totalidxk + kid] = edgeid + 1.0;
			probdis_bxhxwxk[totalidxk + kid] = prob;
			probdep_bxhxwxk[totalidxk + kid] = probdep_bxhxwxk[shift1];
			kid++;

			if (kid >= knum)
				break;
		}

		scalar_t allprob = 1.0;
		for (int i = 0; i < kid; i++) {
			scalar_t prob = probdis_bxhxwxk[totalidxk + i];
			allprob *= (1.0 - prob);
		}

		// final result
		allprob = 1.0 - allprob;
		improb_bxhxwx1[totalidx1] = allprob;
	}
}

void dr_cuda_forward_batch(at::Tensor points3d_bxfx9, at::Tensor points2d_bxfx6,
		at::Tensor pointsdirect_bxfx1, at::Tensor pointsbbox_bxfx4,
		at::Tensor pointsbbox2_bxfx4, at::Tensor pointsdep_bxfx1,
		at::Tensor colors_bxfx3d, at::Tensor imidx_bxhxwx1,
		at::Tensor imdep_bxhxwx1, at::Tensor imwei_bxhxwx3,
		at::Tensor probface_bxhxwxk, at::Tensor probcase_bxhxwxk,
		at::Tensor probdis_bxhxwxk, at::Tensor probdep_bxhxwxk,
		at::Tensor probacc_bxhxwxk, at::Tensor im_bxhxwxd,
		at::Tensor improb_bxhxwx1, int multiplier, int sigmainv) {

	int bnum = points3d_bxfx9.size(0);
	int fnum = points3d_bxfx9.size(1);
	int height = im_bxhxwxd.size(1);
	int width = im_bxhxwxd.size(2);
	int dnum = im_bxhxwxd.size(3);

	int knum = probface_bxhxwxk.size(3);

	// for fxbxhxw image size
	const int threadnum = 1024;
	const int totalthread = bnum * height * width;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	AT_DISPATCH_FLOATING_TYPES(points3d_bxfx9.type(),
			"dr_cuda_forward_render_batch", ([&] {
				dr_cuda_forward_render_batch<scalar_t><<<blocks, threads>>>(
						points3d_bxfx9.data<scalar_t>(),
						points2d_bxfx6.data<scalar_t>(),
						pointsdirect_bxfx1.data<scalar_t>(),
						pointsbbox_bxfx4.data<scalar_t>(),
						colors_bxfx3d.data<scalar_t>(),
						imidx_bxhxwx1.data<scalar_t>(),
						imdep_bxhxwx1.data<scalar_t>(),
						imwei_bxhxwx3.data<scalar_t>(),
						im_bxhxwxd.data<scalar_t>(),
						bnum, height, width, fnum, dnum, multiplier);
			}));

	AT_DISPATCH_FLOATING_TYPES(points3d_bxfx9.type(),
			"dr_cuda_forward_prob_batch", ([&] {
				dr_cuda_forward_prob_batch<scalar_t><<<blocks, threads>>>(
						points2d_bxfx6.data<scalar_t>(),
						pointsbbox2_bxfx4.data<scalar_t>(),
						pointsdep_bxfx1.data<scalar_t>(),
						imidx_bxhxwx1.data<scalar_t>(),
						probface_bxhxwxk.data<scalar_t>(),
						probcase_bxhxwxk.data<scalar_t>(),
						probdis_bxhxwxk.data<scalar_t>(),
						probdep_bxhxwxk.data<scalar_t>(),
						probacc_bxhxwxk.data<scalar_t>(),
						improb_bxhxwx1.data<scalar_t>(),
						bnum, height, width, fnum, knum, multiplier, sigmainv);
			}));

	return;
}

