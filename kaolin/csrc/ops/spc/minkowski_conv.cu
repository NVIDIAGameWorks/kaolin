// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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


/* Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*
* Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
* Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
* of the code.
*/

// Use the torch for GPU memory management. Thrust resize gives segfulat during
// debugging -g #include <torch/extension.h>

#include "convolution.cuh"
#include "../../utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

namespace kaolin {

/**
	* Matrix multiplication (CUDA Kernel) on the device: C = A * B
	* wA is A's width and wB is B's width
	*/
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void matmul(const Dtype *A, const int wA, const int hA,
	const Dtype *B, const int wB, const int hB, Dtype *C,
	const Itype *in_map, const Itype *out_map) {
	// Use in_feat as A and kernel as B

	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	// Coordinate. x is for rows, y is for columns.
	const int x = BLOCK_SIZE * bx + tx;
	const int y = BLOCK_SIZE * by + ty;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	Dtype Csub = 0;

	const Itype in_row = y < hA ? in_map[y] : 0;
	const Itype out_row = y < hA ? out_map[y] : 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int s = 0; s < wA; s += BLOCK_SIZE) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * in_row + s + tx] : 0;
		Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : 0;

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	if (y < hA && x < wB)
		atomicAdd(&C[wB * out_row + x], Csub);
	// C[wB * out_row + x] += Csub;
}

/**
	* Matrix multiplication (CUDA Kernel) on the device: C = A * B^T, E = D^T * A
	* wA is A's width and wB is B's width
	*
	*                +---+
	*                |B^T|
	*            +-------+
	*            |   |   |
	*            | A | C |
	*            |   |   |
	*            |   |   |
	* +------------------+
	* |    D^T   | E |
	* +----------+---+
	*
	*/
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void matmul2(const Dtype *A, const int wA, const int hA,
	const Dtype *B, const int wB, const int hB,
	const Dtype *D, const int wD, const int hD, Dtype *C,
	Dtype *E, const Itype *in_map, const Itype *out_map) {
	// Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	// Coordinate. y is for rows, x is for columns.
	const int x = BLOCK_SIZE * bx + tx;
	const int y = BLOCK_SIZE * by + ty;

	const Itype in_row = y < hA ? in_map[y] : 0;
	const Itype out_row = y < hA ? out_map[y] : 0;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	Dtype Csub = 0;
	Dtype Esub = 0;

	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	__shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

	// Declaration of the shared memory array Bs used to
	// store the sub-matrix of B
	__shared__ Dtype BTs[BLOCK_SIZE][BLOCK_SIZE];

	// Declaration of the shared memory array Ds used to
	// store the sub-matrix of D
	__shared__ Dtype DTs[BLOCK_SIZE][BLOCK_SIZE];

	// For Ds = D^T[...:..., ...:...], use the transposed grid dimension for A
	DTs[ty][tx] = (x < wD && y < hD) ? D[wD * in_row + x] : 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int s = 0; s < wA; s += BLOCK_SIZE) {
		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * out_row + s + tx] : 0;

		// Transposed kernel
		BTs[ty][tx] = ((s + ty) < wB && x < hB) ? B[wB * x + s + ty] : 0;

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * BTs[k][tx];
		}

		// For Esub, reset to 0
		Esub = 0;
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Esub += DTs[k][ty] * As[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();

		// For the E matrix which requires accmulation of multiple blocks, use
		// atomic addition. This can be replaced with a more sophisticaed reduction
		// algorithm.
		if ((bx * BLOCK_SIZE + ty) < wD && (s + tx) < wA)
			atomicAdd(&E[wA * (bx * BLOCK_SIZE + ty) + (s + tx)], Esub);
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	if (y < hA && x < hB)
		atomicAdd(&C[hB * in_row + x], Csub);
}

namespace minkowski {

	template <typename Dtype, typename Itype>
	void ConvolutionForwardKernelGPU(const Dtype *d_in_feat, int in_nchannel,
		Dtype *d_out_feat, int out_nchannel,
		const Dtype *d_kernel,
		const pInOutMaps<Itype> &in_maps,
		const pInOutMaps<Itype> &out_maps,
		int out_nrows, cublasHandle_t cuhandle,
		cudaStream_t stream) {

		AT_CUDA_CHECK(cudaDeviceSynchronize());

		int n_active_in_volume, shared_mem_size = -1;

		// Define the shared memory size
		if ((in_nchannel > 16 && out_nchannel > 16 &&
			in_nchannel * out_nchannel >= 512) ||
			(in_nchannel > 24 && out_nchannel > 24))
			shared_mem_size = 32;
		else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
			shared_mem_size = 24;
		else if ((in_nchannel > 8 && out_nchannel > 8) ||
			(in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
			shared_mem_size = 16;
		else
			shared_mem_size = 8;

		dim3 threads(shared_mem_size, shared_mem_size);

		// Iterate through each spatial kernel and get indices for in_map and out_map
		for (int k = 0; k < in_maps.size(); k++) {
			n_active_in_volume = in_maps[k].size();
			if (n_active_in_volume == 0)
				continue;

			int num_grid = (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
			int num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
			int step = (n_active_in_volume + num_div - 1) / num_div;

			for (int s = 0; s < num_div; s++) {
				int offset = step * s;
				int remainder = n_active_in_volume - step * s;
				int curr_num_active = remainder < step ? remainder : step;
				dim3 grid((out_nchannel + threads.x - 1) / threads.x,
					(curr_num_active + threads.y - 1) / threads.y);
				switch (shared_mem_size) {
				case 32:
					matmul<Dtype, Itype, 32> << <grid, threads, 0, stream >> > (
						d_in_feat, in_nchannel, curr_num_active,
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel, d_out_feat, in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				case 24:
					matmul<Dtype, Itype, 24> << <grid, threads, 0, stream >> > (
						d_in_feat, in_nchannel, curr_num_active,
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel, d_out_feat, in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				case 16:
					matmul<Dtype, Itype, 16> << <grid, threads, 0, stream >> > (
						d_in_feat, in_nchannel, curr_num_active,
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel, d_out_feat, in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				case 8:
					matmul<Dtype, Itype, 8> << <grid, threads, 0, stream >> > (
						d_in_feat, in_nchannel, curr_num_active,
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel, d_out_feat, in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				}
			}
			AT_CUDA_CHECK(cudaGetLastError());
		}
		AT_CUDA_CHECK(cudaDeviceSynchronize());
	}

	template void ConvolutionForwardKernelGPU<float, int32_t>(
		const float *d_in_feat, int in_nchannel, float *d_out_feat,
		int out_nchannel, const float *d_kernel, const pInOutMaps<int32_t> &in_map,
		const pInOutMaps<int32_t> &out_map, int out_nrows, cublasHandle_t cuhandle,
		cudaStream_t stream);

	template void ConvolutionForwardKernelGPU<double, int32_t>(
		const double *d_in_feat, int in_nchannel, double *d_out_feat,
		int out_nchannel, const double *d_kernel, const pInOutMaps<int32_t> &in_map,
		const pInOutMaps<int32_t> &out_map, int out_nrows, cublasHandle_t cuhandle,
		cudaStream_t stream);

	template <typename Dtype, typename Itype>
	void ConvolutionBackwardKernelGPU(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
		int in_nchannel, const Dtype *d_grad_out_feat,
		int out_nchannel, const Dtype *d_kernel,
		Dtype *d_grad_kernel,
		const pInOutMaps<Itype> &in_maps,
		const pInOutMaps<Itype> &out_maps,
		int out_nrows, cublasHandle_t cuhandle,
		cudaStream_t stream) {

		AT_CUDA_CHECK(cudaDeviceSynchronize());

		int n_active_in_volume, shared_mem_size = -1;

		// Define the shared memory size
		if ((in_nchannel > 16 && out_nchannel > 16 &&
			in_nchannel * out_nchannel >= 512) ||
			(in_nchannel % 32 == 0 && out_nchannel % 32 == 0))
			shared_mem_size = 32;
		else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
			shared_mem_size = 24;
		else if ((in_nchannel > 8 && out_nchannel > 8) ||
			(in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
			shared_mem_size = 16;
		else
			shared_mem_size = 8;

		dim3 threads(shared_mem_size, shared_mem_size);

		for (int k = 0; k < in_maps.size(); k++) {
			n_active_in_volume = in_maps[k].size();
			if (n_active_in_volume == 0)
				continue;

			int num_grid = (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
			int num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
			int step = (n_active_in_volume + num_div - 1) / num_div;

			for (int s = 0; s < num_div; s++) {
				int offset = step * s;
				int remainder = n_active_in_volume - step * s;
				int curr_num_active = remainder < step ? remainder : step;
				dim3 grid((in_nchannel + threads.x - 1) / threads.x,
					(curr_num_active + threads.y - 1) / threads.y);
				switch (shared_mem_size) {
				case 32:
					matmul2<Dtype, Itype, 32> << <grid, threads, 0, stream >> > (
						d_grad_out_feat, out_nchannel, curr_num_active, // A
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel,                                    // B
						d_in_feat, in_nchannel, curr_num_active,        // D
						d_grad_in_feat,                                 // C
						&d_grad_kernel[k * in_nchannel * out_nchannel], // E
						in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				case 24:
					matmul2<Dtype, Itype, 24> << <grid, threads, 0, stream >> > (
						d_grad_out_feat, out_nchannel, curr_num_active, // A
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel,                                    // B
						d_in_feat, in_nchannel, curr_num_active,        // D
						d_grad_in_feat,                                 // C
						&d_grad_kernel[k * in_nchannel * out_nchannel], // E
						in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				case 16:
					matmul2<Dtype, Itype, 16> << <grid, threads, 0, stream >> > (
						d_grad_out_feat, out_nchannel, curr_num_active, // A
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel,                                    // B
						d_in_feat, in_nchannel, curr_num_active,        // D
						d_grad_in_feat,                                 // C
						&d_grad_kernel[k * in_nchannel * out_nchannel], // E
						in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				case 8:
					matmul2<Dtype, Itype, 8> << <grid, threads, 0, stream >> > (
						d_grad_out_feat, out_nchannel, curr_num_active, // A
						&d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
						in_nchannel,                                    // B
						d_in_feat, in_nchannel, curr_num_active,        // D
						d_grad_in_feat,                                 // C
						&d_grad_kernel[k * in_nchannel * out_nchannel], // E
						in_maps[k].data() + offset, out_maps[k].data() + offset);
					break;
				}
			}
			AT_CUDA_CHECK(cudaGetLastError());
		}
		AT_CUDA_CHECK(cudaDeviceSynchronize());
	}

	template void ConvolutionBackwardKernelGPU<float, int32_t>(
		const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
		const float *d_grad_out_feat, int out_nchannel, const float *d_kernel,
		float *p_grad_kernel, const pInOutMaps<int32_t> &in_map,
		const pInOutMaps<int32_t> &out_map, int out_nrows, cublasHandle_t cuhandle,
		cudaStream_t stream);

	template void ConvolutionBackwardKernelGPU<double, int32_t>(
		const double *d_in_feat, double *d_grad_in_feat, int in_nchannel,
		const double *d_grad_out_feat, int out_nchannel, const double *d_kernel,
		double *p_grad_kernel, const pInOutMaps<int32_t> &in_map,
		const pInOutMaps<int32_t> &out_map, int out_nrows, cublasHandle_t cuhandle,
		cudaStream_t stream);

} // end namespace minkowski

}  // namespace kaolin
