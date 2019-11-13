#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif



namespace{


template <typename scalar_t>
__global__ void voxelize_sub1_kernel(
        const scalar_t* __restrict__ faces,
        int32_t* voxels,
        int batch_size,
        int num_faces,
        int voxel_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * voxel_size * voxel_size) {
        return;
    }
    const int bs = batch_size;
    const int nf = num_faces;
    const int vs = voxel_size;

    int y = i % vs;
    int x = (i / vs) % vs;
    int bn = i / (vs * vs);
    //
    for (int fn = 0; fn < nf; fn++) {
        const scalar_t* face = &faces[(bn * nf + fn) * 9];
        scalar_t y1d = face[3] - face[0];
        scalar_t x1d = face[4] - face[1];
        scalar_t z1d = face[5] - face[2];
        scalar_t y2d = face[6] - face[0];
        scalar_t x2d = face[7] - face[1];
        scalar_t z2d = face[8] - face[2];
        scalar_t ypd = y - face[0];
        scalar_t xpd = x - face[1];
        scalar_t det = x1d * y2d - x2d * y1d;
        if (det == 0) continue;
        scalar_t t1 = (y2d * xpd - x2d * ypd) / det;
        scalar_t t2 = (-y1d * xpd + x1d * ypd) / det;
        if (t1 < 0) continue;
        if (t2 < 0) continue;
        if (1 < t1 + t2) continue;
        int zi = floor(t1 * z1d + t2 * z2d + face[2]);
        int yi, xi;
        yi = y;
        xi = x;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        yi = y - 1;
        xi = x;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        yi = y;
        xi = x - 1;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        yi = y - 1;
        xi = x - 1;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
    }
}


template <typename scalar_t>
__global__ void voxelize_sub2_kernel(
        const scalar_t* __restrict__ faces,
        int32_t* voxels,
        int batch_size,
        int num_faces,
        int voxel_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int bs = batch_size;
    const int nf = num_faces;
    const int vs = voxel_size;

    int fn = i % nf;
    int bn = i / nf;
    const scalar_t* face = &faces[(bn * nf + fn) * 9];
    for (int k = 0; k < 3; k++) {
        int yi = floor(face[3 * k + 0]);
        int xi = floor(face[3 * k + 1]);
        int zi = floor(face[3 * k + 2]);
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs)) {
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        }
    }
}

template <typename scalar_t>
__global__ void voxelize_sub3_kernel(
        int32_t* voxels,
        int32_t* visible,
        int batch_size,
        int voxel_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * voxel_size * voxel_size * voxel_size) {
        return;
    }
    const int bs = batch_size;
    const int vs = voxel_size;

    int z = i % vs;
    int x = (i / vs) % vs;
    int y = (i / (vs * vs)) % vs;
    int bn = i / (vs * vs * vs);
    int pn = i;
    if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) {
        if (voxels[pn] == 0) visible[pn] = 1;
    }
}

template <typename scalar_t>
__global__ void voxelize_sub4_kernel(
        int32_t* voxels,
        int32_t* visible,
        int batch_size,
        int voxel_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * voxel_size * voxel_size * voxel_size) {
        return;
    }
    const int bs = batch_size;
    const int vs = voxel_size;

    int z = i % vs;
    int x = (i / vs) % vs;
    int y = (i / (vs * vs)) % vs;
    int bn = i / (vs * vs * vs);
    int pn = i;
    if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) return;
    if (voxels[pn] == 0 && visible[pn] == 0) {
        int yi, xi, zi;
        yi = y - 1;
        xi = x;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y + 1;
        xi = x;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x - 1;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x + 1;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x;
        zi = z - 1;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x;
        zi = z + 1;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
    }
}

}



std::vector<at::Tensor> voxelize_sub1_cuda(
        at::Tensor faces,
        at::Tensor voxels) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto voxel_size = voxels.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * voxel_size * voxel_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "voxelize_sub1_cuda", ([&] {
      voxelize_sub1_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          voxels.data<int32_t>(),
          batch_size,
          num_faces,
          voxel_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in voxelize_sub1_kernel: %s\n", cudaGetErrorString(err));

    return {voxels};
}



std::vector<at::Tensor> voxelize_sub2_cuda(
        at::Tensor faces,
        at::Tensor voxels) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto voxel_size = voxels.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * num_faces - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "voxelize_sub2_cuda", ([&] {
      voxelize_sub2_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          voxels.data<int32_t>(),
          batch_size,
          num_faces,
          voxel_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in voxelize_sub2_kernel: %s\n", cudaGetErrorString(err));

    return {voxels};
}

std::vector<at::Tensor> voxelize_sub3_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {

    const auto batch_size = voxels.size(0);
    const auto voxel_size = voxels.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * voxel_size  * voxel_size  * voxel_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "voxelize_sub3_cuda", ([&] {
      voxelize_sub3_kernel<scalar_t><<<blocks, threads>>>(
          voxels.data<int32_t>(),
          visible.data<int32_t>(),
          batch_size,
          voxel_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in voxelize_sub3_kernel: %s\n", cudaGetErrorString(err));

    return {voxels, visible};
}

std::vector<at::Tensor> voxelize_sub4_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {

    const auto batch_size = voxels.size(0);
    const auto voxel_size = voxels.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * voxel_size  * voxel_size  * voxel_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "voxelize_sub4_cuda", ([&] {
      voxelize_sub4_kernel<scalar_t><<<blocks, threads>>>(
          voxels.data<int32_t>(),
          visible.data<int32_t>(),
          batch_size,
          voxel_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in voxelize_sub4_kernel: %s\n", cudaGetErrorString(err));

    return {voxels, visible};
}