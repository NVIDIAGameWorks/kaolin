// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#include <ATen/ATen.h>

#define CUB_STDERR
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <math_constants.h>

#include "../../../spc_math.h"
#include "../../../spc_utils.cuh"

namespace kaolin {

using namespace std;
using namespace at::indexing;

#define NUM_THREADS 256
namespace {

size_t get_cub_storage_bytes_sort_pairs(
  void* d_temp_storage, 
  const uint64_t* d_morton_codes_in, 
  uint64_t* d_morton_codes_out, 
  const uint64_t* d_values_in, 
  uint64_t* d_values_out, 
  uint32_t num_items,
  cudaStream_t stream = 0)
{
    size_t    temp_storage_bytes = 0;
    CubDebugExit(
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        d_morton_codes_in, d_morton_codes_out, 
                                        d_values_in, d_values_out, num_items, 0, sizeof(uint64_t) * 8, stream)
    );
    return temp_storage_bytes;
}

__global__ void
d_MarkDuplicates(
  uint32_t num, 
  morton_code* mcode, 
  uint32_t* occupancy)
{
  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  if (tidx == 0)
    occupancy[tidx] = 1;
  else
    occupancy[tidx] = mcode[tidx - 1] == mcode[tidx] ? 0 : 1;
}

__global__ void
d_Compactify(
  const uint32_t num, 
  const morton_code* mIn, 
  morton_code* mOut, 
  const uint32_t* occupancy, 
  const uint32_t* prefix_sum)
{
  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  if (occupancy[tidx]) {
    uint32_t base_idx = tidx == 0 ? 0 : prefix_sum[tidx-1];
    mOut[base_idx] = mIn[tidx];
  }
}

// This function will subdivide input voxel to out voxels 
__global__ void
subdivide_cuda_kernel(
    const uint32_t num, 
    const uint64_t* __restrict__ morton_codes_in, // input voxel morton codes
    const uint64_t* __restrict__ triangle_id_in, // input face ids
    uint64_t* __restrict__ morton_codes_out, // output voxel morton codes
    uint64_t* __restrict__ triangle_id_out, // output face ids
    const uint32_t* __restrict__ occupancy,  
    const uint32_t* __restrict__ prefix_sum) {

  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  if (occupancy[tidx]) {
    uint64_t triangle_id = triangle_id_in[tidx]; //triangle id

    point_data p_in = to_point(morton_codes_in[tidx]);
    point_data p_out = make_point_data(2 * p_in.x, 2 * p_in.y, 2 * p_in.z);

    uint32_t base_idx = tidx == 0 ? 0 : prefix_sum[tidx-1];

    for (uint32_t i = 0; i < 8; i++) {
      point_data p = make_point_data(p_out.x+(i>>2), p_out.y+((i>>1)&0x1), p_out.z+(i&0x1));
      morton_codes_out[base_idx] = to_morton(p);
      triangle_id_out[base_idx] = triangle_id;
      base_idx++;
    }
  }
}

// compacify input buffers
__global__ void
compactify_cuda_kernel(
    const uint32_t num, 
    const uint64_t* __restrict__ morton_codes_in, 
    const uint64_t* __restrict__ triangle_id_in, 
    uint64_t* __restrict__ morton_codes_out, 
    uint64_t* __restrict__ triangle_id_out, 
    const uint32_t* __restrict__ occupancy,
    const uint32_t* __restrict__ prefix_sum) {

  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  if (occupancy[tidx]) { 
    uint32_t base_idx = tidx == 0 ? 0 : prefix_sum[tidx-1];
    morton_codes_out[base_idx] = morton_codes_in[tidx];
    triangle_id_out[base_idx] = triangle_id_in[tidx];
  }
}

} // namespace

///////// methods below are unique to gs_to_spc

__device__ bool AABBinsideVoxel(
  const float3& ab0,
  const float3& ab1,
  const float3& v,
  const float vsize) {

    return (
      v.x <= ab0.x &&
      v.y <= ab0.y &&
      v.z <= ab0.z &&
      (v.x+vsize) >= ab1.x &&
      (v.y+vsize) >= ab1.y &&
      (v.z+vsize) >= ab1.z
    );
  }

__device__ bool VoxelOverlapAABB(
  const float3& ab0,
  const float3& ab1,
  const float3& v,
  const float vsize) {

      if (ab1.x < v.x) return false;
      if (ab1.y < v.y) return false;
      if (ab1.z < v.z) return false;
      if (ab0.x > (v.x+vsize)) return false;
      if (ab0.y > (v.y+vsize)) return false;
      if (ab0.z > (v.z+vsize)) return false;

      return true;
  }

__device__ bool edge_test(
    const float c0,
    const float c1,
    const float c2,
    const float c3,
    const float c4,
    const float c5,
    const float cp6,
    const float s,
    const float t,
    const float l,
    const float u) 
    { 
      double a = c0;
      double b = 2*(c1*s + c2*t);
      double c = (c3*s*s + 2*c4*s*t + c5*t*t) - cp6;
      double dcrm = max(b*b - 4*a*c, 0.0);
      double b2a = -0.5*b/a;
      double r0 = b2a;
      double r1 = b2a;

      if (dcrm > 0) {
        double rdcrm = 0.5*sqrt(dcrm)/a;
        r0 = b2a - rdcrm;
        r1 = b2a + rdcrm;

        return !(u <= r0 || r1 <= l);
      }

      return false;
    }

__device__ bool VoxelEdgesIntersectEllipsoid(
    const float3& mean, 
    const float* Covi, 
    const float3& v, 
    const float vsize,
    const float iso) 
    {

   float3 p = v - mean;
    // if edge crossings, true true
    for (int i = 0; i < 4; i++)
    {
      if (edge_test(Covi[0], Covi[1], Covi[2], Covi[3], Covi[4], Covi[5], iso, 
          p.y+(i/2?vsize:0.0), p.z+(i%2?vsize:0.0), p.x, p.x+vsize)) return true;

      if (edge_test(Covi[3], Covi[1], Covi[4], Covi[0], Covi[2], Covi[5], iso, 
          p.x+(i/2?vsize:0.0), p.z+(i%2?vsize:0.0), p.y, p.y+vsize)) return true;

      if (edge_test(Covi[5], Covi[2], Covi[4], Covi[0], Covi[1], Covi[3], iso,
          p.x+(i/2?vsize:0.0), p.y+(i%2?vsize:0.0), p.z, p.z+vsize)) return true;
    }

    return  false; 
  }

__device__ bool facetest(
    const float* p, 
    float* q, 
    const float vsize, 
    const int i,
    const int j,
    const int k,
    const float vs) 
    {
      float t = (q[i]-(p[i]+vs))/(2.0*q[i]);

      if (0.0f <= t && t <= 1.0f)
      {
        float h[3];
        for (int l=0; l<3; l++)
          h[l] = (1.0-2.0*t)*q[l];

        return p[j] <= h[j] && h[j] <= (p[j] + vsize) && p[k] <= h[k] && h[k] <= (p[k] + vsize);
      }

      return false;
    }

__device__ bool
VoxelFaceOverlapEllipsoid(
    const float3& mean, 
    float3 cp0,
    float3 cp1,
    float3 cp2,
    const float3& v, 
    const float vsize) 
    {
      float3 p = v - mean;
      bool b[6];

      // need to index float3 components, so cast to float[]
      b[0] = facetest((float*)(&p), (float*)(&cp0), vsize, 0,1,2, 0.0);
      b[1] = facetest((float*)(&p), (float*)(&cp0), vsize, 0,1,2, vsize);

      b[2] = facetest((float*)(&p), (float*)(&cp1), vsize, 1,0,2, 0.0);
      b[3] = facetest((float*)(&p), (float*)(&cp1), vsize, 1,0,2, vsize);

      b[4] = facetest((float*)(&p), (float*)(&cp2), vsize, 2,0,1, 0.0);
      b[5] = facetest((float*)(&p), (float*)(&cp2), vsize, 2,0,1, vsize);

      return (b[0] || b[1] || b[2] || b[3] || b[4] || b[5]);
    }

// compute 3x3 matrix that maps 3d cartesian coords to 3d barycentriuc coods, w.r.t a given triangle
__device__ bool
GSVoxelInOut(
    const uint64_t gaus_idx,
    const float3& mean, 
    const float* Covi, 
    const float3& ab0,
    const float3& ab1,
    const float3& cp0,
    const float3& cp1,
    const float3& cp2,
    const float3& v, 
    const float vsize,
    const float iso) {

    if (AABBinsideVoxel(ab0, ab1, v, vsize)) 
      return true;

    if (!VoxelOverlapAABB(ab0, ab1, v, vsize))
      return false;

    if (VoxelFaceOverlapEllipsoid(mean, cp0, cp1, cp2, v, vsize))
      return true;

    return VoxelEdgesIntersectEllipsoid(mean, Covi, v, vsize, iso);

}

// This function will iterate over t(triangle intersection proposals) and determine if they result in an 
// intersection. If they do, the occupancy tensor is set to number of voxels in subdivision or compaction
__global__ void
decide_cuda_kernel(
  const uint32_t num, 
  const float3* __restrict__ means3D, 
  const float* __restrict__ cov3DInvs, 
  const float3* __restrict__ AABBgs, 
  const float3* __restrict__ ContactPnts,
  const uint64_t* __restrict__ morton_codes, // voxel morton codes
  const uint64_t* __restrict__ gaus_id, // face ids
  uint32_t* __restrict__ occupancy, 
  const uint32_t level, 
  const uint32_t not_done,
  const float iso) {

  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  uint64_t gaus_idx = gaus_id[tidx];

  float3 mean = means3D[gaus_idx];
  point_data gridPos = to_point(morton_codes[tidx]);

  float two_level = (float)(0x1 << level);
  float voxelSize = 2.0f/two_level;

  float3 v;
  v.x = fmaf(gridPos.x, voxelSize, -1.0f);
  v.y = fmaf(gridPos.y, voxelSize, -1.0f);
  v.z = fmaf(gridPos.z, voxelSize, -1.0f);

  float3 ab0 = AABBgs[2*gaus_idx+0];
  float3 ab1 = AABBgs[2*gaus_idx+1];

  float3 cp0 = ContactPnts[3*gaus_idx + 0];
  float3 cp1 = ContactPnts[3*gaus_idx + 1];
  float3 cp2 = ContactPnts[3*gaus_idx + 2];

  if (GSVoxelInOut(gaus_idx, mean, cov3DInvs+6*gaus_idx, ab0, ab1, cp0, cp1, cp2, v, voxelSize, iso))
    occupancy[tidx] = not_done ? 8 : 1;
  else
    occupancy[tidx] = 0;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D inverse covariance matrix in world space. Also takes care
// of quaternion normalization.
// Modified from Guass Splat code
__device__ void computeCov3DInv(
  const uint32_t tidx, 
  const float3 mean, 
  float3 scale, 
  const float4 rot, 
  float* cov3D, 
  float3* AABBgs, 
  float3* ContactPnts,
  const float iso, 
  const float tol,
  const uint32_t level)
{
  float two_level = (float)(0x1 << level);
  float voxelSize = 2.0f/two_level;
  float min_scale = tol*voxelSize;

  scale.x = max(scale.x, min_scale);
  scale.y = max(scale.y, min_scale);
  scale.z = max(scale.z, min_scale);

  double detS = ((double)scale.x) * ((double)scale.y) * ((double)scale.z);

	float3x3 S = make_float3x3(
    1.0f/scale.x, 0.0f, 0.0f,
    0.0f, 1.0f/scale.y, 0.0f,
    0.0f, 0.0f, 1.0f/scale.z); // to get Cov^{-1}

	// Normalize quaternion to get valid rotation
	float4 q = rot;// / glm::length(rot); //is done upstream
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	// float3x3 R = make_float3x3(
	// 	1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
	// 	2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
	// 	2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	// );
	float3x3 R = make_float3x3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
		2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
		2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
	);

	float3x3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	float3x3 Sigma = transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma.m[0][0];
	cov3D[1] = Sigma.m[0][1];
	cov3D[2] = Sigma.m[0][2];
	cov3D[3] = Sigma.m[1][1];
	cov3D[4] = Sigma.m[1][2];
	cov3D[5] = Sigma.m[2][2];

  // Compute closest points to coordinate planes
  double c0 = cov3D[0];
  double c1 = cov3D[1];
  double c2 = cov3D[2];
  double c3 = cov3D[3];
  double c4 = cov3D[4];
  double c5 = cov3D[5];

  double h0 = c3*c5 - c4*c4;
  double h1 = c2*c4 - c1*c5;
  double h2 = c1*c4 - c2*c3;
  double h3 = c0*c5 - c2*c2;
  double h4 = c1*c2 - c0*c4;
  double h5 = c0*c3 - c1*c1;

  double w[3];
  w[0] = detS*sqrt(iso/h0);
  w[1] = detS*sqrt(iso/h3);
  w[2] = detS*sqrt(iso/h5);

  double3 Q[3];
  Q[0] = make_double3(h0, h1, h2);
  Q[1] = make_double3(h1, h3, h4);
  Q[2] = make_double3(h2, h4, h5);

  float3 P[6];
  for (int i = 0; i < 3; i++)
  {
    P[2*i] = make_float3((float)(w[i]*Q[i].x), (float)(w[i]*Q[i].y), (float)(w[i]*Q[i].z));
    P[2*i+1] = -1.0f*P[2*i];
  }

  ContactPnts[0] = P[0];
  ContactPnts[1] = P[2];
  ContactPnts[2] = P[4];
 
  // find AABB of closest points
  float3 Pmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 Pmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  for (int i = 0; i < 6; i++)
  {
    Pmin.x = fminf(Pmin.x, P[i].x);
    Pmin.y = fminf(Pmin.y, P[i].y);
    Pmin.z = fminf(Pmin.z, P[i].z);
    Pmax.x = fmaxf(Pmax.x, P[i].x);
    Pmax.y = fmaxf(Pmax.y, P[i].y);
    Pmax.z = fmaxf(Pmax.z, P[i].z);
  }

  AABBgs[0] = Pmin + mean;
  AABBgs[1] = Pmax + mean;
}

__global__ void
computeCov3DInv_kernel(
  const uint32_t num, 
  const float3* __restrict__ means3D, 
  const float3* __restrict__ scales, 
  const float4* __restrict__ rotations, 
  float* __restrict__ Cov3DInvs,
  float3* __restrict__ AABBgs,
  float3* __restrict__ ContactPnts,
  const float iso,
  const float tol,
  const uint32_t level) {

  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

	computeCov3DInv(tidx, means3D[tidx], scales[tidx], rotations[tidx], 
                  Cov3DInvs + tidx * 6, AABBgs + tidx * 2, ContactPnts + tidx * 3,
                  iso, tol, level);
}

__device__ float atomicAlpha(float* address, float val)
{
    uint32_t* address_as_ui = (uint32_t*)address;
    uint32_t old = *address_as_ui, assumed;

    do {
        assumed = old;
        float f = __uint_as_float(assumed);
        uint32_t tmp = __float_as_uint(val + f - val*f);
        // uint32_t tmp = __float_as_uint(val*f);
        old = atomicCAS(address_as_ui, assumed, tmp);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __uint_as_float(old);
}

__global__ void
MergeOpacities_kernel(
  const uint32_t num, 
  const uint64_t* __restrict__ morton_codes, // voxel morton codes, sorted
  const uint64_t* __restrict__ gaus_id, // corresponding gauss ids
  const uint32_t* __restrict__ prefix_sum, // inclusive sum of morton code boundary marks
  const float3* __restrict__ means3D, 
  const float* __restrict__ Cov3DInvs,
  const float* __restrict__ opacities, 
  float* __restrict__ merged_opacities,
  const uint32_t level) {

  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  point_data gridPos = to_point(morton_codes[tidx]); // morton code to point (voxel coordinates)
  uint64_t gidx =  gaus_id[tidx]; // corresponding gauss id
  uint32_t base_idx = tidx == 0 ? 0 : prefix_sum[tidx-1]; // voxel index in unquified list
  float3 mean = means3D[gidx]; // mean of gaussian

  float c0 = Cov3DInvs[6*gidx + 0]; // inverse covariance values of gaussian
  float c1 = Cov3DInvs[6*gidx + 1];
  float c2 = Cov3DInvs[6*gidx + 2];
  float c3 = Cov3DInvs[6*gidx + 3];
  float c4 = Cov3DInvs[6*gidx + 4];
  float c5 = Cov3DInvs[6*gidx + 5];

  float two_level = (float)(0x1 << level);
  float voxelSize = 2.0f/two_level;

  float3 v; // centroid of voxel
  v.x = fmaf(gridPos.x + 0.5f, voxelSize, -1.0f - mean.x);
  v.y = fmaf(gridPos.y + 0.5f, voxelSize, -1.0f - mean.y);
  v.z = fmaf(gridPos.z + 0.5f, voxelSize, -1.0f - mean.z);

  float gval = c0*v.x*v.x + 2.0f*c1*v.x*v.y + 2.0f*c2*v.x*v.z + 
                c3*v.y*v.y + 2.0f*c4*v.y*v.z + c5*v.z*v.z;

  float power = -0.5f * gval;
  float alpha = min(0.99, opacities[gidx] * exp(power));

  atomicAlpha(&(merged_opacities[base_idx]), alpha);
}

// This is a newer version of gs_to_spc that does not merge opacities

std::vector<at::Tensor> gs_to_spc_cuda_impl(
  const at::Tensor& means3D,
  const at::Tensor& scales,
  const at::Tensor& rotations,
  const float iso,
  const float tol,
  const uint32_t target_level) {

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(means3D));
  auto stream = at::cuda::getCurrentCUDAStream();

  // number of Gaussians input
  int ngaus = means3D.size(0);

  // compute upper right block of covariance matices
  at::Tensor Cov3DInvs = at::empty({ngaus, 6}, means3D.options());
  // the AABB of the ellisoid
  at::Tensor AABBgs = at::empty({ngaus, 6}, means3D.options());
  // axes aligned contact points with the ellisoid - untranslated
  at::Tensor ContactPnts = at::empty({ngaus, 9}, means3D.options());

  computeCov3DInv_kernel<<<(ngaus + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
    ngaus,
    reinterpret_cast<float3*>(means3D.data_ptr<float>()),
    reinterpret_cast<float3*>(scales.data_ptr<float>()),
    reinterpret_cast<float4*>(rotations.data_ptr<float>()),
    Cov3DInvs.data_ptr<float>(),
    reinterpret_cast<float3*>(AABBgs.data_ptr<float>()),
    reinterpret_cast<float3*>(ContactPnts.data_ptr<float>()),
    iso, tol, target_level);

  // allocate local GPU storage
  at::Tensor morton_codes[2];
  morton_codes[0] = at::zeros({ngaus}, means3D.options().dtype(at::kLong));
  morton_codes[1] = at::zeros({0}, means3D.options().dtype(at::kLong));

  at::Tensor gaus_id[2];
  gaus_id[0] = at::arange({ngaus}, means3D.options().dtype(at::kLong));
  gaus_id[1] = at::arange({0}, means3D.options().dtype(at::kLong));

  at::Tensor occupancy = at::empty({0}, means3D.options().dtype(at::kInt));
  at::Tensor prefix_sum = at::empty({0}, means3D.options().dtype(at::kInt));
  at::Tensor temp_storage = at::empty({0}, means3D.options().dtype(at::kByte));

  uint64_t temp_storage_bytes;
  uint32_t next_cnt, curr_cnt = ngaus;
  uint32_t curr_buf = 0;
  for (uint32_t l = 0; l <= target_level; l++) {
    occupancy.resize_({curr_cnt});
    prefix_sum.resize_({curr_cnt});

    // Do the proposals hit?
    decide_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
      curr_cnt,
      reinterpret_cast<float3*>(means3D.data_ptr<float>()),
      Cov3DInvs.data_ptr<float>(),
      reinterpret_cast<float3*>(AABBgs.data_ptr<float>()),
      reinterpret_cast<float3*>(ContactPnts.data_ptr<float>()),
      reinterpret_cast<uint64_t*>(morton_codes[curr_buf].data_ptr<int64_t>()),
      reinterpret_cast<uint64_t*>(gaus_id[curr_buf].data_ptr<int64_t>()),
      reinterpret_cast<uint32_t*>(occupancy.data_ptr<int>()),
      l,
      target_level - l,
      iso
    );

    // set up memory for DeviceScan calls
    temp_storage_bytes = get_cub_storage_bytes(NULL, reinterpret_cast<uint32_t*>(occupancy.data_ptr<int>()), 
      reinterpret_cast<uint32_t*>(prefix_sum.data_ptr<int>()), curr_cnt, stream);
    temp_storage.resize_({static_cast<int64_t>(temp_storage_bytes)});

    CubDebugExit(cub::DeviceScan::InclusiveSum(
      (void*)temp_storage.data_ptr<uint8_t>(), temp_storage_bytes, 
      reinterpret_cast<uint32_t*>(occupancy.data_ptr<int>()), 
      reinterpret_cast<uint32_t*>(prefix_sum.data_ptr<int>()), curr_cnt, stream));

    cudaMemcpyAsync(&next_cnt, reinterpret_cast<uint32_t*>(prefix_sum.data_ptr<int>()) + curr_cnt - 1, 
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

    if (next_cnt == 0) {
      at::Tensor mcodes = at::empty({0}, means3D.options().dtype(at::kLong));
      return { mcodes };
    } else {
      // re-allocate local GPU storage
      morton_codes[(curr_buf + 1) % 2].resize_({next_cnt});
      gaus_id[(curr_buf + 1) % 2].resize_({next_cnt});
    }

    // Subdivide if more levels remain, repeat
    if (l < target_level) {
      subdivide_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
        curr_cnt, 
        reinterpret_cast<uint64_t*>(morton_codes[curr_buf].data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(gaus_id[curr_buf].data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(morton_codes[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(gaus_id[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
        reinterpret_cast<uint32_t*>(occupancy.data_ptr<int>()), 
        reinterpret_cast<uint32_t*>(prefix_sum.data_ptr<int>()));
    } else {
      compactify_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
        curr_cnt, 
        reinterpret_cast<uint64_t*>(morton_codes[curr_buf].data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(gaus_id[curr_buf].data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(morton_codes[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(gaus_id[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
        reinterpret_cast<uint32_t*>(occupancy.data_ptr<int>()), 
        reinterpret_cast<uint32_t*>(prefix_sum.data_ptr<int>()));
    }

    curr_buf = (curr_buf + 1) % 2;
    curr_cnt = next_cnt;
  }

  // re-allocate local GPU storage
  morton_codes[(curr_buf + 1) % 2].resize_({curr_cnt});
  gaus_id[(curr_buf + 1) % 2].resize_({curr_cnt});

  // set up memory for DeviceScan calls
  temp_storage_bytes = get_cub_storage_bytes_sort_pairs(
    NULL, 
    reinterpret_cast<uint64_t*>(morton_codes[curr_buf].data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(morton_codes[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(gaus_id[curr_buf].data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(gaus_id[(curr_buf + 1) % 2].data_ptr<int64_t>()),
    curr_cnt,
    stream);
  temp_storage.resize_({(int64_t)temp_storage_bytes});

  CubDebugExit(cub::DeviceRadixSort::SortPairs(
    (void*)temp_storage.data_ptr<uint8_t>(), 
    temp_storage_bytes, 
    reinterpret_cast<uint64_t*>(morton_codes[curr_buf].data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(morton_codes[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(gaus_id[curr_buf].data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(gaus_id[(curr_buf + 1) % 2].data_ptr<int64_t>()),
    curr_cnt,
    0,
    sizeof(uint64_t) * 8,
    stream));

  occupancy.resize_({curr_cnt});

  // Mark boundaries of unique  
  d_MarkDuplicates <<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>> (
    curr_cnt, 
    reinterpret_cast<uint64_t*>(morton_codes[(curr_buf + 1) % 2].data_ptr<int64_t>()), 
    reinterpret_cast<uint32_t*>(occupancy.data_ptr<int>()));

  at::Tensor points = at::empty({curr_cnt, 3}, means3D.options().dtype(at::kShort));

  morton_to_points_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
    reinterpret_cast<uint64_t*>(morton_codes[(curr_buf + 1) % 2].data_ptr<int64_t>()),
    reinterpret_cast<point_data*>(points.data_ptr<short>()),
    curr_cnt);

  return {
    points.contiguous(),
    gaus_id[(curr_buf + 1) % 2].contiguous(),
    occupancy.to(at::kBool).contiguous(),
    Cov3DInvs.contiguous()};
}

__global__ void
integrate_gs_kernel(
  const uint32_t                  num, 
  const point_data* __restrict__  points, // voxel morton codes, sorted
  const float                     voxel_size,
  const uint64_t*   __restrict__  gaus_id, // corresponding gauss ids
  const float3*     __restrict__  means3D, 
  const float*      __restrict__  Cov3DInvs,
  const float*      __restrict__  opacities, 
  const uint32_t                  STEP,
  float*            __restrict__  values) {

  uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= num) return;

  point_data p = points[tidx];
  float3 fp = make_float3(voxel_size * p.x - 1.0, voxel_size * p.y - 1.0, voxel_size * p.z - 1.0);

  uint64_t gidx = gaus_id[tidx];
  float3 mean = means3D[gidx];

  // inverse covariance values of gaussian
  double c0 = (double)Cov3DInvs[6 * gidx + 0];
  double c1 = (double)Cov3DInvs[6 * gidx + 1];
  double c2 = (double)Cov3DInvs[6 * gidx + 2];
  double c3 = (double)Cov3DInvs[6 * gidx + 3];
  double c4 = (double)Cov3DInvs[6 * gidx + 4];
  double c5 = (double)Cov3DInvs[6 * gidx + 5];
  double alpha = (double)opacities[gidx];

  float step_size = voxel_size / ((STEP > 1) ? STEP - 1 : 1);

  double integral_sum = 0.0f;

  for (int i = 0; i < STEP; i++) {
    double vx = fp.x + i * step_size - mean.x;
    double a = c0 * vx * vx;
    for (int j = 0; j < STEP; j++) {
      double vy = fp.y + j * step_size - mean.y;
      double intermediate_sum = a + 2.0 * c1 * vx * vy + c3 * vy * vy;
      for (int k = 0; k < STEP; k++) {
        double vz = fp.z + k * step_size - mean.z;
	double c = 2.0 * c2 * vx * vz;
	double e = 2.0 * c4 * vy * vz;
	double f = c5 * vz * vz;
	integral_sum += exp(-0.5 * (intermediate_sum + c + e + f));
      }
    }
  }

  values[tidx] = (float)(alpha * integral_sum / (STEP * STEP * STEP));  
}


at::Tensor integrate_gs_cuda_impl(
  const at::Tensor& points,
  const at::Tensor& gaus_id,
  const at::Tensor& means3D,
  const at::Tensor& Cov3DInvs,
  const at::Tensor& opacities,
  const uint32_t level,
  const uint32_t step) {

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
  auto stream = at::cuda::getCurrentCUDAStream();

  int32_t psize = points.size(0);

  float voxel_size = 2.0f / ((float)(0x1 << level));

  at::Tensor results = at::empty({psize}, points.options().dtype(at::kFloat));

  integrate_gs_kernel << <(psize + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>> (
    psize,
    reinterpret_cast<point_data*>(points.data_ptr<short>()),
    voxel_size,
    reinterpret_cast<uint64_t*>(gaus_id.data_ptr<int64_t>()),
    reinterpret_cast<float3*>(means3D.data_ptr<float>()),
    Cov3DInvs.data_ptr<float>(),
    opacities.data_ptr<float>(),
    step,
    results.data_ptr<float>()
  );

  return results;
}

}  // namespace kaolin
