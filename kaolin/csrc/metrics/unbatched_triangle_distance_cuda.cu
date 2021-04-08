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
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <THC/THCAtomics.cuh>

#include "../utils.h"

#define BLOCK_SIZE 512

namespace kaolin {

template<typename T>
struct ScalarTypeToVec3Type { using type = float3; };

template <> struct ScalarTypeToVec3Type<float> { using type = float3; };
template <> struct ScalarTypeToVec3Type<double> { using type = double3; };

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t make_vectorize(scalar_t x, scalar_t y, scalar_t z) {
  vector_t output = {x, y, z};
  return output;
}

template <>
__device__ __forceinline__ float3 make_vectorize<float, float3>(float x, float y, float z) {
  return make_float3(x, y, z);
}

template <>
__device__ __forceinline__ double3 make_vectorize<double, double3>(double x, double y, double z) {
  return make_double3(x, y, z);
}


template <typename scalar_t, typename vector_t>
__device__ scalar_t dot(vector_t a, vector_t b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z ;
}

template<typename scalar_t, typename vector_t>
__device__ scalar_t dot2(vector_t v)
{
  return dot<scalar_t, vector_t>(v, v);
}

template<typename scalar_t>
__device__ scalar_t clamp(scalar_t x, scalar_t a, scalar_t b)
{
  return max(a, min(b, x));
}

template<typename scalar_t, typename vector_t>
__device__ vector_t cross(vector_t a, vector_t b)
{
  return make_vectorize<scalar_t, vector_t>(a.y * b.z - a.z * b.y,
                                            a.z * b.x - a.x * b.z,
                                            a.x * b.y - a.y * b.x);
}

template<typename scalar_t>
__device__ int sign(scalar_t a)
{
  if (a <= 0) {return -1;}
  else {return 1;}
}

template<typename scalar_t, typename vector_t>
__device__ vector_t operator* (vector_t a, scalar_t b)
{
  return make_vectorize<scalar_t, vector_t>(a.x * b, a.y * b, a.z * b);
}

template<typename scalar_t, typename vector_t>
__device__ vector_t operator+ (vector_t a, scalar_t b)
{
  return make_vectorize<scalar_t, vector_t>(a.x + b, a.y + b, a.z + b);
}

template<typename scalar_t, typename vector_t>
__device__ vector_t operator/ (vector_t a, scalar_t b)
{
  return make_vectorize<scalar_t, vector_t>(a.x / b, a.y / b, a.z / b);
}

template<typename scalar_t, typename vector_t>
__device__ vector_t add(vector_t a, vector_t b)
{
  return make_vectorize<scalar_t, vector_t>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename scalar_t, typename vector_t>
__device__ vector_t substract(vector_t a, vector_t b)
{
  return make_vectorize<scalar_t, vector_t>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename scalar_t, typename vector_t>
__device__ int signage(vector_t a, vector_t b, vector_t c)
{
  return sign<scalar_t>(dot<scalar_t, vector_t>(cross<scalar_t, vector_t>(a, b), c));
}

template<typename scalar_t, typename vector_t>
__device__ scalar_t edge_distance(vector_t a, vector_t b)
{
  return dot2<scalar_t, vector_t>(substract<scalar_t, vector_t>(a * clamp<scalar_t>(dot<scalar_t, vector_t>(a, b) / dot2<scalar_t, vector_t> (a), 0.0, 1.0), b));
}

template<typename scalar_t, typename vector_t>
__device__ scalar_t plane_distance(vector_t a, vector_t b)
{
  return dot<scalar_t, vector_t>(a, b) * dot<scalar_t, vector_t>(a, b) / dot2<scalar_t, vector_t>(a);
}

template<typename scalar_t, typename vector_t>
__device__ void compute_edge_backward(vector_t vab, vector_t pb, scalar_t* grad_input_va, scalar_t* grad_input_vb, scalar_t* grad_input_p, int64_t index, int64_t mesh_idx, scalar_t grad)
{
  // variable used in forward pass
  scalar_t l = dot<scalar_t, vector_t>(vab, pb);
  scalar_t m = dot2<scalar_t, vector_t>(vab);// variable used in forward pass
  scalar_t k = l / m;
  scalar_t j = clamp<scalar_t>(k, 0.0, 1.0);
  vector_t i = substract<scalar_t, vector_t>(vab * j, pb);
  scalar_t h = dot2<scalar_t, vector_t>(i);

  vector_t i_bar = make_vectorize<scalar_t, vector_t>(2 * i.x, 2 * i.y, 2 * i.z) * grad;  // horizontal vector

  scalar_t j_bar = dot<scalar_t, vector_t>(i_bar, vab);

  scalar_t dj_dk = (k > 0 && k < 1) ? 1:0;

  scalar_t k_bar = j_bar * dj_dk;

  scalar_t m_bar = k_bar * (- l / (m * m));

  scalar_t l_bar = k_bar * (1 / m);

  // derivative of pb
  vector_t dl_dpb = vab; //vertical vector
  vector_t di_dpb = make_vectorize<scalar_t, vector_t>(-i_bar.x, -i_bar.y, -i_bar.z);

  vector_t pb_bar = add<scalar_t, vector_t>(dl_dpb * l_bar, di_dpb); // vertical vector

  vector_t p_bar = pb_bar;

  vector_t dm_dvab = make_vectorize<scalar_t, vector_t>(vab.x, vab.y, vab.z) * 2;  // horizontal vector
  vector_t dl_dvab = make_vectorize<scalar_t, vector_t>(pb.x, pb.y, pb.z);  // horizontal vector
  vector_t di_dvab = make_vectorize<scalar_t, vector_t>(i_bar.x, i_bar.y, i_bar.z) * j; // horizontal vector

  vector_t vab_bar = add<scalar_t, vector_t>(add<scalar_t, vector_t>(dm_dvab * m_bar, dl_dvab * l_bar), di_dvab);  // horizontal vector

  vector_t va_bar = vab_bar;
  vector_t vb_bar = make_vectorize<scalar_t, vector_t>(-vab_bar.x - pb_bar.x, -vab_bar.y - pb_bar.y, -vab_bar.z - pb_bar.z);

  grad_input_p[index * 3] = p_bar.x;
  grad_input_p[index * 3 + 1] = p_bar.y;
  grad_input_p[index * 3 + 2] = p_bar.z;

  atomicAdd(&(grad_input_va[mesh_idx * 3]), va_bar.x);
  atomicAdd(&(grad_input_va[mesh_idx * 3 + 1]), va_bar.y);
  atomicAdd(&(grad_input_va[mesh_idx * 3 + 2]), va_bar.z);

  atomicAdd(&(grad_input_vb[mesh_idx * 3]), vb_bar.x);
  atomicAdd(&(grad_input_vb[mesh_idx * 3 + 1]), vb_bar.y);
  atomicAdd(&(grad_input_vb[mesh_idx * 3 + 2]), vb_bar.z);
}


template<typename scalar_t, typename vector_t>
__global__  void UnbatchedTriangleDistanceKernel(
  const scalar_t* points,
  const scalar_t* verts_1,
  const scalar_t* verts_2,
  const scalar_t* verts_3,
  int n,
  int m,
  scalar_t* result,
  int64_t* result_i,
  int* result_t)
{
  const int batch = 512;
  __shared__ scalar_t buf_1[batch * 3];
  __shared__ scalar_t buf_2[batch * 3];
  __shared__ scalar_t buf_3[batch * 3];

  for (int k2 = 0; k2 < m; k2 += batch){
      int end_k = min(m, k2 + batch) - k2;
        
    for (int j = threadIdx.x; j < end_k * 3;j += blockDim.x){
      buf_1[j] = verts_1[k2 * 3 + j];
      buf_2[j] = verts_2[k2 * 3 + j];
      buf_3[j] = verts_3[k2 * 3 + j];
    }
    __syncthreads();
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y){ // for points in a batch 

      vector_t p = make_vectorize<scalar_t, vector_t>(points[j * 3 + 0], points[j * 3 + 1], points[j * 3 + 2]);
      int64_t best_i = 0;
      int best_t = 0;
      scalar_t best = 10000;
      for (int k = 0; k < end_k; k++){

        vector_t v1 = make_vectorize<scalar_t, vector_t>(buf_1[k * 3 + 0], buf_1[k * 3 + 1], buf_1[k * 3 + 2]);
        vector_t v2 = make_vectorize<scalar_t, vector_t>(buf_2[k * 3 + 0], buf_2[k * 3 + 1], buf_2[k * 3 + 2]);
        vector_t v3 = make_vectorize<scalar_t, vector_t>(buf_3[k * 3 + 0], buf_3[k * 3 + 1], buf_3[k * 3 + 2]);

        vector_t v21 = substract<scalar_t, vector_t>(v2, v1); //edge length between vertices 1 and 2
        vector_t v32 = substract<scalar_t, vector_t>(v3, v2); //edge length between vertices 2 and 3
        vector_t v13 = substract<scalar_t, vector_t>(v1, v3); //edge length between vertices 1 and 3

        vector_t p1 = substract<scalar_t, vector_t>(p, v1);  //distance between point and vertices 1
        vector_t p2 = substract<scalar_t, vector_t>(p, v2);  //distance between point and vertices 2 
        vector_t p3 = substract<scalar_t, vector_t>(p, v3);  //distance between point and vertices 3

        vector_t normal = cross<scalar_t, vector_t>(v21, v13);  //normal of a triangle's surface

        scalar_t sign_cond = signage<scalar_t, vector_t>(v21, normal, p1) + signage<scalar_t, vector_t>(v32, normal, p2) + signage<scalar_t, vector_t>(v13, normal, p3);

        scalar_t dist = 100; 
        int type = 0;

        if (sign_cond < 2.0) {  //if sign condition is greater or equal to 2, which means the point is closest to the surface, to neither of the edges.
          scalar_t dist1 = edge_distance<scalar_t, vector_t>(v21, p1);
          scalar_t dist2 = edge_distance<scalar_t, vector_t>(v32, p2);
          scalar_t dist3 = edge_distance<scalar_t, vector_t>(v13, p3);

          if (dist1 <= dist2 && dist1 <= dist3){
            dist = dist1;
            type = 0;
          } else if (dist2 <= dist1 && dist2 <= dist3){
            dist = dist2;
            type = 1;
          } else {
            dist = dist3;
            type = 2;
          }
        } else {
          dist = plane_distance<scalar_t, vector_t>(normal, p1);
          type = 3;
        }
        if (k == 0 || dist < best){
          best = dist;
          best_i = k + k2;
          best_t = type;
        }
      }
      if (k2 == 0 || result[j] > best){
        result[j] = best;
        result_i[j] = best_i;
        result_t[j] = best_t;
      }
    }
    __syncthreads();
  }
}


template<typename scalar_t, typename vector_t>
__global__ void UnbatchedTriangleDistanceBackwardKernel(
  const scalar_t* grad_output,
  const scalar_t* points,
  const scalar_t* verts_1,
  const scalar_t* verts_2,
  const scalar_t* verts_3,
  const int n, // num of points
  const int m, // num of faces
  const int64_t* idx,
  const int* dist_type,
  scalar_t* grad_input_p,
  scalar_t* grad_input_v1,
  scalar_t* grad_input_v2,
  scalar_t* grad_input_v3) 
{
  
  for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < n; point_id += blockDim.x) {
    int type = dist_type[point_id];
    int64_t mesh_idx = idx[point_id];

    // printf("mesh id is %i\n", mesh_idx);
    vector_t p = make_vectorize<scalar_t, vector_t>(points[point_id * 3], points[point_id * 3 + 1], points[point_id * 3 + 2]);

    vector_t v1 = make_vectorize<scalar_t, vector_t>(verts_1[mesh_idx * 3], verts_1[mesh_idx * 3 + 1], verts_1[mesh_idx * 3 + 2]);
    vector_t v2 = make_vectorize<scalar_t, vector_t>(verts_2[mesh_idx * 3], verts_2[mesh_idx * 3 + 1], verts_2[mesh_idx * 3 + 2]);
    vector_t v3 = make_vectorize<scalar_t, vector_t>(verts_3[mesh_idx * 3], verts_3[mesh_idx * 3 + 1], verts_3[mesh_idx * 3 + 2]);

    vector_t v21 = substract<scalar_t, vector_t>(v2, v1); //edge length between vertices 1 and 2
    vector_t v32 = substract<scalar_t, vector_t>(v3, v2); //edge length between vertices 2 and 3
    vector_t v13 = substract<scalar_t, vector_t>(v1, v3); //edge length between vertices 1 and 3

    vector_t p1 = substract<scalar_t, vector_t>(p, v1);  //distance between point and vertices 1
    vector_t p2 = substract<scalar_t, vector_t>(p, v2);  //distance between point and vertices 2 
    vector_t p3 = substract<scalar_t, vector_t>(p, v3);  //distance between point and vertices 3

    scalar_t grad = grad_output[point_id];

    vector_t result;

    // Calculate the grad_input_p part
    if (type == 0) { // closest to edge v21
      compute_edge_backward(v21, p1, grad_input_v2, grad_input_v1, grad_input_p, point_id, mesh_idx, grad);
    } else if (type == 1) {  // closest to edge v32
      compute_edge_backward(v32, p2, grad_input_v3, grad_input_v2, grad_input_p, point_id, mesh_idx, grad);
    } else if (type == 2) {  // closest to edge v13
      compute_edge_backward(v13, p3, grad_input_v1, grad_input_v3, grad_input_p, point_id, mesh_idx, grad);
    } else if (type == 3) {  // closest to the surface
      // variable used in forward pass
      vector_t i = cross<scalar_t, vector_t>(v21, v13);
      scalar_t j = dot2<scalar_t, vector_t>(i);
      scalar_t k = dot<scalar_t,vector_t>(i, p1);
      scalar_t l = (k * k) / j;

      scalar_t k_bar = ((2 * k) / j) * grad;
      scalar_t j_bar = - ((k / j) * (k / j)) * grad;

      vector_t dk_di = p1;

      vector_t dj_di = make_vectorize<scalar_t, vector_t>(2 * i.x, 2 * i.y, 2 * i.z);

      vector_t i_bar = add<scalar_t, vector_t>(dj_di * j_bar, dk_di * k_bar);  // horizontal vector
      
      vector_t dk_dp1 = make_vectorize<scalar_t, vector_t>(i.x, i.y, i.z);  // horizontal vector

      vector_t p1_bar = dk_dp1 * k_bar; // horizontal vector

      vector_t di_dv21_x = make_vectorize<scalar_t, vector_t>(0, -v13.z, v13.y);  // vertical vector
      vector_t di_dv21_y = make_vectorize<scalar_t, vector_t>(v13.z, 0, -v13.x);  // vertical vector
      vector_t di_dv21_z = make_vectorize<scalar_t, vector_t>(-v13.y, v13.x, 0);  // vertical vector

      scalar_t v21_bar_x = dot<scalar_t, vector_t>(i_bar, di_dv21_x);  
      scalar_t v21_bar_y = dot<scalar_t, vector_t>(i_bar, di_dv21_y); 
      scalar_t v21_bar_z = dot<scalar_t, vector_t>(i_bar, di_dv21_z); 
      
      vector_t v21_bar = make_vectorize<scalar_t, vector_t>(v21_bar_x, v21_bar_y, v21_bar_z); // horizontal vector

      vector_t di_dv13_x = make_vectorize<scalar_t, vector_t>(0, v21.z, -v21.y);  // vertical vector
      vector_t di_dv13_y = make_vectorize<scalar_t, vector_t>(-v21.z, 0, v21.x);  // vertical vector
      vector_t di_dv13_z = make_vectorize<scalar_t, vector_t>(v21.y, -v21.x, 0);  // vertical vector

      scalar_t v13_bar_x = dot<scalar_t, vector_t>(i_bar, di_dv13_x);  
      scalar_t v13_bar_y = dot<scalar_t, vector_t>(i_bar, di_dv13_y); 
      scalar_t v13_bar_z = dot<scalar_t, vector_t>(i_bar, di_dv13_z);

      vector_t v13_bar = make_vectorize<scalar_t, vector_t>(v13_bar_x, v13_bar_y, v13_bar_z); // horizontal vector

      vector_t v1_bar_v21 = make_vectorize<scalar_t, vector_t>(-v21_bar.x, -v21_bar.y, -v21_bar.z); // horizontal vector

      vector_t v1_bar_v13 = make_vectorize<scalar_t, vector_t>(v13_bar.x, v13_bar.y, v13_bar.z);  // horizontal vector

      vector_t v1_bar_p1 = make_vectorize<scalar_t, vector_t>(-p1_bar.x, -p1_bar.y, -p1_bar.z);  // horizontal vector

      vector_t v1_bar = add<scalar_t, vector_t>(add<scalar_t, vector_t>(v1_bar_v13, v1_bar_v21), v1_bar_p1);  // horizontal vector
      
      vector_t v2_bar = make_vectorize<scalar_t, vector_t>(v21_bar.x, v21_bar.y, v21_bar.z); // horizontal vector

      vector_t v3_bar = make_vectorize<scalar_t, vector_t>(-v13_bar.x, -v13_bar.y, -v13_bar.z);  // horizontal vector

      vector_t p_bar = p1_bar;

      grad_input_p[point_id * 3] = p_bar.x;
      grad_input_p[point_id * 3 + 1] = p_bar.y;
      grad_input_p[point_id * 3 + 2] = p_bar.z;

      atomicAdd(&(grad_input_v1[mesh_idx * 3]), v1_bar.x);
      atomicAdd(&(grad_input_v1[mesh_idx * 3 + 1]), v1_bar.y);
      atomicAdd(&(grad_input_v1[mesh_idx * 3 + 2]), v1_bar.z);

      atomicAdd(&(grad_input_v2[mesh_idx * 3]), v2_bar.x);
      atomicAdd(&(grad_input_v2[mesh_idx * 3 + 1]), v2_bar.y);
      atomicAdd(&(grad_input_v2[mesh_idx * 3 + 2]), v2_bar.z);

      atomicAdd(&(grad_input_v3[mesh_idx * 3]), v3_bar.x);
      atomicAdd(&(grad_input_v3[mesh_idx * 3 + 1]), v3_bar.y);
      atomicAdd(&(grad_input_v3[mesh_idx * 3 + 2]), v3_bar.z);
    }
    __syncthreads();
  } 
}

void unbatched_triangle_distance_forward_cuda_kernel_launcher(
    const at::Tensor points, 
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor dist1,  
    const at::Tensor idx1, 
    const at::Tensor type1) {
  DISPATCH_NUM_TYPES(points.scalar_type(), scalar_t, "unbatched_triangle_distance", [&] {
  using vector_t = ScalarTypeToVec3Type<scalar_t>::type;
  UnbatchedTriangleDistanceKernel<scalar_t, vector_t><<<dim3(32,16,1),512>>>(
      points.data_ptr<scalar_t>(), verts_1.data_ptr<scalar_t>(), verts_2.data_ptr<scalar_t>(),
      verts_3.data_ptr<scalar_t>(), points.size(0), verts_1.size(0), dist1.data_ptr<scalar_t>(),
      idx1.data_ptr<int64_t>(), type1.data_ptr<int>());
  });
}

void unbatched_triangle_distance_backward_cuda_kernel_launcher(
  const at::Tensor grad_output,
  const at::Tensor points, 
  const at::Tensor verts_1,
  const at::Tensor verts_2,
  const at::Tensor verts_3,
  const at::Tensor idx, 
  const at::Tensor dist_type,
  const at::Tensor grad_input_p,
  const at::Tensor grad_input_v1,
  const at::Tensor grad_input_v2,
  const at::Tensor grad_input_v3) {
DISPATCH_NUM_TYPES(points.scalar_type(), scalar_t, "unbatched_triangle_distance", [&] {
  using vector_t = ScalarTypeToVec3Type<scalar_t>::type;
  int n = points.size(0);
  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  UnbatchedTriangleDistanceBackwardKernel<scalar_t, vector_t><<<num_blocks, BLOCK_SIZE>>>(
      grad_output.data_ptr<scalar_t>(), points.data_ptr<scalar_t>(), verts_1.data_ptr<scalar_t>(),
      verts_2.data_ptr<scalar_t>(), verts_3.data_ptr<scalar_t>(), points.size(0), verts_1.size(0),
      idx.data_ptr<int64_t>(), dist_type.data_ptr<int>(), grad_input_p.data_ptr<scalar_t>(),
      grad_input_v1.data_ptr<scalar_t>(), grad_input_v2.data_ptr<scalar_t>(), grad_input_v3.data_ptr<scalar_t>());
  });
}

}  // namespace kaolin
