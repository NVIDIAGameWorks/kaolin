// Copyright (c) 2019,20-22 NVIDIA CORPORATION & AFFILIATES.
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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "../../utils.h"
#include "../../3d_math.cuh"
#include "../../2d_math.cuh"

#define PRIVATE_CASE_TYPE_AND_VAL(ENUM_TYPE, TYPE, TYPE_NAME, VAL, ...) \
  case ENUM_TYPE: { \
    using TYPE_NAME = TYPE; \
    const int shm_batch_size = VAL; \
    return __VA_ARGS__(); \
  }

#define DISPATCH_INPUT_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Float, float, TYPE_NAME, 512, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Double, double, TYPE_NAME, 512, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()


namespace kaolin {

namespace {

// 3D
template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static inline bool bbox_check(const V& a, const V& b, const V& c, const V& d) {
  // Return True if the point a is outside the bounding box of the triangle b-c-d.
  float y_min = min(b.y, min(c.y, d.y));
  float y_max = max(b.y, max(c.y, d.y));
  float z_min = min(b.z, min(c.z, d.z));
  float z_max = max(b.z, max(c.z, d.z));
  return (a.y < y_min || y_max < a.y || a.z < z_min || z_max < a.z);
}

template<typename V,
	 typename T = typename Vec3TypeToScalar<V>::type,
	 std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static inline T signed_volume(const V& a, const V& b, const V& c, const V& d) {
  V v = cross(b-a, c-a);
  return dot(v, d-a);
}

// 2D
template<typename V,
	 typename T = typename Vec2TypeToScalar<V>::type,
	 std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static inline T signed_area(const V& a, V b, V c) {
  // We want to have the same numerical result regardless of direction of the edge b-c
  // Otherwise you may have 0 for one face and !0 for another
  if (c.x > b.x || (b.x == c.x && c.y < b.y)) {
      return -((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y));
  } else {
      return (c.y - b.y) * (a.x - b.x) + (b.x - c.x) * (a.y - b.y);
  }
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static inline bool is_point_above_line(const V& v, const V& left_p, const V& right_p) {
  const V v1 = right_p - left_p;
  const V v2 = v - left_p;
  
  const bool output = (v1.x * v2.y - v1.y * v2.x) > 0.;
  return output;
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static inline bool is_valid_overlap_vertice(const V& v, const V& left_p, const V& right_p) {
  return is_point_above_line(v, left_p, right_p) && (left_p.x < v.x) && (right_p.x >= v.x);
}

}  // namespace

template<int shm_batch_size, typename scalar_t,
         typename vec2_t = typename ScalarTypeToVec2<scalar_t>::type,
         typename vec3_t = typename ScalarTypeToVec3<scalar_t>::type>
__global__ 
void unbatched_mesh_intersection_cuda_kernel(
    int n,
    const scalar_t* points,
    int m,
    const scalar_t* verts_1,
    const scalar_t* verts_2,
    const scalar_t* verts_3,
    scalar_t* result) {
  
  __shared__ scalar_t buf_1[shm_batch_size * 3];
  __shared__ scalar_t buf_2[shm_batch_size * 3];
  __shared__ scalar_t buf_3[shm_batch_size * 3];
  
  for (int k2 = 0; k2 < m; k2 += shm_batch_size){
    int end_k = min(m, k2 + shm_batch_size) - k2;
    for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
      buf_1[j] = verts_1[k2 * 3 + j];
      buf_2[j] = verts_2[k2 * 3 + j];
      buf_3[j] = verts_3[k2 * 3 + j];
    }
    __syncthreads();
    for (int j = threadIdx.x + blockIdx.x * blockDim.x;
         j < n; j += blockDim.x * gridDim.x) { // for points in a batch 
      vec3_t q1 = make_vec3(points[j * 3 + 0], points[j * 3 + 1], points[j * 3 + 2]);
      // Distance point outside the mesh, at +10. on x-axis
      // the mesh is normalized so it's always gonna be outside the mesh)
      vec3_t q2 = make_vec3(points[j * 3 + 0] + static_cast<scalar_t>(10.),
                            points[j * 3 + 1], points[j * 3 + 2]);

      for (int k = 0; k < end_k; k++) {
        vec3_t p1 = make_vec3(buf_1[k * 3 + 0], buf_1[k * 3 + 1], buf_1[k * 3 + 2]);
        vec3_t p2 = make_vec3(buf_2[k * 3 + 0], buf_2[k * 3 + 1], buf_2[k * 3 + 2]);
        vec3_t p3 = make_vec3(buf_3[k * 3 + 0], buf_3[k * 3 + 1], buf_3[k * 3 + 2]);

        // Can the point be project bounding box of the face?
        if (bbox_check(q1, p1, p2, p3)){continue;}
        const bool cond_1 = signed_volume(q1, p1, p2, p3) > 0.;
        const bool cond_2 = signed_volume(q2, p1, p2, p3) > 0.;
        // Is the face between the two points?
        if (cond_1 != cond_2){
          vec2_t& q1_2d = *reinterpret_cast<vec2_t*>(&q1.y);
          vec2_t& p1_2d = *reinterpret_cast<vec2_t*>(&p1.y);
          vec2_t& p2_2d = *reinterpret_cast<vec2_t*>(&p2.y);
          vec2_t& p3_2d = *reinterpret_cast<vec2_t*>(&p3.y);
          scalar_t dist_1 = signed_area(q1_2d, p1_2d, p2_2d);
          scalar_t dist_2 = signed_area(q1_2d, p2_2d, p3_2d);
          // Can the point be projected on the face?
          if (dist_1 * dist_2 >= 0) {
            scalar_t dist_3 = signed_area(q1_2d, p3_2d, p1_2d);
            if (dist_3 * dist_1 >= 0 && dist_2 * dist_3 >= 0) {
              // Is the point is projected on an edge or vertice?
              bool is_on_edge = false;
              bool is_on_vertice = false;
              vec2_t v1;
              vec2_t v2;
              vec2_t other;
              if (q1_2d == p1_2d) {
                is_on_vertice = true;
                v1 = p2_2d;
                v2 = p3_2d;
              } else if (q1_2d == p2_2d) {
                is_on_vertice = true;
                v1 = p1_2d;
                v2 = p3_2d;
              } else if (q1_2d == p3_2d) {
                is_on_vertice = true;
                v1 = p1_2d;
                v2 = p2_2d;
              } else if (dist_1 == 0.) {
                is_on_edge = true;
                v1 = p1_2d;
                v2 = p2_2d;
                other = p3_2d;
              } else if (dist_2 == 0.) {
                is_on_edge = true;
                v1 = p2_2d;
                v2 = p3_2d;
                other = p1_2d;
              } else if (dist_3 == 0.) {
                is_on_edge = true;
                v1 = p3_2d;
                v2 = p1_2d;
                other = p2_2d;
              }
              if (v1.x > v2.x || (v1.x == v2.x && v1.y > v2.y)) {
                vec2_t tmp = v1;
                v1 = v2;
                v2 = tmp;
              }
              // If the point is projected on an edge or a vertice
              // it may count multiple faces for a single point of intersection.
              // So we only count the face that is at the "bottom" of the point
              // in the rare case the aligned edge is completely vertical
              // we take the face at the left of the edge
              bool is_valid = true;
              if (is_on_edge && is_point_above_line(other, v1, v2)) {
                is_valid = false;
              } else if (is_on_vertice &&
                         !is_valid_overlap_vertice(q1_2d, v1, v2)) {
                is_valid = false;
              }
              if (is_valid) {
                //printf("    - is_valid\n");
                atomicAdd(&result[j], 1.);
              }
            }
          }
        }   
      }   
    }
    __syncthreads();
  }
    
}

void unbatched_mesh_intersection_cuda_impl(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    at::Tensor result) {
  
  const int num_points = points.size(0);
  const int num_faces = verts_1.size(0);

  DISPATCH_INPUT_TYPES(points.scalar_type(), scalar_t,
    "unbatched_mesh_intersection_cuda", [&] {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
      auto stream = at::cuda::getCurrentCUDAStream();

      const int num_threads = 512;
      const int num_blocks = (num_faces + num_threads - 1) / num_threads;
      
      unbatched_mesh_intersection_cuda_kernel<shm_batch_size><<<num_blocks, num_threads, 0, stream>>>(
          num_points,
          points.data_ptr<scalar_t>(),
          num_faces,
          verts_1.data_ptr<scalar_t>(),
          verts_2.data_ptr<scalar_t>(),
          verts_3.data_ptr<scalar_t>(),
          result.data_ptr<scalar_t>()
      );
      AT_CUDA_CHECK(cudaGetLastError());
  });
  return;
}

}  // namespace kaolin
