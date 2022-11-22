// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "../utils.h"

#define PRIVATE_CASE_TYPE_AND_VAL(ENUM_TYPE, TYPE, TYPE_NAME, VAL, ...) \
  case ENUM_TYPE: { \
    using TYPE_NAME = TYPE; \
    const int num_threads = VAL; \
    return __VA_ARGS__(); \
  }


#define DISPATCH_INPUT_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Float, float, TYPE_NAME, 1024, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Double, double, TYPE_NAME, 512, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()

namespace kaolin {

template<typename T>
struct ScalarTypeToVec3 { using type = void; };
template <> struct ScalarTypeToVec3<float> { using type = float3; };
template <> struct ScalarTypeToVec3<double> { using type = double3; };

template<typename T>
struct Vec3TypeToScalar { using type = void; };
template <> struct Vec3TypeToScalar<float3> { using type = float; };
template <> struct Vec3TypeToScalar<double3> { using type = double; };

__device__ __forceinline__ float3 make_vector(float x, float y, float z) {
  return make_float3(x, y, z);
}

__device__ __forceinline__ double3 make_vector(double x, double y, double z) {
  return make_double3(x, y, z);
}

template <typename vector_t>
__device__ __forceinline__ typename Vec3TypeToScalar<vector_t>::type dot(vector_t a, vector_t b) {
  return a.x * b.x + a.y * b.y + a.z * b.z ;
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ scalar_t dot2(vector_t v) {
  return dot<scalar_t, vector_t>(v, v);
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t clamp(scalar_t x, scalar_t a, scalar_t b) {
  return max(a, min(b, x));
}

template<typename vector_t>
__device__ __forceinline__ vector_t cross(vector_t a, vector_t b) {
  return make_vector(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

template<typename scalar_t>
__device__ __forceinline__ int sign(scalar_t a) {
  if (a <= 0) {return -1;}
  else {return 1;}
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator* (vector_t a, scalar_t b) {
  return make_vector(a.x * b, a.y * b, a.z * b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator* (vector_t a, vector_t b) {
  return make_vector(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator+ (vector_t a, scalar_t b) {
  return make_vector(a.x + b, a.y + b, a.z + b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator+ (vector_t a, vector_t b) {
  return make_vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a, scalar_t b) {
  return make_vector(a.x - b, a.y - b, a.z - b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a, vector_t b) {
  return make_vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator/ (vector_t a, scalar_t b) {
  return make_vector(a.x / b, a.y / b, a.z / b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator/ (vector_t a, vector_t b) {
  return make_vector(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename vector_t>
__device__ __forceinline__ typename Vec3TypeToScalar<vector_t>::type project_edge(
    vector_t vertex, vector_t edge, vector_t point) {
  typedef typename Vec3TypeToScalar<vector_t>::type scalar_t;
  vector_t point_vec = point - vertex;
  scalar_t length = dot(edge, edge);
  return dot(point_vec, edge) / length;
}

template<typename vector_t>
__device__ __forceinline__ vector_t project_plane(vector_t vertex, vector_t normal, vector_t point) {
  typedef typename Vec3TypeToScalar<vector_t>::type scalar_t;
  scalar_t inv_len = rsqrt(dot(normal, normal));
  vector_t unit_normal = normal * inv_len;
  scalar_t dist = (point.x - vertex.x) * unit_normal.x + \
                  (point.y - vertex.y) * unit_normal.y + \
                  (point.z - vertex.z) * unit_normal.z;
  return point - (unit_normal * dist);
}

template<typename scalar_t>
__device__ __forceinline__ bool in_range(scalar_t a) {
  return (a <= 1 && a >= 0);
}

template<typename vector_t>
__device__ __forceinline__ bool is_above(vector_t vertex, vector_t edge, vector_t normal, vector_t point) {
  vector_t edge_normal = cross(normal, edge);
  return dot(edge_normal, point - vertex) > 0;
}

template<typename vector_t>
__device__ __forceinline__ bool is_not_above(vector_t vertex, vector_t edge, vector_t normal,
                                     vector_t point) {
  vector_t edge_normal = cross(normal, edge);
  return dot(edge_normal, point - vertex) <= 0;
}


template<typename vector_t>
__device__ __forceinline__ vector_t point_at(vector_t vertex, vector_t edge, float t) {
  return vertex + (edge * t);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ void compute_edge_backward(
    vector_t vab,
    vector_t pb,
    scalar_t* grad_input_va,
    scalar_t* grad_input_vb,
    scalar_t* grad_input_p,
    int64_t index,
    scalar_t grad) {
  // variable used in forward pass
  scalar_t l = dot(vab, pb);
  scalar_t m = dot(vab, vab);// variable used in forward pass
  scalar_t k = l / m;
  scalar_t j = clamp<scalar_t>(k, 0.0, 1.0);
  vector_t i = (vab * j) - pb;
  scalar_t h = dot(i, i);

  vector_t i_bar = i * grad;  // horizontal vector

  scalar_t j_bar = dot(i_bar, vab);

  scalar_t dj_dk = (k > 0 && k < 1) ? 1:0;

  scalar_t k_bar = j_bar * dj_dk;

  scalar_t m_bar = k_bar * (- l / (m * m));

  scalar_t l_bar = k_bar * (1 / m);

  // derivative of pb
  vector_t dl_dpb = vab; //vertical vector
  vector_t di_dpb = make_vector(-i_bar.x, -i_bar.y, -i_bar.z);

  vector_t pb_bar = dl_dpb * l_bar + di_dpb; // vertical vector

  vector_t p_bar = pb_bar;

  vector_t dm_dvab = vab * static_cast<scalar_t>(2.);  // horizontal vector
  vector_t dl_dvab = pb;  // horizontal vector
  vector_t di_dvab = i_bar * j; // horizontal vector

  vector_t vab_bar = ((dm_dvab * m_bar) + (dl_dvab * l_bar)) + di_dvab;  // horizontal vector

  vector_t va_bar = vab_bar;
  vector_t vb_bar = make_vector(-vab_bar.x - pb_bar.x, -vab_bar.y - pb_bar.y, -vab_bar.z - pb_bar.z);

  grad_input_p[index * 3] = p_bar.x;
  grad_input_p[index * 3 + 1] = p_bar.y;
  grad_input_p[index * 3 + 2] = p_bar.z;

  atomicAdd(&(grad_input_va[0]), va_bar.x);
  atomicAdd(&(grad_input_va[1]), va_bar.y);
  atomicAdd(&(grad_input_va[2]), va_bar.z);

  atomicAdd(&(grad_input_vb[0]), vb_bar.x);
  atomicAdd(&(grad_input_vb[1]), vb_bar.y);
  atomicAdd(&(grad_input_vb[2]), vb_bar.z);
}



template<typename scalar_t, typename vector_t, int BLOCK_SIZE>
__global__ void unbatched_triangle_distance_forward_cuda_kernel(
    const vector_t* points,
    const vector_t* vertices,
    int num_points,
    int num_faces,
    scalar_t* out_dist,
    int64_t* out_closest_face_idx,
    int* out_dist_type) {
  __shared__ vector_t shm[BLOCK_SIZE * 3];

  for (int start_face_idx = 0; start_face_idx < num_faces; start_face_idx += BLOCK_SIZE) {
    int num_faces_iter = min(num_faces - start_face_idx, BLOCK_SIZE);
    for (int j = threadIdx.x; j < num_faces_iter * 3; j += blockDim.x) {
      shm[j] = vertices[start_face_idx * 3 + j];
    }
    __syncthreads();
    for (int point_idx = threadIdx.x + blockDim.x * blockIdx.x; point_idx < num_points;
         point_idx += blockDim.x * gridDim.x) {
      vector_t p = points[point_idx];
      int best_face_idx = 0;
      int best_dist_type = 0;
      scalar_t best_dist = INFINITY;
      for (int sub_face_idx = 0; sub_face_idx < num_faces_iter; sub_face_idx++) {
        vector_t closest_point;
        int dist_type = 0;

        vector_t v1 = shm[sub_face_idx * 3];
        vector_t v2 = shm[sub_face_idx * 3 + 1];
        vector_t v3 = shm[sub_face_idx * 3 + 2];

        vector_t e12 = v2 - v1;
        vector_t e23 = v3 - v2;
        vector_t e31 = v1 - v3;
        vector_t normal = cross(v1 - v2, e31);
        scalar_t uab = project_edge(v1, e12, p);
        scalar_t uca = project_edge(v3, e31, p);
        if (uca > 1 && uab < 0) {
          closest_point = v1;
          dist_type = 1;
        } else {
          scalar_t ubc = project_edge(v2, e23, p);
          if (uab > 1 && ubc < 0) {
            closest_point = v2;
            dist_type = 2;
          } else if (ubc > 1 && uca < 0) {
            closest_point = v3;
            dist_type = 3;
          } else {
            if (in_range(uab) && (is_not_above(v1, e12, normal, p))) {
              closest_point = point_at(v1, e12, uab);
              dist_type = 4;
            } else if (in_range(ubc) && (is_not_above(v2, e23, normal, p))) {
              closest_point = point_at(v2, e23, ubc);
              dist_type = 5;
            } else if (in_range(uca) && (is_not_above(v3, e31, normal, p))) {
              closest_point = point_at(v3, e31, uca);
              dist_type = 6;
            } else {
              closest_point = project_plane(v1, normal, p);
              dist_type = 0;
            }
          }
        }
        vector_t dist_vec = p - closest_point;
        float dist = dot(dist_vec, dist_vec);
        if (sub_face_idx == 0 || best_dist > dist) {
          best_dist = dist;
          best_dist_type = dist_type;
          best_face_idx = start_face_idx + sub_face_idx;
        }
      }
      if (start_face_idx == 0 || out_dist[point_idx] > best_dist) {
        out_dist[point_idx] = best_dist;
        out_closest_face_idx[point_idx] = best_face_idx;
        out_dist_type[point_idx] = best_dist_type;
      }
    }
    __syncthreads();
  }
}

template<typename scalar_t, typename vector_t>
__global__ void unbatched_triangle_distance_backward_cuda_kernel(
    const scalar_t* grad_dist,
    const vector_t* points,
    const vector_t* vertices,
    int64_t* closest_face_idx,
    int* dist_type,
    int num_points,
    int num_faces,
    scalar_t* grad_points,
    scalar_t* grad_face_vertices) {
  for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points;
       point_id += blockDim.x * gridDim.x) {
    int type = dist_type[point_id];
    int64_t face_id = closest_face_idx[point_id];
    vector_t p = points[point_id];
    vector_t v1 = vertices[face_id * 3];
    vector_t v2 = vertices[face_id * 3 + 1];
    vector_t v3 = vertices[face_id * 3 + 2];
    vector_t e12 = v2 - v1;
    vector_t e23 = v3 - v2;
    vector_t e31 = v1 - v3;
    scalar_t grad_out = 2. * grad_dist[point_id];
    if (type == 0) {  // plane distance
      vector_t point_vec = p - v1;
      vector_t e21 = v1 - v2;
      vector_t normal = cross(e21, e31);
      scalar_t len = sqrt(dot(normal, normal));
      vector_t unit_normal = normal / len;
      scalar_t dist = dot(point_vec, unit_normal);

      vector_t grad_dist_vec = unit_normal * (dist * grad_out);
      scalar_t grad_dist = dot(unit_normal, grad_dist_vec);
      vector_t grad_point_vec = unit_normal * grad_dist;
      vector_t grad_unit_normal = grad_dist_vec * dist + point_vec * grad_dist;
      scalar_t grad_len = - dot(normal, grad_unit_normal) / (len * len);
      scalar_t grad_dot2_normal = grad_len / (2 * sqrt(dot(normal, normal)));
      vector_t grad_normal = (grad_unit_normal / len) + \
                             normal * (grad_dot2_normal * static_cast<scalar_t>(2.));
      vector_t grad_e31 = cross(grad_normal, e21);
      vector_t grad_e21 = cross(e31, grad_normal);

      grad_points[point_id * 3] = grad_point_vec.x;
      grad_points[point_id * 3 + 1] = grad_point_vec.y;
      grad_points[point_id * 3 + 2] = grad_point_vec.z;
      vector_t tmp = grad_e31 + grad_e21 - grad_point_vec;
      atomicAdd(&(grad_face_vertices[face_id * 9]), tmp.x);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 1]), tmp.y);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 2]), tmp.z);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 3]), -grad_e21.x);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 4]), -grad_e21.y);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 5]), -grad_e21.z);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 6]), -grad_e31.x);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 7]), -grad_e31.y);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 8]), -grad_e31.z);
    } else if (type == 1) {  // distance to v1
      vector_t grad_dist_vec = (p - v1) * grad_out;
      atomicAdd(&(grad_face_vertices[face_id * 9]), -grad_dist_vec.x);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 1]), -grad_dist_vec.y);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 2]), -grad_dist_vec.z);
      grad_points[point_id * 3] = grad_dist_vec.x;
      grad_points[point_id * 3 + 1] = grad_dist_vec.y;
      grad_points[point_id * 3 + 2] = grad_dist_vec.z;
    } else if (type == 2) {  // distance to v2
      vector_t grad_dist_vec = (p - v2) * grad_out;
      atomicAdd(&(grad_face_vertices[face_id * 9 + 3]), -grad_dist_vec.x);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 4]), -grad_dist_vec.y);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 5]), -grad_dist_vec.z);
      grad_points[point_id * 3] = grad_dist_vec.x;
      grad_points[point_id * 3 + 1] = grad_dist_vec.y;
      grad_points[point_id * 3 + 2] = grad_dist_vec.z;
    } else if (type == 3) {  // distance to v3
      vector_t grad_dist_vec = (p - v3) * grad_out;
      atomicAdd(&(grad_face_vertices[face_id * 9 + 6]), -grad_dist_vec.x);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 7]), -grad_dist_vec.y);
      atomicAdd(&(grad_face_vertices[face_id * 9 + 8]), -grad_dist_vec.z);
      grad_points[point_id * 3] = grad_dist_vec.x;
      grad_points[point_id * 3 + 1] = grad_dist_vec.y;
      grad_points[point_id * 3 + 2] = grad_dist_vec.z;
    } else if (type == 4) {  // distance to e12
      compute_edge_backward(e12, p - v1,
                            &(grad_face_vertices[face_id * 9 + 3]),
                            &(grad_face_vertices[face_id * 9]),
                            grad_points, point_id, grad_out);
    } else if (type == 5) {  // distance to e23
      compute_edge_backward(e23, p - v2,
                            &(grad_face_vertices[face_id * 9 + 6]),
                            &(grad_face_vertices[face_id * 9 + 3]),
                            grad_points, point_id, grad_out);
    } else {  // distance to e31
      compute_edge_backward(e31, p - v3,
                            &(grad_face_vertices[face_id * 9]),
                            &(grad_face_vertices[face_id * 9 + 6]),
                            grad_points, point_id, grad_out);
    }
  }

}

void unbatched_triangle_distance_forward_cuda_impl(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor face_idx,
    at::Tensor dist_type) {
  const int num_threads = 512;
  const int num_points = points.size(0);
  const int num_blocks = (num_points + num_threads - 1) / num_threads;
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(),
                             "unbatched_triangle_distance_forward_cuda", [&] {
    using vector_t = ScalarTypeToVec3<scalar_t>::type;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    auto stream = at::cuda::getCurrentCUDAStream();
    unbatched_triangle_distance_forward_cuda_kernel<scalar_t, vector_t, 512><<<
      num_blocks, num_threads, 0, stream>>>(
        reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(face_vertices.data_ptr<scalar_t>()),
        points.size(0),
        face_vertices.size(0),
        dist.data_ptr<scalar_t>(),
        face_idx.data_ptr<int64_t>(),
        dist_type.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
  });
}

void unbatched_triangle_distance_backward_cuda_impl(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points,
    at::Tensor grad_face_vertices) {

  DISPATCH_INPUT_TYPES(points.scalar_type(), scalar_t,
                       "unbatched_triangle_distance_backward_cuda", [&] {
    const int num_points = points.size(0);
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    using vector_t = ScalarTypeToVec3<scalar_t>::type;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    auto stream = at::cuda::getCurrentCUDAStream();
    unbatched_triangle_distance_backward_cuda_kernel<scalar_t, vector_t><<<
      num_blocks, num_threads, 0, stream>>>(
        grad_dist.data_ptr<scalar_t>(),
        reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(face_vertices.data_ptr<scalar_t>()),
        face_idx.data_ptr<int64_t>(),
        dist_type.data_ptr<int32_t>(),
        points.size(0),
        face_vertices.size(0),
        grad_points.data_ptr<scalar_t>(),
        grad_face_vertices.data_ptr<scalar_t>());
    AT_CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace kaolin

#undef PRIVATE_CASE_TYPE_AND_VAL
#undef DISPATCH_INPUT_TYPES
