// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef KAOLIN_3D_MATH_CUH_
#define KAOLIN_3D_MATH_CUH_

#include <type_traits>


namespace kaolin {

// TODO(cfujitsang): at some point we need coverage of fp16 and integers but it might be trickier
template<typename T>
struct ScalarTypeToVec3 { using type = void; };
template <> struct ScalarTypeToVec3<float> { using type = float3; };
template <> struct ScalarTypeToVec3<double> { using type = double3; };

template<typename V>
struct Vec3TypeToScalar { using type = void; };
template <> struct Vec3TypeToScalar<float3> { using type = float; };
template <> struct Vec3TypeToScalar<double3> { using type = double; };

template<typename T>
struct IsVec3Type: std::false_type {};
template <> struct IsVec3Type<float3>: std::true_type {};
template <> struct IsVec3Type<double3>: std::true_type {};

__device__
static __forceinline__ float3 make_vec3(float x, float y, float z) {
  return make_float3(x, y, z);
}

__device__
static __forceinline__ double3 make_vec3(double x, double y, double z) {
  return make_double3(x, y, z);
}

template<typename V,
	 typename T = typename Vec3TypeToScalar<V>::type,
	 std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__ __forceinline__ T dot(const V a, const V b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static __forceinline__ V cross(const V& a, const V& b) {
  return make_vec3(a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x);
}

template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator- (V a, const V& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator+ (V a, const V& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator* (V a, const V& b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  return a;
}

template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator/ (V a, const V& b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  return a;
}

template<typename V, std::enable_if_t<IsVec3Type<V>::value>* = nullptr>
__device__
static __forceinline__ bool operator== (const V& a, const V& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

}

#endif  // KAOLIN_3D_MATH_CUH_
