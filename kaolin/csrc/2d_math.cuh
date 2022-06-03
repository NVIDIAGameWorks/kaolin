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

#ifndef KAOLIN_2D_MATH_CUH_
#define KAOLIN_2D_MATH_CUH_

#include <type_traits>


namespace kaolin {

// TODO(cfujitsang): at some point we need coverage of fp16 and integers but it might be trickier
template<typename T>
struct ScalarTypeToVec2 { using type = void; };
template <> struct ScalarTypeToVec2<float> { using type = float2; };
template <> struct ScalarTypeToVec2<double> { using type = double2; };

template<typename V>
struct Vec2TypeToScalar { using type = void; };
template <> struct Vec2TypeToScalar<float2> { using type = float; };
template <> struct Vec2TypeToScalar<double2> { using type = double; };

template<typename T>
struct IsVec2Type: std::false_type {};
template <> struct IsVec2Type<float2>: std::true_type {};
template <> struct IsVec2Type<double2>: std::true_type {};

__device__
static __forceinline__ float2 make_vec2(float x, float y) {
  return make_float2(x, y);
}

__device__
static __forceinline__ double2 make_vec2(double x, double y) {
  return make_double2(x, y);
}

template<typename V,
	 typename T = typename Vec2TypeToScalar<V>::type,
	 std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__ __forceinline__ T dot(const V a, const V b) {
  return a.x * b.x + a.y * b.y;
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator- (V a, const V& b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator+ (V a, const V& b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator* (V a, const V& b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static __forceinline__ V operator/ (V a, const V& b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

template<typename V, std::enable_if_t<IsVec2Type<V>::value>* = nullptr>
__device__
static __forceinline__ bool operator== (const V& a, const V& b) {
  return a.x == b.x && a.y == b.y;
}

}

#endif  // KAOLIN_2D_MATH_CUH_
