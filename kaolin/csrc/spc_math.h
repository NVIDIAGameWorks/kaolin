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

#ifndef KAOLIN_SPC_MATH_H_
#define KAOLIN_SPC_MATH_H_

#ifdef WITH_CUDA

#include <stdint.h>
#include <vector_types.h>
#include <vector_functions.h>

#ifndef __CUDACC__

static inline __host__ __device__ double rsqrt(double a) {
  return 1. / sqrt(a);
}

static inline __host__ __device__ float rsqrtf(float a) {
  return 1. / sqrtf(a);
}
#endif

// limit induced by use of short for point coordinates
#define KAOLIN_SPC_MAX_LEVELS           15

typedef unsigned char       uchar;
typedef unsigned short      ushort;
typedef unsigned int        uint;

typedef uint64_t            morton_code;
typedef short3              point_data;
typedef ulong3               tri_index;

typedef struct {
  float m[3][3];
} float3x3;

typedef struct {
  float m[4][4];
} float4x4;

static __inline__ __host__ __device__ point_data make_point_data(short x, short y, short z) {
  point_data p;
  p.x = x; p.y = y; p.z = z;
  return p;
}

static __inline__ __host__ __device__ point_data add_point_data(point_data a, point_data b) {
  point_data p;
  p.x = a.x + b.x;
  p.y = a.y + b.y;
  p.z = a.z + b.z;
  return p;
}

static __inline__ __host__ __device__ point_data sub_point_data(point_data a, point_data b) {
  point_data p;
  p.x = a.x - b.x;
  p.y = a.y - b.y;
  p.z = a.z - b.z;
  return p;
}

static __inline__ __host__ __device__ point_data mul_point_data(int s, point_data a) {
  point_data p;
  p.x = s * a.x;
  p.y = s * a.y;
  p.z = s * a.z;
  return p;
}

static __inline__ __host__ __device__ point_data div_point_data(point_data a, int s) {
  point_data p;
  p.x = a.x / s;
  p.y = a.y / s;
  p.z = a.z / s;
  return p;
}

static __inline__ __host__ __device__ morton_code to_morton(point_data V) {
  morton_code mcode = 0;

  for (uint i = 0; i < KAOLIN_SPC_MAX_LEVELS; i++) {
    uint i2 = i + i;
    morton_code x = V.x;
    morton_code y = V.y;
    morton_code z = V.z;

    mcode |= (z&(0x1 << i)) << i2;
    mcode |= (y&(0x1 << i)) << ++i2;
    mcode |= (x&(0x1 << i)) << ++i2;
  }

  return mcode;
}


static __inline__ __host__ __device__ point_data to_point(morton_code mcode) {
  point_data p = make_point_data(0, 0, 0);

  for (int i = 0; i < KAOLIN_SPC_MAX_LEVELS; i++) {
    p.x |= (mcode&(0x1ll << (3 * i + 2))) >> (2 * i + 2);
    p.y |= (mcode&(0x1ll << (3 * i + 1))) >> (2 * i + 1);
    p.z |= (mcode&(0x1ll << (3 * i + 0))) >> (2 * i + 0);
  }

  return p;
}

static __inline__ __host__ __device__ float3 mul3x3(float3 a, float3x3 m) {
  return make_float3(
    a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0],
    a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1],
    a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2]
  );
}

static __inline__ __host__ __device__ float3 mul3x4(float3 a, float4x4 m) {
  return make_float3(
    a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + m.m[3][0],
    a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + m.m[3][1],
    a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + m.m[3][2]
  );
}


static __inline__ __host__ __device__ float4 mul4x4(float4 a, float4x4 m) {
  return make_float4(
    a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + a.w * m.m[3][0],
    a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + a.w * m.m[3][1],
    a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + a.w * m.m[3][2],
    a.x * m.m[0][3] + a.y * m.m[1][3] + a.z * m.m[2][3] + a.w * m.m[3][3]
  );
}

static __inline__ __host__ __device__  float3 crs3(float3 a, float3 b) {
  return make_float3(
    a.y * b.z - b.y * a.z,
    a.z * b.x - b.z * a.x,
    a.x * b.y - b.x * a.y);
}

static __inline__ __host__ __device__ float3x3 make_float3x3(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22) {
  float3x3 a;
  a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02;
  a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12;
  a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22;
  return a;
}

static __inline__ __host__ __device__ float4x4 make_float4x4(
    float a00, float a01, float a02, float a03,
    float a10, float a11, float a12, float a13,
    float a20, float a21, float a22, float a23,
    float a30, float a31, float a32, float a33) {
  float4x4 a;
  a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02; a.m[0][3] = a03;
  a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12; a.m[1][3] = a13;
  a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22; a.m[2][3] = a23;
  a.m[3][0] = a30; a.m[3][1] = a31; a.m[3][2] = a32; a.m[3][3] = a33;
  return a;
}

static __inline__ __host__ __device__ void  matmul4x4(const float4x4& a, const float4x4& b, float4x4& c)
{
    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    c.m[3][0] = a.m[3][0] * b.m[0][0] + a.m[3][1] * b.m[1][0] + a.m[3][2] * b.m[2][0] + a.m[3][3] * b.m[3][0];
    c.m[3][1] = a.m[3][0] * b.m[0][1] + a.m[3][1] * b.m[1][1] + a.m[3][2] * b.m[2][1] + a.m[3][3] * b.m[3][1];
    c.m[3][2] = a.m[3][0] * b.m[0][2] + a.m[3][1] * b.m[1][2] + a.m[3][2] * b.m[2][2] + a.m[3][3] * b.m[3][2];
    c.m[3][3] = a.m[3][0] * b.m[0][3] + a.m[3][1] * b.m[1][3] + a.m[3][2] * b.m[2][3] + a.m[3][3] * b.m[3][3];
}

static __inline__ __host__ __device__ float4x4 operator* (const float4x4& ma, const float4x4& mb)
{
    float4x4 mc; matmul4x4(ma, mb, mc); return mc;
}

static __inline__ __host__ __device__ float4x4 transpose(const float4x4& a) {
  float4x4 b;
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      b.m[i][j] = a.m[j][i];
  return b;
}

static inline __host__ __device__ void copy(float3x3 &a, float3x3 b) {
  a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1]; a.m[0][2] = b.m[0][2];
  a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1]; a.m[1][2] = b.m[1][2];
  a.m[2][0] = b.m[2][0]; a.m[2][1] = b.m[2][1]; a.m[2][2] = b.m[2][2];
}

// cut&paste from helper_math.h
// DOUBLE
static inline __host__ __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline __host__ __device__ double dot2(double3 a) {
    return dot(a, a);
}

static inline __host__ __device__ double3 cross(double3 a, double3 b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline __host__ __device__ double3 normalize(double3 v) {
    double invLen = rsqrt(dot2(v));
    return make_double3(invLen * v.x, invLen * v.y, invLen * v.z);
}

static inline __host__ __device__ double3 operator-(double3 a, double3 b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// FLOAT
static inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline __host__ __device__ void operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

static inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline __host__ __device__ void operator-=(float3 &a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

static inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

static inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

static inline __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

static inline __host__ __device__ float3 operator/(float3 a, const float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

static inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline __host__ __device__ float dot2(float3 a) {
    return dot(a, a);
}

static inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline __host__ __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot2(v));
    return v * invLen;
}


static inline __host__ __device__ bool equals(float3 a, float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

static __inline__ __host__ __device__  double4 crs4(float3 aa, float3 bb, float3 cc) {

  double3 a = make_double3(aa.x, aa.y, aa.z);
  double3 b = make_double3(bb.x, bb.y, bb.z);
  double3 c = make_double3(cc.x, cc.y, cc.z);
  double4 P;
  P.x = a.z*b.y - a.y*b.z - a.z*c.y + b.z*c.y + a.y*c.z - b.y*c.z;
  P.y = a.x*b.z + a.z*c.x - b.z*c.x - a.x*c.z + b.x*c.z - a.z*b.x;
  P.z = a.y*b.x - a.x*b.y - a.y*c.x + b.y*c.x + a.x*c.y - b.x*c.y;
  P.w = a.y*b.z*c.x + a.z*b.x*c.y - a.x*b.z*c.y - a.y*b.x*c.z + a.x*b.y*c.z -a.z*b.y*c.x;
  return P;
}

static __host__ __device__ __forceinline__ float project_edge(
    const float3& vertex, const float3& edge, const float3& point) {
  const float3 point_vec = point - vertex;
  const float length = dot2(edge);
  return dot(point_vec, edge) / length;
}

static __host__ __device__ __forceinline__ float3 project_plane(
    const float3& vertex, const float3& normal, const float3& point) {
  const float3 unit_normal = normalize(normal);
  const float dist = (point.x - vertex.x) * unit_normal.x + \
                     (point.y - vertex.y) * unit_normal.y + \
                     (point.z - vertex.z) * unit_normal.z;
  return point - (unit_normal * dist);
}

static __host__ __device__ __forceinline__ bool is_not_above(
    const float3& vertex, const float3& edge, const float3& normal, const float3& point) {
  const float3 edge_normal = cross(normal, edge);
  return dot(edge_normal, point - vertex) <= 0;
}

static __host__ __device__ __forceinline__ float3 point_at(
    const float3& vertex, const float3& edge, const float& t) {
  return vertex + (edge * t);
}


static __host__ __device__  float3 triangle_closest_point(
    const float3& v1, const float3& v2, const float3& v3,
    const float3& p) {
  const float3 e12 = v2 - v1;
  const float3 e23 = v3 - v2;
  const float3 e31 = v1 - v3;
  const float3 normal = cross(v1 - v2, e31);
  const float uab = project_edge(v1, e12, p);
  const float uca = project_edge(v3, e31, p);
  if (uca > 1 && uab < 0) {
    return v1;
  } else {
    const float ubc = project_edge(v2, e23, p);
    if (uab > 1 && ubc < 0) {
      return v2;
    } else if (ubc > 1 && uca < 0) {
      return v3;
    } else {
      if (uab <= 1. && uab >= 0. && (is_not_above(v1, e12, normal, p))) {
        return point_at(v1, e12, uab);
      } else if (ubc <= 1. && ubc >= 0. && (is_not_above(v2, e23, normal, p))) {
        return point_at(v2, e23, ubc);
      } else if (uca <= 1. && uca >= 0. && (is_not_above(v3, e31, normal, p))) {
        return point_at(v3, e31, uca);
      } else {
        return project_plane(v1, normal, p);
      }
    }
  }
}

#endif  // WITH_CUDA
#endif  // KAOLIN_SPC_MATH_H_
