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

using namespace std;

#define KAOLIN_SPC_MAX_LEVELS           15
#define KAOLIN_SPC_MAX_OCTREE    (0x1<<25)
#define KAOLIN_SPC_MAX_POINTS    (0x1<<27)


typedef unsigned char       uchar;
typedef unsigned short      ushort;
typedef unsigned int        uint;

typedef uint64_t            morton_code;
typedef short3              point_data;
typedef long3               tri_index;

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

static __inline__ __host__ __device__  float4 crs4(float3 a, float3 b, float3 c) {
  return make_float4(
    a.z*b.y - a.y*b.z - a.z*c.y + b.z*c.y + a.y*c.z - b.y*c.z,
    -(a.z*b.x) + a.x*b.z + a.z*c.x - b.z*c.x - a.x*c.z + b.x*c.z,
    a.y*b.x - a.x*b.y - a.y*c.x + b.y*c.x + a.x*c.y - b.x*c.y,
    -(a.z*b.y*c.x) + a.y*b.z*c.x + a.z*b.x*c.y - a.x*b.z*c.y - a.y*b.x*c.z + a.x*b.y*c.z
  );
}

static __inline__ __host__ __device__  float3 crs3(float3 a, float3 b) {
  return make_float3(
    a.y * b.z - b.y * a.z,
    a.z * b.x - b.z * a.x,
    a.x * b.y - b.x * a.y);
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

// cut&pawst from helper_math.h
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(float3 &a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(float3 v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

#endif  // WITH_CUDA
#endif  // KAOLIN_SPC_MATH_H_
