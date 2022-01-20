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

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "../../../spc_math.h"

namespace kaolin {

using namespace std;


uint64_t GetStorageBytes(void* d_temp_storageA, morton_code* d_M0, morton_code* d_M1, uint max_total_points)
{
    uint64_t    temp_storage_bytesA = 0;
    CubDebugExit(
        cub::DeviceRadixSort::SortKeys(d_temp_storageA, temp_storage_bytesA,
                                       d_M0, d_M1, max_total_points)
    );
    return temp_storage_bytesA;
}


__global__ void
d_Transform2VNC(uint num, float3* pnts, float4x4 mM)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
    pnts[tidx] = mul3x4(pnts[tidx], mM);
}


__global__ void
d_RemoveDuplicates(uint num, morton_code* mcode, uint* info)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    if (tidx == 0)
      info[tidx] = 1;
    else
      info[tidx] = mcode[tidx - 1] == mcode[tidx] ? 0 : 1;
  }
}


__global__ void
d_Compactify(uint num, morton_code* mIn, morton_code* mOut, uint* info, uint* prefix_sum)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx])
    mOut[prefix_sum[tidx]] = mIn[tidx];
}


__global__ void
d_ProcessTriangles(uint npnts, float3* P, uint ntris, tri_index* T, float3* dl0, float3* dl1, float3* dl2, float3* dF, uchar* daxis, ushort* dW, ushort2* dpmin, uint* info)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < ntris)
  {
    tri_index t = T[tidx];

    float3 h0 = P[t.x];
    float3 h1 = P[t.y];
    float3 h2 = P[t.z];

    // quantize vertex coords
    float3 p0 = make_float3((int)(h0.x+0.5), (int)(h0.y+0.5), (int)(h0.z+0.5));
    float3 p1 = make_float3((int)(h1.x+0.5), (int)(h1.y+0.5), (int)(h1.z+0.5));
    float3 p2 = make_float3((int)(h2.x+0.5), (int)(h2.y+0.5), (int)(h2.z+0.5));

    float3 l0;
    float3 l1;
    float3 l2;
    float3 F;
    float3 q0, q1, q2;
    uint axis;
    // compute spanning plane
    float4 pln = crs4(p0, p1, p2);

    if (pln.x == 0.0f && pln.y == 0.0f && pln.z == 0.0f && pln.w == 0.0f)
    {
      // IMPLEMENTATION FOR 1D/2D CORNER CASEs
      float3 pmin = make_float3(10000000.0f, 10000000.0f, 10000000.0f);
      float3 pmax = make_float3(-10000000.0f, -10000000.0f, -10000000.0f);

      if (p0.x > pmax.x) pmax.x = p0.x;
      if (p1.x > pmax.x) pmax.x = p1.x;
      if (p2.x > pmax.x) pmax.x = p2.x;

      if (p0.y > pmax.y) pmax.y = p0.y;
      if (p1.y > pmax.y) pmax.y = p1.y;
      if (p2.y > pmax.y) pmax.y = p2.y;

      if (p0.z > pmax.z) pmax.z = p0.z;
      if (p1.z > pmax.z) pmax.z = p1.z;
      if (p2.z > pmax.z) pmax.z = p2.z;

      if (p0.x < pmin.x) pmin.x = p0.x;
      if (p1.x < pmin.x) pmin.x = p1.x;
      if (p2.x < pmin.x) pmin.x = p2.x;

      if (p0.y < pmin.y) pmin.y = p0.y;
      if (p1.y < pmin.y) pmin.y = p1.y;
      if (p2.y < pmin.y) pmin.y = p2.y;

      if (p0.z < pmin.z) pmin.z = p0.z;
      if (p1.z < pmin.z) pmin.z = p1.z;
      if (p2.z < pmin.z) pmin.z = p2.z;

      if (pmin.x == pmax.y && pmin.y == pmax.y && pmin.z == pmax.z)
      {
        // IMPLEMENTATION FOR 1D CORNER CASE
        q0 = q1 = q2 = pmin;
        l0 = l1 = l2 = -1.0f*pmin;
        F = make_float3(0.0f, 0.0f, pmin.z);
        axis = 2;
      }
      else
      {
        // IMPLEMENTATION FOR 2D CORNER CASE
        float3 diff = pmax - pmin;
        if (diff.x < diff.y)
          if (diff.x < diff.z)
            axis = 0; //x
          else
            axis = 2; //z
        else
          if (diff.y < diff.z)
            axis = 1; //y
          else
            axis = 2; //z

        switch (axis)
        {
        case 0:
          q0 = make_float3(pmin.y, pmin.z, 1.0f);
          q1 = make_float3(pmax.y, pmax.z, 1.0f);
          q2 = q1;

          if (diff.y != 0.0)
            F = make_float3(diff.x/diff.y, 0.0f, (pmin.x*pmax.y-pmin.y*pmax.x)/diff.y);
          else
            F = make_float3(0.0f, diff.x/diff.z, (pmin.x*pmax.z-pmin.z*pmax.x)/diff.z);

          break;
        case 1:
          q0 = make_float3(pmin.z, pmin.x, 1.0f);
          q1 = make_float3(pmax.z, pmax.x, 1.0f);
          q2 = q1;

          if (diff.z != 0.0)
            F = make_float3(diff.y/diff.z, 0.0f, (pmin.y*pmax.z-pmin.z*pmax.y)/diff.z);
          else
            F = make_float3(0.0f, diff.y/diff.x, (pmin.y*pmax.x-pmin.x*pmax.y)/diff.x);

          break;
        case 2:
          q0 = make_float3(pmin.x, pmin.y, 1.0f);
          q1 = make_float3(pmax.x, pmax.y, 1.0f);
          q2 = q1;

          if (diff.x != 0.0)
            F = make_float3(diff.z/diff.x, 0.0f, (pmin.z*pmax.x-pmin.x*pmax.z)/diff.x);
          else
            F = make_float3(0.0f, diff.z/diff.y, (pmin.z*pmax.y-pmin.y*pmax.z)/diff.y);

          break;
        }
        // find bounding lines
        l1 = -1.0f*crs3(q0, q1);
        l0 = -1.0f*crs3(q1, q0);
        l2 = l1;
      }
    } else {
      // find coordinate with largest normal component
      if (fabs(pln.x) > fabs(pln.y))
        if (fabs(pln.x) > fabs(pln.z))
          axis = 0; //x
        else
          axis = 2; //z
      else
        if (fabs(pln.y) > fabs(pln.z))
          axis = 1; //y
        else
          axis = 2; //z

      // project to 2d plane with largest normal component
      float sign = 0.0f;

      switch (axis)
      {
      case 0:
        q0 = make_float3(p0.y, p0.z, 1.0f);
        q1 = make_float3(p1.y, p1.z, 1.0f);
        q2 = make_float3(p2.y, p2.z, 1.0f);

        sign = pln.x > 0.0f ? 1.0f : -1.0f;
        F = make_float3(pln.y, pln.z, pln.w);
        F *= -1.0f/pln.x;
        break;
      case 1:
        q0 = make_float3(p0.z, p0.x, 1.0f);
        q1 = make_float3(p1.z, p1.x, 1.0f);
        q2 = make_float3(p2.z, p2.x, 1.0f);

        sign = pln.y > 0.0f ? 1.0f : -1.0f;
        F = make_float3(pln.z, pln.x, pln.w);
        F *= -1.0f/pln.y;
        break;
      case 2:
        q0 = make_float3(p0.x, p0.y, 1.0f);
        q1 = make_float3(p1.x, p1.y, 1.0f);
        q2 = make_float3(p2.x, p2.y, 1.0f);

        sign = pln.z > 0.0f ? 1.0f : -1.0f;
        F = make_float3(pln.x, pln.y, pln.w);
        F *= -1.0f/pln.z;
        break;
      }

      // find bounding lines
      l0 = sign*crs3(q1, q2);
      l1 = sign*crs3(q2, q0);
      l2 = sign*crs3(q0, q1);
    }

    // enlarge lines for conservative rasterization
    l0.z += (l0.x>0.0f?-0.5f:0.5f)*l0.x + (l0.y>0.0f?-0.5f:0.5f)*l0.y;
    l1.z += (l1.x>0.0f?-0.5f:0.5f)*l1.x + (l1.y>0.0f?-0.5f:0.5f)*l1.y;
    l2.z += (l2.x>0.0f?-0.5f:0.5f)*l2.x + (l2.y>0.0f?-0.5f:0.5f)*l2.y;

    // find bound rectangle
    ushort2 pmin = make_ushort2(0xffff,0xffff);
    ushort2 pmax = make_ushort2(0, 0);

    if (q0.x < pmin.x) pmin.x = q0.x;
    if (q0.y < pmin.y) pmin.y = q0.y;
    if (q1.x < pmin.x) pmin.x = q1.x;
    if (q1.y < pmin.y) pmin.y = q1.y;
    if (q2.x < pmin.x) pmin.x = q2.x;
    if (q2.y < pmin.y) pmin.y = q2.y;

    if (q0.x > pmax.x) pmax.x = q0.x;
    if (q0.y > pmax.y) pmax.y = q0.y;
    if (q1.x > pmax.x) pmax.x = q1.x;
    if (q1.y > pmax.y) pmax.y = q1.y;
    if (q2.x > pmax.x) pmax.x = q2.x;
    if (q2.y > pmax.y) pmax.y = q2.y;

    ushort W = pmax.x - pmin.x + 1;
    ushort H = pmax.y - pmin.y + 1;

    dpmin[tidx] = pmin;

    info[tidx] = W*H;
    dW[tidx] = W;

    daxis[tidx] = axis;
    dl0[tidx] = l0;
    dl1[tidx] = l1;
    dl2[tidx] = l2;
    dF[tidx] = F;
  }
}


__global__ void
d_ProcessVoxels(uint nvxls, float3* l0, float3* l1, float3* l2, float3* F, uchar* axis, ushort* W, ushort2* pmin, uint* info, uint* psum, uint* PrefixSum, uint* Info, morton_code* M)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < nvxls)
  {
    uint t = PrefixSum[tidx];
    uint pid = tidx - (psum[t] - info[t]);

    float x = pmin[t].x + (pid % W[t]);
    float y = pmin[t].y + (pid / W[t]);
    float3 p = make_float3(x, y, 1.0f);

    bool t0 = dot(p, l0[t]) < 0.0f;
    bool t1 = dot(p, l1[t]) < 0.0f;
    bool t2 = dot(p, l2[t]) < 0.0f;

    short z = (short)(dot(p, F[t]) + 0.5f);

    point_data v;
    if (t0 & t1 & t2)
    {
      switch (axis[t])
      {
      case 0:
        v = make_point_data(z, x, y);
        break;
      case 1:
        v = make_point_data(y, z, x);
        break;
      case 2:
        v = make_point_data(x, y, z);
        break;
      }

      M[tidx] = to_morton(v);
      Info[tidx] = 1;
    } else {
      Info[tidx] = 0;
    }
  }
}


__global__ void
d_prepTriTable(uint num, uint* prefix_sum, uint* info)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
    atomicAdd(info + prefix_sum[tidx], 1);
}


__global__ void d_CommonParent(
  const uint Psize,
  const morton_code *d_Mdata,
  uint *d_Info)
{
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize)
  {
    if (tidx == 0)
      d_Info[tidx] = 1;
    else
      d_Info[tidx] = (d_Mdata[tidx - 1] >> 3) == (d_Mdata[tidx] >> 3) ? 0 : 1;
  }
}


__global__ void d_CompactifyNodes(
  const uint Psize,
  const uint *d_Info,
  const uint *d_PrefixSum,
  morton_code *d_Min,
  morton_code *d_Mout,
  uchar *O)
{
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize)
  {
    if (d_Info[tidx] != 0)
    {
      uint IdxOut = d_PrefixSum[tidx]-1;
      d_Mout[IdxOut] = d_Min[tidx] >> 3;

      uint code = 0;
      do
      {
        uint child_idx = static_cast<uint>(d_Min[tidx] & 0x7);
        code |= 0x1 << child_idx;
        tidx++;
      } while (tidx != Psize && d_Info[tidx] != 1);

      O[IdxOut] = code;
    }
  }
}


__global__ void MortonToPointa(
  const uint     Psize,
  morton_code*  Mdata,
  point_data*    Pdata)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize)
    Pdata[tidx] = to_point(Mdata[tidx]);
}

__global__ void PointToMorton(
  const uint    Psize,
  morton_code*  Mdata,
  point_data*   Pdata)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize)
    Mdata[tidx] = to_morton(Pdata[tidx]);
}

// Uses the morton buffer to construct an octree. It is the user's responsibility to allocate
// space for these zero-init buffers, and for the morton buffer, to allocate the buffer from the back 
// with the occupied positions.
uint ConstructOctree(morton_code* d_morton_buffer, uint* d_info, uint* d_psum, 
    void* d_temp_storage, uint64_t temp_storage_bytes, uchar* d_octree, int* h_pyramid, uint psize, uint level) {
    

    h_pyramid[level] = psize;
    uint curr, prev = psize;
    morton_code* mroot = d_morton_buffer+KAOLIN_SPC_MAX_POINTS-psize;
    uchar* oroot = d_octree+KAOLIN_SPC_MAX_OCTREE;
    
    // Start from deepest layer
    for (uint i=level; i>0; i--) {
        // Mark boundaries of morton code octants
        d_CommonParent << <(prev + 1023) / 1024, 1024 >> >(prev, mroot, d_info);
        // Count number of allocated nodes in layer above
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_info, d_psum, prev);
        cudaMemcpy(&curr, d_psum+prev-1, sizeof(uint), cudaMemcpyDeviceToHost);
        h_pyramid[i-1] = curr;

        // Move pointer back
        oroot -= curr;
        mroot -= curr;

        // Populate octree & next level of morton codes
        d_CompactifyNodes << <(prev + 1023) / 1024, 1024 >> >(prev, d_info, d_psum, mroot+curr, mroot, oroot);
        prev = curr;
    }

    // Populate pyramid
    int* h_pyramidsum = h_pyramid + level + 2;
    h_pyramidsum[0] = 0;
    for (uint i=0; i<=level; i++)
    {
        h_pyramidsum[i+1] = h_pyramidsum[i] + h_pyramid[i];
    }
    uint osize = h_pyramidsum[level];

    return osize;
}

uint PointToOctree(point_data* d_points, morton_code* d_morton, uint* d_info, uint* d_psum, 
    void* d_temp_storage, uint64_t temp_storage_bytes, uchar* d_octree, int* h_pyramid, 
    uint psize, uint level) {
    
    // Populate from the back
    morton_code* mroot = d_morton+KAOLIN_SPC_MAX_POINTS-psize;
    PointToMorton<<<(psize+1023)/1024, 1024>>>(psize, mroot, d_points);

    return ConstructOctree(d_morton, d_info, d_psum, d_temp_storage, temp_storage_bytes, 
        d_octree, h_pyramid, psize, level);
}

uint VoxelizeGPU(uint npnts, float3* d_Pnts, uint ntris, tri_index* d_Tris, uint Level,
    point_data* d_P, morton_code* d_M0, morton_code* d_M1, 
    uint* d_Info, uint* d_PrefixSum, uint* d_info, uint* d_psum,
    float3* d_l0, float3* d_l1, float3* d_l2, float3* d_F,
    uchar* d_axis, ushort* d_W, ushort2* d_pmin,
    void* d_temp_storageA, uint64_t temp_storage_bytesA, uchar* d_Odata, int* h_Pyramid) {

    // Transform vertices to [0, 2^l]
    float g = (0x1<<Level) - 1.0f;
    float4x4 mM = make_float4x4(g, 0.0f, 0.0f, 0.0f,
                     0.0f, g, 0.0f, 0.0f,
                     0.0f, 0.0f, g, 0.0f,
                     0.0f, 0.0f, 0.0f, 1.0f);
    d_Transform2VNC<<<(npnts + 63) / 64, 64>>>(npnts, d_Pnts, mM);

    // Rasterize triangles onto the 3D voxel grid
    d_ProcessTriangles<<<(ntris + 63) / 64, 64>>>(npnts, d_Pnts, ntris, d_Tris,
                                                  d_l0, d_l1, d_l2, d_F, d_axis,
                                                  d_W, d_pmin, d_info);
    
    // Info contains triangle ID -> # rasterized voxels. 
    // Count total # rasterized voxels and flush the buffer.
    uint cnt;
    cub::DeviceScan::InclusiveSum(d_temp_storageA, temp_storage_bytesA, d_info, d_psum, ntris);
    cudaMemcpy(&cnt, d_psum + ntris - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemset(d_Info, 0, cnt+1);

    // Mark boundaries of packed voxels. Some boundaries will have duplciate triangles.
    d_prepTriTable << <(ntris + 63) / 64, 64 >> > (ntris, d_psum, d_Info);
    
    // Create voxel->triangle ID correspondences.
    cub::DeviceScan::InclusiveSum(d_temp_storageA, temp_storage_bytesA, d_Info, d_PrefixSum, cnt+1);

    // Fill morton buffer with voxels
    d_ProcessVoxels << <(cnt + 63) / 64, 64 >> > (cnt, d_l0, d_l1, d_l2, d_F, d_axis, d_W, d_pmin, d_info, d_psum, d_PrefixSum, d_Info, d_M0);
    uint mcnt;

    // Sort mortons, find unique, and fill the buffer (d_M1)
    // d_M1 = d_M0[d_Info]
    cub::DeviceScan::ExclusiveSum(d_temp_storageA, temp_storage_bytesA, d_Info, d_PrefixSum, cnt+1);
    cudaMemcpy(&mcnt, d_PrefixSum + cnt, sizeof(uint), cudaMemcpyDeviceToHost);
    d_Compactify << <(cnt + 63) / 64, 64 >> > (cnt, d_M0, d_M1, d_Info, d_PrefixSum);

    // d_M0 = d_M1.sort()
    CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storageA, temp_storage_bytesA, d_M1, d_M0, mcnt));

    // Mark boundaries of unique  
    d_RemoveDuplicates << <(mcnt + 63) / 64, 64 >> > (mcnt, d_M0, d_Info);

    // d_M1[-psize:] = d_M0[d_Info]
    uint psize;
    cub::DeviceScan::ExclusiveSum(d_temp_storageA, temp_storage_bytesA, d_Info, d_PrefixSum, mcnt+1);
    cudaMemcpy(&psize, d_PrefixSum + mcnt, sizeof(uint), cudaMemcpyDeviceToHost);
    d_Compactify << <(mcnt + 63) / 64, 64 >> > (mcnt, d_M0, d_M1+KAOLIN_SPC_MAX_POINTS-psize, d_Info, d_PrefixSum);
    // printf("cnt= %d, mcnt= %d, psize= %d\n", cnt, mcnt, psize);
    
    uint osize = ConstructOctree(d_M1, d_Info, d_PrefixSum, d_temp_storageA, temp_storage_bytesA, 
                                 d_Odata, h_Pyramid, psize, Level);

    // MortonToPointa << <(totalPoints + 1023) / 1024, 1024 >> >(totalPoints, M, d_P);
    CubDebugExit(cudaGetLastError());

    return osize;
}

}  // namespace kaolin

