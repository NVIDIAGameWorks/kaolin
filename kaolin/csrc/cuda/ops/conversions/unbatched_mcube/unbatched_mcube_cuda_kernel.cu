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

#include <stdio.h>
#include <string.h>

#include "tables.h"
#include "helper_math.h"

#include <ATen/ATen.h>

// #include <cub/cub.cuh> 

// textures containing look-up tables
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numUniqueVertsTex;
texture<uint, 1, cudaReadModeElementType> numTrianglesTex;
texture<uint, 1, cudaReadModeElementType> numPartialVertsTex;
texture<uint, 1, cudaReadModeElementType> vertsOrderTex;

void allocateTextures(at::Tensor d_triTable, at::Tensor d_numUniqueVertsTable, 
                      at::Tensor d_numTrianglesTable, at::Tensor d_numPartialVertsTable,
                      at::Tensor d_vertsOrderTable)
{
  // TODO: rename allocateTextures
  // TODO: check if texture is already binded.
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaMemcpy((void *) d_triTable.data_ptr<int>(), (void *)triTable, 256*16*sizeof(int), cudaMemcpyHostToDevice);
  cudaBindTexture(0, triTex, d_triTable.data_ptr<int>(), channelDesc);

  cudaMemcpy((void *) d_numUniqueVertsTable.data_ptr<int>(), (void *)numUniqueVertsTable, 256*sizeof(int), cudaMemcpyHostToDevice);
  cudaBindTexture(0, numUniqueVertsTex, d_numUniqueVertsTable.data_ptr<int>(), channelDesc);

  cudaMemcpy((void *) d_numTrianglesTable.data_ptr<int>(), (void *)numTrianglesTable, 256*sizeof(int), cudaMemcpyHostToDevice);
  cudaBindTexture(0, numTrianglesTex, d_numTrianglesTable.data_ptr<int>(), channelDesc);

  cudaMemcpy((void *) d_numPartialVertsTable.data_ptr<int>(), (void *)numPartialVertsTable, 256*sizeof(int), cudaMemcpyHostToDevice);
  cudaBindTexture(0, numPartialVertsTex, d_numPartialVertsTable.data_ptr<int>(), channelDesc);

  cudaMemcpy((void *) d_vertsOrderTable.data_ptr<int>(), (void *)vertsOrderTable, 256*3*sizeof(int), cudaMemcpyHostToDevice);
  cudaBindTexture(0, vertsOrderTex, d_vertsOrderTable.data_ptr<int>(), channelDesc);
}

// sample volume data set at a point
__device__
float sampleVolume(float* data, int3 p, int3 gridSize)
{
  p.x = min(p.x, gridSize.x - 1);
  p.y = min(p.y, gridSize.y - 1);
  p.z = min(p.z, gridSize.z - 1);
  int i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;

  return data[i];
}

// compute position in 3d grid from 1d index
__device__
int3 calcGridPos(int i, int3 gridSize)
{
  int3 gridPos;
  gridPos.x = i % gridSize.x;
  gridPos.y = (i / gridSize.x) % gridSize.y;
  gridPos.z = (i / gridSize.x / gridSize.y) % gridSize.z;
  return gridPos;
}

__global__ void
classifyVoxel(int *voxelOccupied, int *voxelTriangles, int *voxelPartialVerts,
              int *voxelVertsOrder,
              float* volume, int3 gridSize, int numVoxels,
              float3 voxelSize, float isoValue)
{
  int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  int i = __mul24(blockId, blockDim.x) + threadIdx.x;

  int3 gridPos = calcGridPos(i, gridSize);

  // read field values at neighbouring grid vertices// find the target voxel index which is responsible to generate the vertex
  float field[8];
  field[0] = sampleVolume(volume, gridPos, gridSize);
  field[1] = sampleVolume(volume, gridPos + make_int3(1, 0, 0), gridSize);
  field[2] = sampleVolume(volume, gridPos + make_int3(1, 1, 0), gridSize);
  field[3] = sampleVolume(volume, gridPos + make_int3(0, 1, 0), gridSize);
  field[4] = sampleVolume(volume, gridPos + make_int3(0, 0, 1), gridSize);
  field[5] = sampleVolume(volume, gridPos + make_int3(1, 0, 1), gridSize);
  field[6] = sampleVolume(volume, gridPos + make_int3(1, 1, 1), gridSize);
  field[7] = sampleVolume(volume, gridPos + make_int3(0, 1, 1), gridSize);

  // calculate flag indicating if each vertex is inside or outside isosurface
  int cubeindex;
  cubeindex =  int(field[0] < isoValue);
  cubeindex += int(field[1] < isoValue)*2;
  cubeindex += int(field[2] < isoValue)*4;
  cubeindex += int(field[3] < isoValue)*8;
  cubeindex += int(field[4] < isoValue)*16;
  cubeindex += int(field[5] < isoValue)*32;
  cubeindex += int(field[6] < isoValue)*64;
  cubeindex += int(field[7] < isoValue)*128;

  // read number of vertices from texture for half cube
  int numVerts = tex1Dfetch(numUniqueVertsTex, cubeindex);
  int numPartialVerts = tex1Dfetch(numPartialVertsTex, cubeindex);
  int numTriangles = tex1Dfetch(numTrianglesTex, cubeindex);

  int vertsOrder1 = tex1Dfetch(vertsOrderTex, cubeindex*3);
  int vertsOrder2 = tex1Dfetch(vertsOrderTex, cubeindex*3 + 1);
  int vertsOrder3 = tex1Dfetch(vertsOrderTex, cubeindex*3 + 2);

  if (i < numVoxels)
  {
    voxelPartialVerts[i] = numPartialVerts;
    voxelOccupied[i] = (numVerts > 0);
    voxelTriangles[i] = numTriangles;
    
    voxelVertsOrder[i*3] = vertsOrder1;
    voxelVertsOrder[i*3 + 1] = vertsOrder2;
    voxelVertsOrder[i*3 + 2] = vertsOrder3;
  }
}

void launch_classifyVoxel(at::Tensor voxelOccupied, at::Tensor voxelTriangles, at::Tensor voxelPartialVerts,
                          at::Tensor voxelVertsOrder,
                          int3 gridSize, int numVoxels, at::Tensor voxelgrid,
                          float3 voxelSize, float isoValue)
{   
  int threads;
  dim3 grid(1, 1, 1);

  // For smaller voxelgrid
  if (numVoxels < 128) {
    threads = numVoxels;
  } else {
    threads = 128;
    grid.x = numVoxels / threads;
  }

  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535)
  {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }
    // calculate number of vertices need per voxel
    classifyVoxel<<<grid, threads>>>(voxelOccupied.data_ptr<int>(), 
                                     voxelTriangles.data_ptr<int>(), voxelPartialVerts.data_ptr<int>(),
                                     voxelVertsOrder.data_ptr<int>(),
                                     voxelgrid.data_ptr<float>(), gridSize,
                                     numVoxels, voxelSize, isoValue);
}

// compact voxel array
__global__ void
compactVoxels(int *compactedVoxelArray, int *voxelOccupied, int *voxelOccupiedScan, int numVoxels)
{
  int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  int i = __mul24(blockId, blockDim.x) + threadIdx.x;

  if (voxelOccupied[i] && (i < numVoxels))
  {
    compactedVoxelArray[voxelOccupiedScan[i]] = i;
  }
}

void launch_compactVoxels(at::Tensor compactedVoxelArray, at::Tensor voxelOccupied, at::Tensor voxelOccupiedScan, int numVoxels)
{
  int threads;
  dim3 grid(1, 1, 1);

  // For smaller voxelgrid
  if (numVoxels < 128) {
    threads = numVoxels;
  } else {
    threads = 128;
    grid.x = numVoxels / threads;
  }

  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535)
  {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }

  compactVoxels<<<grid, threads>>>(compactedVoxelArray.data_ptr<int>(), voxelOccupied.data_ptr<int>(),
                                   voxelOccupiedScan.data_ptr<int>(), numVoxels);
}

// compute interpolated vertex along an edge
__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
  float t = (isolevel - f0) / (f1 - f0);
  return lerp(p0, p1, t);
}

// find the target voxel index which is responsible to generate the vertex
__device__
int find_target_voxel(int3 gridSize, int face_idx, int current_voxel)
{
  int target_voxel_idx;
  // x-axis increase -> to the right
  // y-axis increase -> to the top
  // z-axis increase -> to the back
  switch(face_idx) {
    case 0:  // looking for vertices in bot-front voxel
      target_voxel_idx = current_voxel - gridSize.x - gridSize.x*gridSize.y;
      break;
    
    case 1:  // looking for vertices in right-front voxel
      target_voxel_idx = current_voxel + 1 - gridSize.x*gridSize.y;
      break;

    case 2:  // looking for vertices in front voxel
      target_voxel_idx = current_voxel - gridSize.x*gridSize.y;
      break;

    case 3: // looking for vertices in front voxel
      target_voxel_idx = current_voxel - gridSize.x*gridSize.y;
      break;

    case 4: // looking for vertices in bot voxel
      target_voxel_idx = current_voxel - gridSize.x;
      break;

    case 5: // looking for vertices in right voxel
      target_voxel_idx = current_voxel + 1;
      break;
    
    case 6:  // looking for vertices in current voxel
      target_voxel_idx = current_voxel;
      break;

    case 7:  // looking for vertices in current voxel
      target_voxel_idx = current_voxel;
      break;

    case 8: // looking for vertices in bot voxel
      target_voxel_idx = current_voxel - gridSize.x;
      break;

    case 9: // looking for vertices in right-bot voxel
      target_voxel_idx = current_voxel + 1 - gridSize.x;
      break;

    case 10: // looking for vertices in right voxel
      target_voxel_idx = current_voxel + 1;
      break;
    
    case 11: // looking for vertices in current voxel
      target_voxel_idx = current_voxel;
      break;

    default:
      target_voxel_idx = current_voxel;
      break;
  }
  return target_voxel_idx;
}

// find the offset, given the vertex is on what edge
__device__
int find_offset(int face_idx, int voxel_index, int* voxelVertsOrder)
{
  int offset;
  int corresponding_edge;  // corresponding edge number in current voxel_index

  switch(face_idx) {
    case 0:  // looking for vertices in bot-front voxel
      corresponding_edge = 6; // corresponds to edge 6
      break;
    
    case 1:  // looking for vertices in right-front voxel
      corresponding_edge = 7; // corresponds to edge 7
      break;

    case 2:  // looking for vertices in front voxel
      corresponding_edge = 6; // corresponds to edge 6
      break;

    case 3: // looking for vertices in front voxel
      corresponding_edge = 7; // corresponds to edge 7
      break;

    case 4: // looking for vertices in bot voxel
      corresponding_edge = 6; // corresponds to edge 6
      break;

    case 5: // looking for vertices in right voxel
      corresponding_edge = 7; // corresponds to edge 7
      break;
    
    case 6:  // looking for vertices in current voxel
      corresponding_edge = 6;
      break;

    case 7:  // looking for vertices in current voxel
      corresponding_edge = 7;
      break;

    case 8: // looking for vertices in bot voxel
      corresponding_edge = 11; // corresponds to edge 11
      break;

    case 9: // looking for vertices in right-bot voxel
      corresponding_edge = 11; // corresponds to edge 11
      break;

    case 10: // looking for vertices in right voxel
      corresponding_edge = 11; // corresponds to edge 11
      break;
    
    case 11: // looking for vertices in current voxel
      corresponding_edge = 11;
      break;

    default:
      corresponding_edge = face_idx;
      break;
  }

  int verts_order_1 = voxelVertsOrder[voxel_index * 3];
  int verts_order_2 = voxelVertsOrder[voxel_index * 3 + 1];
  int verts_order_3 = voxelVertsOrder[voxel_index * 3 + 2];

  if (verts_order_1 == 255 && verts_order_2 == 255 && verts_order_3 == 255) {
    return 0;
  }

  if (corresponding_edge == verts_order_1) {
    offset = 0;
  } else if (corresponding_edge == verts_order_2) {
    offset = 1;
  } else if (corresponding_edge == verts_order_3) {
    offset = 2;
  }

  return offset;
}


// version that calculates flat surface normal for each triangle
__global__ void
generateTriangles2(float *pos, int *faces, int *compactedVoxelArray,
                   int *numTrianglesScanned, int *numPartialVertsScanned, int *numPartialVerts,
                   int *voxelVertsOrder,
                   float* volume, int3 gridSize,
                   float3 voxelSize, float isoValue, int activeVoxels, int maxVerts)
{
  int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  int grid_index = __mul24(blockId, blockDim.x) + threadIdx.x;

  if (grid_index > activeVoxels - 1)
  {
    grid_index = activeVoxels - 1;
  }

  int voxel = compactedVoxelArray[grid_index];

  // compute position in 3d grid
  int3 gridPos = calcGridPos(voxel, gridSize);

  float3 p;

  p.x = gridPos.x;
  p.y = gridPos.y;
  p.z = gridPos.z;

  // calculate unnormalized cell vertex positions
  float3 v[8];
  v[0] = p;
  v[1] = p + make_float3(1, 0, 0);
  v[2] = p + make_float3(1, 1, 0);
  v[3] = p + make_float3(0, 1, 0);
  v[4] = p + make_float3(0, 0, 1);
  v[5] = p + make_float3(1, 0, 1);
  v[6] = p + make_float3(1, 1, 1);
  v[7] = p + make_float3(0, 1, 1);

  float field[8];
  field[0] = sampleVolume(volume, gridPos, gridSize);
  field[1] = sampleVolume(volume, gridPos + make_int3(1, 0, 0), gridSize);
  field[2] = sampleVolume(volume, gridPos + make_int3(1, 1, 0), gridSize);
  field[3] = sampleVolume(volume, gridPos + make_int3(0, 1, 0), gridSize);
  field[4] = sampleVolume(volume, gridPos + make_int3(0, 0, 1), gridSize);
  field[5] = sampleVolume(volume, gridPos + make_int3(1, 0, 1), gridSize);
  field[6] = sampleVolume(volume, gridPos + make_int3(1, 1, 1), gridSize);
  field[7] = sampleVolume(volume, gridPos + make_int3(0, 1, 1), gridSize);

  // recalculate flag
  int cubeindex;
  cubeindex =  int(field[0] < isoValue);
  cubeindex += int(field[1] < isoValue)*2;
  cubeindex += int(field[2] < isoValue)*4;
  cubeindex += int(field[3] < isoValue)*8;
  cubeindex += int(field[4] < isoValue)*16;
  cubeindex += int(field[5] < isoValue)*32;
  cubeindex += int(field[6] < isoValue)*64;
  cubeindex += int(field[7] < isoValue)*128;

  // find the vertices where the surface intersects the cube    
  // use shared memory to avoid using local
  __shared__ float3 vertlist[12*NTHREADS];

  vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
  vertlist[NTHREADS+threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
  vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
  vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
  vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
  vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
  vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
  vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
  vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
  vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
  vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
  vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

  __syncthreads();

  int added_vertx_count = 0;
  for (int i=0; i<3; i++) { // maximum 3 newly added vertices for a voxel

    float3 *v[1];

    uint edge = tex1Dfetch(vertsOrderTex, (cubeindex*3) + i);

    if (edge == 255) {
      break;
    }

    // Only add the top-left-back vertices of the cube to the vertices' list
    // Meaning only vertices on the edge 6, 7, 11
    int index = numPartialVertsScanned[voxel] + added_vertx_count;
    added_vertx_count++;

    v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

    if (index < (maxVerts - 3)) {
      pos[index * 3] = (v[0]) -> x;
      pos[index * 3 + 1] = (v[0]) -> y;
      pos[index * 3 + 2] = (v[0]) -> z;
    }
  }

  // Add triangles
  for (int j=0; j<16; j+=3) {
    uint face_idx1 = tex1Dfetch(triTex, cubeindex*16 + j);

    if (face_idx1 == 255) {
      break;
    }

    uint face_idx2 = tex1Dfetch(triTex, cubeindex*16 + j + 1);
    uint face_idx3 = tex1Dfetch(triTex, cubeindex*16 + j + 2);

    int num_prev_verts;
    int num_prev_triangles;

    int target_voxel_idx1 = find_target_voxel(gridSize, face_idx1, voxel);
    int target_voxel_idx2 = find_target_voxel(gridSize, face_idx2, voxel);
    int target_voxel_idx3 = find_target_voxel(gridSize, face_idx3, voxel);

    int offset1 = find_offset(face_idx1, target_voxel_idx1, voxelVertsOrder);
    int offset2 = find_offset(face_idx2, target_voxel_idx2, voxelVertsOrder); 
    int offset3 = find_offset(face_idx3, target_voxel_idx3, voxelVertsOrder);

    // handle first vertex
    num_prev_verts = numPartialVertsScanned[target_voxel_idx1];
    num_prev_triangles = numTrianglesScanned[voxel];

    faces[num_prev_triangles * 3 + j] = num_prev_verts + offset1;

    // handle second vertex
    num_prev_verts = numPartialVertsScanned[target_voxel_idx2];
    num_prev_triangles = numTrianglesScanned[voxel];

    faces[num_prev_triangles * 3 + j + 1] = num_prev_verts + offset2;

    // handle last vertex
    num_prev_verts = numPartialVertsScanned[target_voxel_idx3];
    num_prev_triangles = numTrianglesScanned[voxel];

    faces[num_prev_triangles * 3 + j + 2] = num_prev_verts + offset3;
  }
}

void launch_generateTriangles2(at::Tensor pos, at::Tensor faces, at::Tensor compactedVoxelArray,
                              at::Tensor numTrianglesScanned, at::Tensor numPartialVertsScanned, at::Tensor numPartialVerts,
                              at::Tensor voxelVertsOrder,
                              int3 gridSize, at::Tensor voxelgrid,
                              float3 voxelSize, float isoValue, int activeVoxels, int maxVerts)
{
  dim3 grid2((int) ceil(activeVoxels/ (float) NTHREADS), 1, 1);

  while (grid2.x > 65535) {
    grid2.x/=2;
    grid2.y*=2;
  }

  generateTriangles2<<<grid2, NTHREADS>>>(pos.data_ptr<float>(), faces.data_ptr<int>(),
                                          compactedVoxelArray.data_ptr<int>(), numTrianglesScanned.data_ptr<int>(),
                                          numPartialVertsScanned.data_ptr<int>(), numPartialVerts.data_ptr<int>(),
                                          voxelVertsOrder.data_ptr<int>(),
                                          voxelgrid.data_ptr<float>(), gridSize,
                                          voxelSize, isoValue, activeVoxels,
                                          maxVerts);
}
