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

#include <torch/extension.h>

#ifdef WITH_CUDA
#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "helper_math.h"

#include <ATen/ATen.h>

using namespace torch::indexing;

// CUDA declarations
void launch_classifyVoxel(at::Tensor voxelOccupied, at::Tensor voxelTriangles, at::Tensor voxelPartialVerts,
                          at::Tensor voxelVertsOrder,
                          int3 gridSize, int numVoxels, at::Tensor voxelgrid,
                          float3 voxelSize, float isoValue);

void launch_compactVoxels(at::Tensor compactedVoxelArray, at::Tensor voxelOccupied,
                          at::Tensor voxelOccupiedScan, int numVoxels);

void launch_generateTriangles2(at::Tensor pos, at::Tensor faces, at::Tensor compactedVoxelArray,
                               at::Tensor numTrianglesScanned, at::Tensor numPartialVertsScanned, at::Tensor numPartialVerts,
                               at::Tensor voxelVertsOrder,
                               int3 gridSize, at::Tensor voxelgrid,
                               float3 voxelSize, float isoValue, int activeVoxels, int maxVerts);

void allocateTextures(at::Tensor d_triTable, at::Tensor d_numUniqueVertsTable, at::Tensor d_numTrianglesTable, 
                      at::Tensor d_numPartialVertsTable, at::Tensor d_vertsOrderTable);

void CubScanWrapper(at::Tensor output, at::Tensor input, int numElements);

// torch::Tensor used to store tables
torch::Tensor d_triTable;
torch::Tensor d_numUniqueVertsTable;
torch::Tensor d_numTrianglesTable;
torch::Tensor d_numPartialVertsTable;
torch::Tensor d_vertsOrderTable;

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void
computeIsosurface(int3 gridSize, int3 gridSizeLog2, float isoValue,
                  int *activeVoxels, int *totalVerts, int *totalTriangles, int *totalPartialVerts,
                  int numVoxels, float3 voxelSize, int maxVerts, int maxFaces,
                  torch::Tensor voxelgrid, torch::Tensor d_pos, torch::Tensor d_faces,
                  torch::Tensor d_voxelPartialVerts,
                  torch::Tensor d_voxelTriangles,
                  torch::Tensor d_voxelOccupied,
                  torch::Tensor d_compVoxelArray,
                  torch::Tensor d_voxelVertsOrder)
{
  // calculate number of vertices and triangles need per voxel
  launch_classifyVoxel(d_voxelOccupied,
                       d_voxelTriangles, d_voxelPartialVerts,
                       d_voxelVertsOrder,
                       gridSize, numVoxels, voxelgrid,
                       voxelSize, isoValue);
  
  torch::Tensor d_voxelOccupiedScan = torch::zeros({numVoxels}, voxelgrid.options().dtype(torch::kInt32));
  CubScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

  // read back values to calculate total number of non-empty voxels
  // since we are using an exclusive scan, the total is the last value of
  // the scan result plus the last value in the input array
  {
    int lastElement, lastScanElement;
    cudaMemcpy((void *) &lastElement,
               (void *)(d_voxelOccupied.data_ptr<int>() + numVoxels - 1),
               sizeof(int), cudaMemcpyDeviceToHost);
  
    cudaMemcpy((void *) &lastScanElement,
               (void *)(d_voxelOccupiedScan.data_ptr<int>() + numVoxels - 1),
               sizeof(int), cudaMemcpyDeviceToHost);
  
    *activeVoxels = lastElement + lastScanElement;
  }

  if (activeVoxels==0)
  {
    // return if there are no full voxels
    *totalVerts = 0;
    return;
  }

  // compact voxel index array
  launch_compactVoxels(d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
  
  //scan voxel triangle count array
  torch::Tensor d_voxelTrianglesScan = torch::zeros({numVoxels}, voxelgrid.options().dtype(torch::kInt32));
  CubScanWrapper(d_voxelTrianglesScan, d_voxelTriangles, numVoxels);
  
  //scan partial vertex count array
  torch::Tensor d_voxelPartialVertsScan = torch::zeros({numVoxels}, voxelgrid.options().dtype(torch::kInt32));
  CubScanWrapper(d_voxelPartialVertsScan, d_voxelPartialVerts, numVoxels);

  
  // readback total number of triangles
  {
    int lastElement, lastScanElement;
    cudaMemcpy((void *) &lastElement,
               (void *)(d_voxelTriangles.data_ptr<int>() + numVoxels-1),
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *) &lastScanElement,
               (void *)(d_voxelTrianglesScan.data_ptr<int>() + numVoxels-1),
               sizeof(int), cudaMemcpyDeviceToHost);
    *totalTriangles = lastElement + lastScanElement;
  }

  // readback total number of partial verts
  {
    int lastElement, lastScanElement;
    cudaMemcpy((void *) &lastElement,
               (void *)(d_voxelPartialVerts.data_ptr<int>() + numVoxels-1),
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *) &lastScanElement,
               (void *)(d_voxelPartialVertsScan.data_ptr<int>() + numVoxels-1),
               sizeof(int), cudaMemcpyDeviceToHost);
    *totalPartialVerts = lastElement + lastScanElement;
  }

  launch_generateTriangles2(d_pos,
                            d_faces,
                            d_compVoxelArray,
                            d_voxelTrianglesScan,
                            d_voxelPartialVertsScan,
                            d_voxelPartialVerts,
                            d_voxelVertsOrder,
                            gridSize, voxelgrid,
                            voxelSize, isoValue, *activeVoxels,
                            maxVerts);
}

std::vector<at::Tensor> unbatched_mcube_forward(const at::Tensor voxelgrid, float iso_value) {

  int3 gridSizeLog2;
  int3 gridSize;  // height, width, depth of voxelgrid

  float3 voxelSize;
  int numVoxels    = 0;  // numbel of total voxels of the input voxelgrid
  int maxVerts     = 0;  // maximum number of vertices of the output mesh could have
  int maxFaces     = 0;  // maximum number of faces of the output mesh could have
  int activeVoxels = 0;  // number of total voxels that have vertices
  int totalVerts   = 0;  // number of actual generated vertices of the output mesh
  int totalTriangles = 0;  // number of actual generated triangles(faces) of the output mesh
  int totalPartialVerts = 0;  // number of vertices generated by all voxel, but only count on three edges.

  float isoValue = 0.5f;
  isoValue = iso_value;

  int i = voxelgrid.size(0);
  int j = voxelgrid.size(1);
  int k = voxelgrid.size(2);

  gridSizeLog2.x = (int) log2(i);
  gridSizeLog2.y = (int) log2(j);
  gridSizeLog2.z = (int) log2(k);

  gridSize = make_int3(i, j, k);

  numVoxels = gridSize.x*gridSize.y*gridSize.z;
  voxelSize = make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
  maxVerts = gridSize.x*gridSize.y*100;
  maxFaces = numVoxels * 6;

  // initialize tensors
  auto int_options = voxelgrid.options().dtype(torch::kInt32);

  torch::Tensor d_pos = torch::zeros({maxVerts, 3}, voxelgrid.options().dtype(torch::kFloat32)); // tensor to store output vertices

  torch::Tensor d_faces = torch::zeros({maxFaces, 3}, int_options); // tensor to store output faces

  torch::Tensor d_voxelPartialVerts = torch::zeros({numVoxels}, int_options); // tensor to measure how many vertices a voxel will generate, only count on three edges.
  torch::Tensor d_voxelTriangles = torch::zeros({numVoxels}, int_options); // tensor to measure how many trianlges a voxel will generate
  torch::Tensor d_voxelOccupied = torch::zeros({numVoxels}, int_options); // binary tensor to indicate whether the voxel will generate any vertices or not
  torch::Tensor d_compVoxelArray = torch::zeros({numVoxels}, int_options); // compact representation of d_voxelOccupiedScan

  torch::Tensor d_voxelVertsOrder = torch::zeros({numVoxels, 3}, int_options); // tensor to store the order of added verts for each voxel

  // initialize static pointers
  if (!d_triTable.defined()) {
    d_triTable = torch::zeros({256, 16}, int_options);
    d_numUniqueVertsTable = torch::zeros({256}, int_options);
    d_numTrianglesTable = torch::zeros({256}, int_options);
    d_numPartialVertsTable = torch::zeros({256}, int_options);
    d_vertsOrderTable = torch::zeros({256, 3}, int_options);

    // allocate table textures after we initialize everything.
    allocateTextures(d_triTable, d_numUniqueVertsTable, d_numTrianglesTable, d_numPartialVertsTable, d_vertsOrderTable);
  }

  computeIsosurface(gridSize, gridSizeLog2, isoValue,
                    &activeVoxels, &totalVerts, &totalTriangles, &totalPartialVerts,
                    numVoxels, voxelSize, maxVerts, maxFaces,
                    voxelgrid, d_pos, d_faces,
                    d_voxelPartialVerts,
                    d_voxelTriangles,
                    d_voxelOccupied, d_compVoxelArray, d_voxelVertsOrder);

  std::vector<at::Tensor> result;

  result.push_back(d_pos.index({Slice(None, totalPartialVerts)}));
  result.push_back(d_faces.index({Slice(None, totalTriangles)}));

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if ( err != cudaSuccess )
  {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  return result;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  m.def("forward_cuda", &unbatched_mcube_forward, "Unbatched Marching Cube forward");
#endif
}
