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

#include <ATen/ATen.h>

#define CUB_STDERR
#include <cub/device/device_scan.cuh>

#include "../../spc_math.h"
#include "../../spc_utils.cuh"

namespace kaolin {

using namespace std;


__global__ void PointToMorton(
  const uint    Psize,
  morton_code*  Mdata,
  point_data*   Pdata)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize)
    Mdata[tidx] = to_morton(Pdata[tidx]);
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


at::Tensor morton_to_octree_cuda_impl(at::Tensor mortons, uint32_t level)
{
    uint32_t psize = mortons.size(0);

    morton_code* d_morton = reinterpret_cast<morton_code*>(mortons.data_ptr<int64_t>());

    at::Tensor morton2 = at::empty({psize}, mortons.options().dtype(at::kLong));
    morton_code* d_morton2 = reinterpret_cast<morton_code*>(morton2.data_ptr<int64_t>());

    morton_code* mroot[2] = {d_morton, d_morton2};

    at::Tensor info = at::empty({psize}, mortons.options().dtype(at::kInt));
    uint32_t*  d_info = reinterpret_cast<uint32_t*>(info.data_ptr<int>());
    at::Tensor psum = at::empty({psize}, mortons.options().dtype(at::kInt));
    uint32_t*  d_psum = reinterpret_cast<uint32_t*>(psum.data_ptr<int>());

    // set up memory for DeviceScan calls
    void* temp_storage_ptr = NULL;
    uint64_t temp_storage_bytes = get_cub_storage_bytes(temp_storage_ptr, d_info, d_psum, psize);
    at::Tensor temp_storage = at::empty({(int64_t)temp_storage_bytes}, mortons.options().dtype(at::kByte));
    temp_storage_ptr = (void*)temp_storage.data_ptr<uint8_t>();

    // memory for octree levels (don't know final octree size a priori)
    at::Tensor* ostorage = new at::Tensor[level];
    uchar** olevel = new uchar*[level];
    int*  h_pyramid = new int[level]; // remove +1, might be bug

    uint buf = 0;
    uint curr, prev = psize;
    for (uint i=level; i>0; i--) { // Start from deepest layer
        // Mark boundaries of morton code octants
        d_CommonParent << <(prev + 1023) / 1024, 1024 >> >(prev, mroot[buf], d_info);
        // Count number of allocated nodes in layer above
        cub::DeviceScan::InclusiveSum(temp_storage_ptr, temp_storage_bytes, d_info, d_psum, prev);
        cudaMemcpy(&curr, d_psum+prev-1, sizeof(uint), cudaMemcpyDeviceToHost);
        h_pyramid[i-1] = curr;

        // allocate tensor GPU memory for octree level
        ostorage[i-1] = at::empty({curr}, mortons.options().dtype(at::kByte));
        // keep pointer to tensor to keep garbage collector at bay
        olevel[i-1] = ostorage[i-1].data_ptr<uchar>();

        // Populate octree & next level of morton codes
        d_CompactifyNodes << <(prev + 1023) / 1024, 1024 >> >(prev, d_info, d_psum, mroot[buf], mroot[(buf+1)%2], olevel[i-1]);

        prev = curr;
        buf = (buf+1)%2;
    }
 
    // sum pyramid to get octree size
    uint osize = 0;
    for (uint l=0; l<level; l++)
        osize += h_pyramid[l];
        
    // allocate octree tensor
    at::Tensor octree = at::empty({osize}, mortons.options().dtype(at::kByte));
    uchar* d_octree = octree.data_ptr<uchar>();

    // copy level data into contiguous memory
    for (uint l=0; l<level; l++)
    {
      cudaMemcpy(d_octree, olevel[l], h_pyramid[l], cudaMemcpyDeviceToDevice);
      d_octree += h_pyramid[l];
    }

    delete[] h_pyramid;
    delete[] olevel;
    delete[] ostorage;

    return octree;
}


at::Tensor points_to_octree_cuda_impl(at::Tensor points, uint32_t level) {

    uint32_t psize = points.size(0);
    point_data* d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

    at::Tensor morton = at::empty({psize}, points.options().dtype(at::kLong));
    morton_code* d_morton = reinterpret_cast<morton_code*>(morton.data_ptr<int64_t>());

    // convert points to morton codes
    PointToMorton<<<(psize+1023)/1024, 1024>>>(psize, d_morton, d_points);

    return morton_to_octree_cuda_impl(morton, level);
}


}  // namespace kaolin

