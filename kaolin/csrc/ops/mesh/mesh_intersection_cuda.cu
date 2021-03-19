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

using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../utils.h"

namespace kaolin {

__device__
    float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z ;
}


__device__
    float3 cross(float3 a, float3 b)
{
  return make_float3(a.y*b.z - a.z*b.y, 
                    a.z*b.x - a.x*b.z, 
                    a.x*b.y - a.y*b.x
                    );
    
}


__device__
    float3 operator- (float3 a,float3 b )
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}


__device__
    bool bbox_check(float3 a, float3 b, float3 c, float3 d)
{
    float y_min = min(b.y, min(c.y, d.y));
    float y_max = max(b.y, max(c.y, d.y));
    float z_min = min(b.z, min(c.z, d.z));
    float z_max = max(b.z, max(c.z, d.z));
    if (a.y < y_min || y_max < a.y || a.z < z_min || z_max < a.z) {return false;}
    return true;
}

__device__
    bool signed_volume(float3 a, float3 b, float3 c, float3 d)
{
    float3 v = cross(b-a, c-a);
    return dot(v, d-a) > 0;
}

__device__
    bool signed_area(float3 a, float3 b, float3 c)
{
    return (c.z - b.z)*(a.y - b.y) + (-c.y + b.y)*(a.z - b.z) > 0;
}



__global__ 
void UnbatchedMeshIntersectionKernel(
    int n,
    const float* points,
    int m,
    const float* verts_1,
    const float* verts_2,
    const float* verts_3,
    float* result)
{       
    const int batch=1024;
    
    __shared__ float buf_1[batch*3];
    __shared__ float buf_2[batch*3];
    __shared__ float buf_3[batch*3];
    
    for (int k2=0;k2<m;k2+=batch){
        int end_k=min(m,k2+batch)-k2;
        for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
            buf_1[j]=verts_1[k2*3+j];
            buf_2[j]=verts_2[k2*3+j];
            buf_3[j]=verts_3[k2*3+j];
        }
        __syncthreads();
        for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){ // for points in a batch 
            
            float3 q1 = make_float3(points[ j *3+0], points[ j *3+1], points[ j *3+2]);
            float3 q2 = make_float3(points[ j *3+0] + 10., points[ j *3+1], points[ j *3+2]);
            for (int k=0;k<end_k;k++){
                {       

                    float3 p1 = make_float3(buf_1[k*3+0], buf_1[k*3+1], buf_1[k*3+2]);
                    float3 p2 = make_float3(buf_2[k*3+0], buf_2[k*3+1], buf_2[k*3+2]);
                    float3 p3 = make_float3(buf_3[k*3+0], buf_3[k*3+1], buf_3[k*3+2]);

                    if (!bbox_check(q1, p1, p2, p3)){continue;}
                    bool cond_1 = signed_volume(q1, p1, p2, p3 );
                    bool cond_2 = signed_volume(q2, p1, p2, p3 );
                    if( cond_1 != cond_2 ){
                        bool cond_3 = signed_area(q1, p1, p2);
                        bool cond_4 = signed_area(q1, p2, p3);                
                        if (cond_3 == cond_4){ 
                            bool cond_5 = signed_area(q1, p3, p1);
                            if ( cond_5 == cond_3){ 
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

void UnbatchedMeshIntersectionKernelLauncher(
    const float* points,
    const float* verts_1,
    const float* verts_2,
    const float* verts_3,
    const int n,
    const int m,
    float* result) {
    UnbatchedMeshIntersectionKernel<<<dim3(1,512,1),512>>>(n, points, m, verts_1, verts_2, verts_3, result);

    CUDA_CHECK(cudaGetLastError());
}

}  // namespace kaolin
