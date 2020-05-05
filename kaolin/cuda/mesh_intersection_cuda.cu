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

#include <iostream>

using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>

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
	int sign(float a )
{
	if (a < 0) { return -1;}
	else { return 1; }
}
__device__
 float3 operator* (float3 a,float b )
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
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
	float signed_volume(float3 a, float3 b, float3 c, float3 d )
{
	float3 v = cross(b-a, c-a);
	return  sign(dot(v, d-a));
 
}





__global__ 
void MeshIntersectionKernel(
	int b,
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

					float3 p1 = make_float3(buf_1[k*3+0], buf_1[k*3+1], buf_1[k*3+2] );
					float3 p2 = make_float3(buf_2[k*3+0], buf_2[k*3+1], buf_2[k*3+2] );
					float3 p3 = make_float3(buf_3[k*3+0], buf_3[k*3+1], buf_3[k*3+2] );

					int cond_1 = signed_volume(q1, p1, p2, p3 );
					int cond_2 = signed_volume(q2, p1, p2, p3 );
					if( cond_1 != cond_2 ){
						int cond_3 = signed_volume(q1, q2, p1, p2);
						int cond_4 = signed_volume(q1, q2, p2, p3);
						if (cond_3 == cond_4){ 
							int cond_5 = signed_volume(q1, q2, p3, p1);
							if ( cond_5 == cond_3){ 
								atomicAdd(&result[j], 1./32.);
								
								
							}
						}
					} 	
				}	
			}	
		}
		__syncthreads();
	}
	
}

void MeshIntersectionKernelLauncher(
    const float* points,
    const float* verts_1,
    const float* verts_2,
    const float* verts_3,
    const int b, const int n,
    const int m,
    float* result)
{


	
	MeshIntersectionKernel<<<dim3(32,16,1),512>>>(b, n, points, m, verts_1, verts_2, verts_3, result);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in Intersection updateOutput: %s\n", cudaGetErrorString(err));
}

