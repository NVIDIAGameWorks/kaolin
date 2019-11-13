// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
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
    float dot2( float3 v ) { 
    return dot(v,v); 
}

__device__
float clamp(float x, float a, float b)
{
  return max(a, min(b, x));
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
	int signage(float3 a, float3 b, float3 c)
{

  return sign(dot(cross(a,b), c ));
}

__device__
	float edge_distance(float3 a, float3 b)
{
	
  return dot2(a*clamp(dot(a,b)/dot2(a),0.0,1.0)-b);
}

__device__
	float plane_distance(float3 a, float3 b)
{

  return dot(a,b)*dot(a,b)/dot2(a);
}











__global__ 
void TriangleDistanceKernel(
	int b,
	int n,
	const float* points,
	int m,
	const float* verts_1,
	const float* verts_2,
	const float* verts_3,
	float* result,
	int* result_i,
	int* result_t)
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
			
			float3 p = make_float3(points[ j *3+0], points[ j *3+1], points[ j *3+2]);
			int best_i=0;
			int best_t=0;
			float best=0;
			int end_ka=end_k-(end_k&3);
			for (int k=0;k<end_k;k++){
						
				float3 v1 = make_float3(buf_1[k*3+0], buf_1[k*3+1], buf_1[k*3+2] );
				float3 v2 = make_float3(buf_2[k*3+0], buf_2[k*3+1], buf_2[k*3+2] );
				float3 v3 = make_float3(buf_3[k*3+0], buf_3[k*3+1], buf_3[k*3+2] );

				float3 v21 = v2 - v1; 
			    float3 v32 = v3 - v2; 
			    float3 v13 = v1 - v3; 

			    float3 p1 = p - v1;
			    float3 p2 = p - v2;
			    float3 p3 = p - v3;

			    float3 nor = cross( v21, v13 );

			    float sign_cond = signage(v21,nor,p1) + signage(v32,nor,p2) + signage( v13, nor, p3);

			    float dist = 100; 
			    int type = 0;

			    if (sign_cond < 2.0) { 
			    	float dist1 = edge_distance( v21, p1 );
			    	float dist2 = edge_distance( v32, p2 );
			    	float dist3 = edge_distance( v13, p3 );

			    	if (dist1 <= dist2 && dist1 <= dist3){
			    		dist = dist1;
			    		type = 0;
			    	}
			    	else if (dist2 <= dist1 && dist2 <= dist3){
			    		dist = dist2;
			    		type = 1;
			    	}
			    	else {
			    		dist = dist3;
			    		type = 2;
			    	}
			    }
			    else{
			    	dist = plane_distance( nor, p1);
			    	type = 3;
			    }


				if (k==0 || dist<best){
					best=dist;
					best_i=k+k2;
					best_t= type;
				}
					
			}
			
			if (k2==0 || result[ j ]>best){
				result[ j ]=best;
				result_i[ j ]=best_i;
				result_t[ j ]=best_t;
			}
		}
		__syncthreads();
	}
	
}

void TriangleDistanceKernelLauncher(
    const float* points,
    const float* verts_1,
    const float* verts_2,
    const float* verts_3,
    const int b, const int n,
    const int m,
    float* result,
    int* result_i,
    int* result_t)
{



	TriangleDistanceKernel<<<dim3(32,16,1),512>>>(b, n, points, m, verts_1, verts_2, verts_3, result, result_i, result_t);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in Triangle distance updateOutput: %s\n", cudaGetErrorString(err));
}

