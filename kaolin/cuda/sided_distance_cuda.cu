// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_atomic_functions.h"


__global__ void SidedDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){

    const int batch=512;
    __shared__ float buf[batch*3];
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int k2=0;k2<m;k2+=batch){
            int end_k=min(m,k2+batch)-k2;
            for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
                buf[j]=xyz2[(i*m+k2)*3+j];
            }
            __syncthreads();
            for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
                float x1=xyz[(i*n+j)*3+0];
                float y1=xyz[(i*n+j)*3+1];
                float z1=xyz[(i*n+j)*3+2];
                int best_i=0;
                float best=0;
                int end_ka=end_k-(end_k&3);
                if (end_ka==batch){
                    for (int k=0;k<batch;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (k==0 || d<best){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }else{
                    for (int k=0;k<end_ka;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (k==0 || d<best){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }
                for (int k=end_ka;k<end_k;k++){
                    float x2=buf[k*3+0]-x1;
                    float y2=buf[k*3+1]-y1;
                    float z2=buf[k*3+2]-z1;
                    float d=x2*x2+y2*y2+z2*z2;
                    if (k==0 || d<best){
                        best=d;
                        best_i=k+k2;
                    }
                }
                if (k2==0 || result[(i*n+j)]>best){
                    result[(i*n+j)]=best;
                    result_i[(i*n+j)]=best_i;
                }
            }
            __syncthreads();
        }
    }
}

void SidedDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i)
{
    SidedDistanceKernel<<<dim3(32,16,1),512>>>( b, n, xyz, m, xyz2, result, result_i);
}


