// Copyright (c) 2017 Fei Xia

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "emd_cuda.h"


__global__ void approxmatch(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,float * __restrict__ match,float * temp){
	float * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
	float multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=n/m;
	}else{
		multiL=m/n;
		multiR=1;
	}
	const int Block=1024;
	__shared__ float buf[Block*7];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
			match[i*n*m+j]=0;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			remainL[j]=multiL;
		for (int j=threadIdx.x;j<m;j+=blockDim.x)
			remainR[j]=multiR;
		__syncthreads();
		for (int j=7;j>=-2;j--){
			float level=-powf(4.0f,j);
			if (j==-2){
				level=0;
			}
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				float x1=0,y1=0,z1=0, xn1=0,yn1=0,zn1=0;
				if (k<n){
					x1=xyz1[i*n*6+k*6+0];
					y1=xyz1[i*n*6+k*6+1];
					z1=xyz1[i*n*6+k*6+2];
                                        xn1=xyz1[i*n*6+k*6+3];
                                        yn1=xyz1[i*n*6+k*6+4];
                                        zn1=xyz1[i*n*6+k*6+5];
				}
				float suml=1e-9f;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						float x2=xyz2[i*m*6+l0*6+l*6+0];
						float y2=xyz2[i*m*6+l0*6+l*6+1];
						float z2=xyz2[i*m*6+l0*6+l*6+2];
                                                float xn2=xyz2[i*m*6+l0*6+l*6+3];
                                                float yn2=xyz2[i*m*6+l0*6+l*6+4];
                                                float zn2=xyz2[i*m*6+l0*6+l*6+5];
						buf[l*7+0]=x2;
						buf[l*7+1]=y2;
						buf[l*7+2]=z2;
                                                buf[l*7+3]=xn2;
                                                buf[l*7+4]=yn2;
                                                buf[l*7+5]=zn2;
						buf[l*7+6]=remainR[l0+l];
					}
					__syncthreads();
					for (int l=0;l<lend;l++){
						float x2=buf[l*7+0];
						float y2=buf[l*7+1];
						float z2=buf[l*7+2];
                                                float xn2=buf[l*7+3];
                                                float yn2=buf[l*7+4];
                                                float zn2=buf[l*7+5];
						float d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2));
						float w=__expf(d)*buf[l*7+6];
						suml+=w;
					}
					__syncthreads();
				}
				if (k<n)
					ratioL[k]=remainL[k]/suml;
			}

			__syncthreads();
			for (int l0=0;l0<m;l0+=blockDim.x){
				int l=l0+threadIdx.x;
				float x2=0,y2=0,z2=0,xn2=0,yn2=0,zn2=0;
				if (l<m){
					x2=xyz2[i*m*6+l*6+0];
					y2=xyz2[i*m*6+l*6+1];
					z2=xyz2[i*m*6+l*6+2];
                                        xn2=xyz2[i*m*6+l*6+3];
                                        yn2=xyz2[i*m*6+l*6+4];
                                        zn2=xyz2[i*m*6+l*6+5];
				}
				float sumr=0;
				for (int k0=0;k0<n;k0+=Block){
					int kend=min(n,k0+Block)-k0;
					for (int k=threadIdx.x;k<kend;k+=blockDim.x){
						buf[k*7+0]=xyz1[i*n*6+k0*6+k*6+0];
						buf[k*7+1]=xyz1[i*n*6+k0*6+k*6+1];
						buf[k*7+2]=xyz1[i*n*6+k0*6+k*6+2];
                                                buf[k*7+3]=xyz1[i*n*6+k0*6+k*6+3];
                                                buf[k*7+4]=xyz1[i*n*6+k0*6+k*6+4];
                                                buf[k*7+5]=xyz1[i*n*6+k0*6+k*6+5];
						buf[k*7+6]=ratioL[k0+k];
					}
					__syncthreads();
					for (int k=0;k<kend;k++){
						float x1=buf[k*7+0];
						float y1=buf[k*7+1];
						float z1=buf[k*7+2];
                                                float xn1=buf[k*7+3];
                                                float yn1=buf[k*7+4];
                                                float zn1=buf[k*7+5];
						float w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2)))*buf[k*7+6];
						sumr+=w;
					}
					__syncthreads();
				}
				if (l<m){
					sumr*=remainR[l];
					float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
					ratioR[l]=consumption*remainR[l];
					remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
				}
			}

			__syncthreads();
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				float x1=0,y1=0,z1=0,xn1=0,yn1=0,zn1=0;
				if (k<n){
					x1=xyz1[i*n*6+k*6+0];
					y1=xyz1[i*n*6+k*6+1];
					z1=xyz1[i*n*6+k*6+2];
                                        xn1=xyz1[i*n*6+k*6+3];
                                        yn1=xyz1[i*n*6+k*6+4];
                                        zn1=xyz1[i*n*6+k*6+5];
				}
				float suml=0;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						buf[l*7+0]=xyz2[i*m*6+l0*6+l*6+0];
						buf[l*7+1]=xyz2[i*m*6+l0*6+l*6+1];
						buf[l*7+2]=xyz2[i*m*6+l0*6+l*6+2];
                                                buf[l*7+3]=xyz2[i*m*6+l0*6+l*6+3];
                                                buf[l*7+4]=xyz2[i*m*6+l0*6+l*6+4];
                                                buf[l*7+5]=xyz2[i*m*6+l0*6+l*6+5];
						buf[l*7+6]=ratioR[l0+l];
					}
					__syncthreads();
					float rl=ratioL[k];
					if (k<n){
						for (int l=0;l<lend;l++){
							float x2=buf[l*7+0];
							float y2=buf[l*7+1];
							float z2=buf[l*7+2];
                                                        float xn2=buf[l*7+3];
                                                        float yn2=buf[l*7+4];
                                                        float zn2=buf[l*7+5];
							float w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2)))*rl*buf[l*7+6];
							match[i*n*m+(l0+l)*n+k]+=w;
							suml+=w;
						}
					}
					__syncthreads();
				}
				if (k<n)
					remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}
			__syncthreads();
		}
	}
}
int approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match, float * temp){
	approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match,temp);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
            printf("error in matching: %s\n", cudaGetErrorString(err));
                    //THError("aborting");
                    return 0;
    }
return 1;
}
__global__ void matchcost(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ out){
	__shared__ float allsum[512];
	const int Block=1024;
	__shared__ float buf[Block*6];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		float subsum=0;
		for (int k0=0;k0<n;k0+=blockDim.x){
			int k=k0+threadIdx.x;
			float x1=0,y1=0,z1=0, xn1=0,yn1=0,zn1=0;
			if (k<n){
				x1=xyz1[i*n*6+k*6+0];
				y1=xyz1[i*n*6+k*6+1];
				z1=xyz1[i*n*6+k*6+2];
                                xn1=xyz1[i*n*6+k*6+3];
                                yn1=xyz1[i*n*6+k*6+4];
                                zn1=xyz1[i*n*6+k*6+5];
			}
			for (int l0=0;l0<m;l0+=Block){
				int lend=min(m,l0+Block)-l0;
				for (int l=threadIdx.x;l<lend*6;l+=blockDim.x)
					buf[l]=xyz2[i*m*6+l0*6+l];
				__syncthreads();
				if (k<n){
					for (int l=0;l<lend;l++){
						float x2=buf[l*6+0];
						float y2=buf[l*6+1];
						float z2=buf[l*6+2];
                                                float xn2=buf[l*6+3];
                                                float yn2=buf[l*6+4];
                                                float zn2=buf[l*6+5];
						float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2));
						subsum+=d*match[i*n*m+(l0+l)*n+k];
					}
				}
				__syncthreads();
			}
		}
		allsum[threadIdx.x]=subsum;
		for (int j=1;j<blockDim.x;j<<=1){
			__syncthreads();
			if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
				allsum[threadIdx.x]+=allsum[threadIdx.x+j];
			}
		}
		if (threadIdx.x==0)
			out[i]=allsum[0];
		__syncthreads();
	}
}
int matchcostLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out){
	matchcost<<<32,512>>>(b,n,m,xyz1,xyz2,match,out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
                printf("error in emd updateOutput: %s\n", cudaGetErrorString(err));
                        //THError("aborting");
                        return 0;
    }
    return 1;


}
__global__ void matchcostgrad2(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad2){
	__shared__ float sum_grad[256*6];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			float x2=xyz2[(i*m+k)*6+0];
			float y2=xyz2[(i*m+k)*6+1];
			float z2=xyz2[(i*m+k)*6+2];
                        float xn2=xyz2[(i*m+k)*6+3];
                        float yn2=xyz2[(i*m+k)*6+4];
                        float zn2=xyz2[(i*m+k)*6+5];
			float subsumx=0,subsumy=0,subsumz=0,subsumxn=0,subsumyn=0,subsumzn=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				float x1=x2-xyz1[(i*n+j)*6+0];
				float y1=y2-xyz1[(i*n+j)*6+1];
				float z1=z2-xyz1[(i*n+j)*6+2];
                                float xn1=xn2-xyz1[(i*n+j)*6+3];
                                float yn1=yn2-xyz1[(i*n+j)*6+4];
                                float zn1=zn2-xyz1[(i*n+j)*6+5];
				float d=match[i*n*m+k*n+j]*rsqrtf(fmaxf(x1*x1+y1*y1+z1*z1+xn1*xn1+yn1*yn1+zn1*zn1,1e-20f));
				subsumx+=x1*d;
				subsumy+=y1*d;
				subsumz+=z1*d;
                                subsumxn+=xn1*d;
                                subsumyn+=yn1*d;
                                subsumzn+=zn1*d;
			}
			sum_grad[threadIdx.x*6+0]=subsumx;
			sum_grad[threadIdx.x*6+1]=subsumy;
			sum_grad[threadIdx.x*6+2]=subsumz;
                        sum_grad[threadIdx.x*6+3]=subsumxn;
                        sum_grad[threadIdx.x*6+4]=subsumyn;
                        sum_grad[threadIdx.x*6+5]=subsumzn;
			for (int j=1;j<blockDim.x;j<<=1){
				__syncthreads();
				int j1=threadIdx.x;
				int j2=threadIdx.x+j;
				if ((j1&j)==0 && j2<blockDim.x){
					sum_grad[j1*6+0]+=sum_grad[j2*6+0];
					sum_grad[j1*6+1]+=sum_grad[j2*6+1];
					sum_grad[j1*6+2]+=sum_grad[j2*6+2];
                                        sum_grad[j1*6+3]+=sum_grad[j2*6+3];
                                        sum_grad[j1*6+4]+=sum_grad[j2*6+4];
                                        sum_grad[j1*6+5]+=sum_grad[j2*6+5];
				}
			}
			if (threadIdx.x==0){
				grad2[(i*m+k)*6+0]=sum_grad[0];
				grad2[(i*m+k)*6+1]=sum_grad[1];
				grad2[(i*m+k)*6+2]=sum_grad[2];
                                grad2[(i*m+k)*6+3]=sum_grad[3];
                                grad2[(i*m+k)*6+4]=sum_grad[4];
                                grad2[(i*m+k)*6+5]=sum_grad[5];
			}
			__syncthreads();
		}
	}
}
__global__ void matchcostgrad1(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad1){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int l=threadIdx.x;l<n;l+=blockDim.x){
			float x1=xyz1[i*n*6+l*6+0];
			float y1=xyz1[i*n*6+l*6+1];
			float z1=xyz1[i*n*6+l*6+2];
                        float xn1=xyz1[i*n*6+l*6+3];
                        float yn1=xyz1[i*n*6+l*6+4];
                        float zn1=xyz1[i*n*6+l*6+5];
			float dx=0,dy=0,dz=0,dxn=0,dyn=0,dzn=0;
			for (int k=0;k<m;k++){
				float x2=xyz2[i*m*6+k*6+0];
				float y2=xyz2[i*m*6+k*6+1];
				float z2=xyz2[i*m*6+k*6+2];
                                float xn2=xyz2[i*m*6+k*6+3];
                                float yn2=xyz2[i*m*6+k*6+4];
                                float zn2=xyz2[i*m*6+k*6+5];
				float d=match[i*n*m+k*n+l]*rsqrtf(fmaxf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2),1e-20f));
				dx+=(x1-x2)*d;
				dy+=(y1-y2)*d;
				dz+=(z1-z2)*d;
                                dxn+=(xn1-xn2)*d;
                                dyn+=(yn1-yn2)*d;
                                dzn+=(zn1-zn2)*d;
			}
			grad1[i*n*6+l*6+0]=dx;
			grad1[i*n*6+l*6+1]=dy;
			grad1[i*n*6+l*6+2]=dz;
                        grad1[i*n*6+l*6+3]=dxn;
                        grad1[i*n*6+l*6+4]=dyn;
                        grad1[i*n*6+l*6+5]=dzn;
		}
	}
}
int matchcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2){
	matchcostgrad1<<<32,512>>>(b,n,m,xyz1,xyz2,match,grad1);
	matchcostgrad2<<<dim3(32,32),256>>>(b,n,m,xyz1,xyz2,match,grad2);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
            printf("error in emd backward: %s\n", cudaGetErrorString(err));
                    //THError("aborting");
                    return 0;
    }
    return 1;

}

