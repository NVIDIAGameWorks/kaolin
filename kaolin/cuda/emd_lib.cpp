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

#include <TH/TH.h>


void nnsearch(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int * idx){
    for (int i=0;i<b;i++){
        for (int j=0;j<n;j++){
            float x1=xyz1[(i*n+j)*3+0];
            float y1=xyz1[(i*n+j)*3+1];
            float z1=xyz1[(i*n+j)*3+2];
            double best=0;
            int besti=0;
            for (int k=0;k<m;k++){
                float x2=xyz2[(i*m+k)*3+0]-x1;
                float y2=xyz2[(i*m+k)*3+1]-y1;
                float z2=xyz2[(i*m+k)*3+2]-z1;
                double d=x2*x2+y2*y2+z2*z2;
                if (k==0 || d<best){
                    best=d;
                    besti=k;
                }
            }
            dist[i*n+j]=best;
            idx[i*n+j]=besti;
        }
    }
}

int emd_forward(THFloatTensor *xyz1, THFloatTensor *xyz2,
        THFloatTensor *match, THFloatTensor *cost) {
    int batchsize = xyz1->size[0];
    int n = xyz1->size[1];
    int m = xyz2->size[1];

    //printf("in c: %d %d %d\n", batchsize, n, m);

    float *xyz1_data = THFloatTensor_data(xyz1);
    float *xyz2_data = THFloatTensor_data(xyz2);
    float *match_data = THFloatTensor_data(match);
    float *cost_data = THFloatTensor_data(cost);

    approxmatch_cpu(batchsize, n, m, xyz1_data, xyz2_data, match_data);
    matchcost_cpu(batchsize, n, m, xyz1_data, xyz2_data, match_data, cost_data);

    return 1;
}


int emd_backward(THFloatTensor *xyz1, THFloatTensor *xyz2,
        THFloatTensor *gradxyz1, THFloatTensor *gradxyz2,
        THFloatTensor * match) {

    int b = xyz1->size[0];
    int n = xyz1->size[1];
    int m = xyz2->size[1];

    //printf("%d %d %d\n", batchsize, n, m);

    float *xyz1_data = THFloatTensor_data(xyz1);
    float *xyz2_data = THFloatTensor_data(xyz2);
    float *gradxyz1_data = THFloatTensor_data(gradxyz1);
    float *gradxyz2_data = THFloatTensor_data(gradxyz2);
    float *match_data = THFloatTensor_data(match);


    matchcostgrad_cpu(b, n, m, xyz1_data, xyz2_data, match_data, gradxyz1_data, gradxyz2_data);


    return 1;
}


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void approxmatch_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,float * match){
	for (int i=0;i<b;i++){

        int factorl=MAX(n,m)/n;
		int factorr=MAX(n,m)/m;
		//std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
        double saturatedl[n];

        for (int j = 0; j < n; j++) saturatedl[j] = (double)factorl;

        double saturatedr[m];
        for (int j=0; j < m; j++) saturatedr[j] = (double)factorr;
		//std::vector<double> weight(n*m);
        double * weight = (double *)malloc(sizeof(double) *n * m);
		for (int j=0;j<n*m;j++)
			match[j]=0;

        for (int j=8;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<n;k++){
				double x1=xyz1[k*6+0];
				double y1=xyz1[k*6+1];
				double z1=xyz1[k*6+2];
                double xn1=xyz1[k*6+3];
                double yn1=xyz1[k*6+4];
                double zn1=xyz1[k*6+5];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*6+0];
					double y2=xyz2[l*6+1];
					double z2=xyz2[l*6+2];
                    double xn2=xyz2[l*6+3];
                    double yn2=xyz2[l*6+4];
                    double zn2=xyz2[l*6+5];
					//weight[k*m+l] = 0; //
                    double w = expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2)))*saturatedr[l];
                    weight[k * m + l] = w;
				}
			}

			//std::vector<double> ss(m,1e-9);
            double ss[m];
            for (int j = 0; j < m; j++) ss[j] = 1e-9;
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				for (int l=0;l<m;l++){
					weight[k*m+l]=weight[k*m+l]/s*saturatedl[k];
				}
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=MIN(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			//std::vector<double> ss2(m,0);
            double ss2[m];
            for (int j= 0; j < m; j++) ss2[j] = 0;
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					weight[k*m+l]*=ss[l];
					s+=weight[k*m+l];
					ss2[l]+=weight[k*m+l];
				}
				saturatedl[k]=MAX(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<m;l++){
				saturatedr[l]=MAX(saturatedr[l]-ss2[l],0.0);
			}

        }
		xyz1+=n*6;
		xyz2+=m*6;
		match+=n*m;
        free(weight);
	}
}
void matchcost_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,
        float * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			for (int k=0;k<m;k++){
				float x1=xyz1[j*6+0];
				float y1=xyz1[j*6+1];
				float z1=xyz1[j*6+2];
                                float xn1=xyz1[j*6+3];
                                float yn1=xyz1[j*6+4];
                                float zn1=xyz1[j*6+5];
				float x2=xyz2[k*6+0];
				float y2=xyz2[k*6+1];
				float z2=xyz2[k*6+2];
                                float xn2=xyz2[k*6+3];
                                float yn2=xyz2[k*6+4];
                                float zn2=xyz2[k*6+5];
				float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2))*match[j*m+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*6;
		xyz2+=m*6;
		match+=n*m;
		cost+=1;
	}
}
void matchcostgrad_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,
        const float * match,float * grad1,float * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<6*n;j++)
			grad1[j]=0;
		for (int j=0;j<m;j++){
			float sx=0,sy=0,sz=0,sxn=0,syn=0,szn=0;
			for (int k=0;k<n;k++){
				float x2=xyz2[j*6+0];
				float y2=xyz2[j*6+1];
				float z2=xyz2[j*6+2];
                                float xn2=xyz2[j*6+3];
                                float yn2=xyz2[j*6+4];
                                float zn2=xyz2[j*6+5];
				float x1=xyz1[k*6+0];
				float y1=xyz1[k*6+1];
				float z1=xyz1[k*6+2];
                                float xn1=xyz1[k*6+3];
                                float yn1=xyz1[k*6+4];
                                float zn1=xyz1[k*6+5];
				float d=MAX(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)+(xn1-xn2)*(xn1-xn2)+(yn1-yn2)*(yn1-yn2)+(zn1-zn2)*(zn1-zn2)),1e-20f);
				float dx=match[k*m+j]*((x2-x1)/d);
				float dy=match[k*m+j]*((y2-y1)/d);
				float dz=match[k*m+j]*((z2-z1)/d);
                                float dxn=match[k*m+j]*((xn2-xn1)/d);
                                float dyn=match[k*m+j]*((yn2-yn1)/d);
                                float dzn=match[k*m+j]*((zn2-zn1)/d);
				grad1[k*6+0]-=dx;
				grad1[k*6+1]-=dy;
                                grad1[k*6+2]-=dz;
				grad1[k*6+3]-=dxn;
                                grad1[k*6+4]-=dyn;
                                grad1[k*6+5]-=dzn;
				sx+=dx;
				sy+=dy;
				sz+=dz;
                                sxn+=dxn;
                                syn+=dyn;
                                szn+=dzn;
			}
			grad2[j*6+0]=sx;
			grad2[j*6+1]=sy;
			grad2[j*6+2]=sz;
                        grad2[j*6+3]=sxn;
                        grad2[j*6+4]=syn;
                        grad2[j*6+5]=szn;
		}
		xyz1+=n*6;
		xyz2+=m*6;
		match+=n*m;
		grad1+=n*6;
		grad2+=m*6;
	}
}
