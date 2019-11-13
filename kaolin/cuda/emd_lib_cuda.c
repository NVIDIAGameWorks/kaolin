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

#include <THC/THC.h>
#include "emd_cuda.h"



extern THCState *state;


int emd_forward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2,
        THCudaTensor *match, THCudaTensor * cost, THCudaTensor * temp) {
    int success = 0;


    //approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,
    //float * match, float * temp);
    success = approxmatchLauncher(xyz1->size[0],
	xyz1->size[1],
	xyz2->size[1],
	THCudaTensor_data(state, xyz1),
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, match),
	THCudaTensor_data(state, temp)
    );

    if (!success) {
    THError("aborting");
    }

    success = 0;
    success = matchcostLauncher(xyz1->size[0],
	xyz1->size[1],
	xyz2->size[1],
	THCudaTensor_data(state, xyz1),
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, match),
	THCudaTensor_data(state, cost)
	);

    if (!success) {
    THError("aborting");
    }
    return 1;
}


int emd_backward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *gradxyz1,
        THCudaTensor *gradxyz2, THCudaTensor *match) {

    int success = 0;
    success = matchcostgradLauncher(xyz1->size[0],
	xyz1->size[1],
	xyz2->size[1],
	THCudaTensor_data(state, xyz1),
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, match),
	THCudaTensor_data(state, gradxyz1),
	THCudaTensor_data(state, gradxyz2)
	);
	//int NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream)

    if (!success) {
    THError("aborting");
    }

    return 1;
}



