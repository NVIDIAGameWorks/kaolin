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

void approxmatch_cpu(int b,int n,int m,const float * xyz1,
        const float * xyz2, float * match);

int emd_forward(THFloatTensor *xyz1, THFloatTensor *xyz2,
                THFloatTensor *match, THFloatTensor *cost);

int emd_backward(THFloatTensor *xyz1, THFloatTensor *xyz2,
                THFloatTensor *gradxyz1, THFloatTensor *gradxyz2,
                        THFloatTensor * match);

void matchcost_cpu(int b, int n, int m, const float * xyz1,
        const float * xyz2, const float * match, float * cost);

void matchcostgrad_cpu(int b, int n, int m, const float * xyz1,
        const float * xyz2, const float * match, float * grad1, float * grad2);
