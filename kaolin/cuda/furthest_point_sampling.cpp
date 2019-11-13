// Copyright (c) 2017, Geometric Computation Group of Stanford University

// The MIT License (MIT)

// Copyright (c) 2017 Charles R. Qi

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

#include "util.h"
#include <torch/extension.h>

void gather_by_index_kernel_launcher(int b, int c, int n, int npoints,
                                     const float *points, const int *idx,
                                     float *out);
void gather_by_index_grad_kernel_launcher(int b, int c, int n, int npoints,
                                          const float *grad_out, const int *idx,
                                          float *grad_points);
void furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp,
                                             int *idxs);

at::Tensor gather_by_index(at::Tensor points, at::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  gather_by_index_kernel_launcher(
      points.size(0), points.size(1), points.size(2), idx.size(1),
      points.data<float>(), idx.data<int>(), output.data<float>());

  return output;
}

at::Tensor gather_by_index_grad(at::Tensor grad_out, at::Tensor idx,
                                const int n) {
  CHECK_INPUT(grad_out);
  CHECK_IS_FLOAT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  gather_by_index_grad_kernel_launcher(grad_out.size(0), grad_out.size(1), n,
                                       idx.size(1), grad_out.data<float>(),
                                       idx.data<int>(), output.data<float>());

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  furthest_point_sampling_kernel_launcher(
      points.size(0), points.size(1), nsamples, points.data<float>(),
      tmp.data<float>(), output.data<int>());

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sampling", &furthest_point_sampling,
        "Furthest Point Sampling");
  m.def("gather_by_index", &gather_by_index, "Gather Points By Index");
  m.def("gather_by_index_grad", &gather_by_index_grad,
        "Gather Points By Index (Gradient)");
}
