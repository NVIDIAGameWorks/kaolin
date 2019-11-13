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

void query_ball_point_kernel_launcher(int b, int n, int m, float radius,
                                      int nsample, const float *new_xyz,
                                      const float *xyz, int *idx);

void query_ball_random_point_kernel_launcher(int seed, int b, int n, int m,
                                             float radius, int nsample,
                                             const float *new_xyz,
                                             const float *xyz, int *idx);

void gather_by_index_kernel_launcher(int b, int c, int n, int npoints,
                                     int nsample, const float *points,
                                     const int *idx, float *out);

void gather_by_index_grad_kernel_launcher(int b, int c, int n, int npoints,
                                          int nsample, const float *grad_out,
                                          const int *idx, float *grad_points);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_INPUT(new_xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_INPUT(xyz);
  CHECK_IS_FLOAT(xyz);

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  query_ball_point_kernel_launcher(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                   radius, nsample, new_xyz.data<float>(),
                                   xyz.data<float>(), idx.data<int>());

  return idx;
}

at::Tensor ball_random_query(int seed, at::Tensor new_xyz, at::Tensor xyz,
                             const float radius, const int nsample) {
  CHECK_INPUT(new_xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_INPUT(xyz);
  CHECK_IS_FLOAT(xyz);

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  query_ball_random_point_kernel_launcher(
      seed, xyz.size(0), xyz.size(1), new_xyz.size(1), radius, nsample,
      new_xyz.data<float>(), xyz.data<float>(), idx.data<int>());

  return idx;
}

at::Tensor gather_by_index(at::Tensor points, at::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  gather_by_index_kernel_launcher(
      points.size(0), points.size(1), points.size(2), idx.size(1), idx.size(2),
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

  gather_by_index_grad_kernel_launcher(
      grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
      grad_out.data<float>(), idx.data<int>(), output.data<float>());

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query, "Ball Query");
  m.def("ball_random_query", &ball_random_query,
        "Ball Query With Random Sampling");
  m.def("gather_by_index", &gather_by_index, "Gather Points By Index");
  m.def("gather_by_index_grad", &gather_by_index_grad,
        "Gather Points By Index (Gradient)");
}
