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

void three_nn_kernel_launcher(int b, int n, int m, const float *unknown,
                              const float *known, float *dist2, int *idx);
void three_interpolate_kernel_launcher(int b, int c, int m, int n,
                                       const float *points, const int *idx,
                                       const float *weight, float *out);
void three_interpolate_grad_kernel_launcher(int b, int c, int n, int m,
                                            const float *grad_out,
                                            const int *idx, const float *weight,
                                            float *grad_points);

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_INPUT(unknowns);
  CHECK_IS_FLOAT(unknowns);
  CHECK_INPUT(knows);
  CHECK_IS_FLOAT(knows);

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  three_nn_kernel_launcher(unknowns.size(0), unknowns.size(1), knows.size(1),
                           unknowns.data<float>(), knows.data<float>(),
                           dist2.data<float>(), idx.data<int>());

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight) {
  CHECK_INPUT(points);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_INPUT(idx);
  CHECK_IS_FLOAT(weight);
  CHECK_INPUT(weight);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  three_interpolate_kernel_launcher(points.size(0), points.size(1),
                                    points.size(2), idx.size(1),
                                    points.data<float>(), idx.data<int>(),
                                    weight.data<float>(), output.data<float>());

  return output;
}
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_INPUT(grad_out);
  CHECK_IS_FLOAT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);
  CHECK_INPUT(weight);
  CHECK_IS_FLOAT(weight);

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  three_interpolate_grad_kernel_launcher(grad_out.size(0), grad_out.size(1),
                                         grad_out.size(2), m,
                                         grad_out.data<float>(),
                                         idx.data<int>(), weight.data<float>(),
                                         output.data<float>());

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("three_nn", &three_nn, "3 Nearest Neighbor");
  m.def("three_interpolate", &three_interpolate,
        "3 Nearest Neighbor Interpolate");
  m.def("three_interpolate_grad", &three_interpolate_grad,
        "3 Nearest Neighbor Interpolate (Gradient)");
}
