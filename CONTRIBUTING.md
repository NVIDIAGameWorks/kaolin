# Contributing guidelines

Contributions are welcome!
You can send us pull requests to help improve Kaolin, if you are just getting started, Gitlab has a [how to](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html).

Kaolin team members will be assigned to review your pull requests. Once your change passes the review and the continuous integration checks, a Kaolin member will approve and merge them to the repository.

If you want to contribute, [Gitlab issues](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat/-/issues) are a good starting point, especially the ones with the label [good first issue](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat/-/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=good%20first%20issue). If you started working on a issue, leave a comment so other people know that you're working on it, you can also coordinate with others on the issue comment threads.

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

1) Read these Guidelines in full.

2) Please take a look at the LICENSE (it's Apache 2.0).

3) Make sure you sign your commits. E.g. use ``git commit -s`` when commiting.

4) Check your changes are consistent with the [Standards and Coding Style](CONTRIBUTING.md#standards-and-coding-style).

5) Make sure all unittests finish successfully before sending PR.

6) Send your Pull Request to the `master` branch

## Guidelines for Contributing Code

### Running tests

In order to verify that your change does not break anything, a number of checks must be 
passed. For example, running unit tests, making sure that all example
notebooks and recipes run without error, and docs build correctly. For unix-based
system, we provide a script to execute all of these tests locally:

```
pip install -r tools/ci_requirements.txt
pip install -r tools/doc_requirements.txt 

bash tools/linux/run_tests.sh all 
```

If you also want to run integration tests, see [tests/integration/](tests/integration/), specifically
[Dash3D tests](tests/integration/experimental/dash3d/README.md).

### Documentation

All new additions to the Kaolin API must be properly documented. Additional information
on documentation is provided in [our guide](docs/README.md).


### Signing your commits

All commits must be signed using ``git commit -s``.
If you forgot to sign previous commits you can amend them as follows:
* ``git commit -s --amend`` for the last commit.
* ``git rebase --signoff`` for all the commits of your pull request.


### Standards and Coding Style
#### General guidelines
* New features must include unit tests which help guarantee correctness in the present and future.
* API changes should be minimal and backward compatible. Any changes that break backward compatibility should be carefully considered and tracked so that they can be included in the release notes.
* New features may not accepted if the cost of maintenance is too high in comparison of its benefit, they may also be integrated to contrib subfolders for minimal support and maintenance before eventually being integrated to the core.

#### Writing Tests
All tests should use [pytest](https://docs.pytest.org/en/latest/) and [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) frameworks. The tests should be placed in [tests/python directory](tests/python/), which should follows the directory structure of [kaolin](kaolin/). For example,
test for `kaolin/io/obj.py` should be placed into `tests/pyhon/kaolin/io/test_obj.py`. 

#### License
Include a license at the top of new files.



##### C/C++/CUDA
```cpp
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

##### Python
```python
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

When non-trivial changes are made, the license should be changed accordingly. For instance, if the file is originally authored in 2021, a few typos get fixed in 2022, a paragraph or subroutine is added in 2023, and a major rev2.0 is created in 2024, you would in 2024 write:
"Copyright (c) 2021,23-24 NVIDIA CORPORATION & AFFILIATES"

#### Code organization
* [kaolin](kaolin/) - The core Kaolin library, comprised of python modules, 
except for code under [csrc](kaolin/csrc) or [experimental](kaolin/experimental).
  * [csrc](kaolin/csrc/) - Directory for all the C++ / CUDA implementations of custom ops.
    The gpu ops parts will be under the subdirectory [csrc/cuda](kaolin/csrc/cuda)
    while the cpu parts will be under the subdirectory [csrc/cpu](kaolin/csrc/cpu).
  * [io](kaolin/io/) - Module of all the I/O features of Kaolin, such a importing and exporting 3D models.
  * [metrics](kaolin/metrics) - Module of all the metrics that can be used as differentiable loss or distance.
  * [ops](kaolin/ops/) - Module of all the core operations of kaolin on different 3D representations.
  * [render](kaolin/render/) - Module of all the differentiable renderers modules and advanced implementations.
  * [utils](kaolin/utils/) - Module of all the utility features for debugging and testing.
  * [visualize](kaolin/visualize/) - Module of all the visualization modules.
  * [experimental](kaolin/experimental/) - Contains less thoroughly tested components for early adoption.
* [examples](examples/) - Examples of Kaolin usage
* [tests](tests/) - Tests for all Kaolin

#### C++ coding style
We follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), with exception on naming for functions / methods, where we use **snake_case**.

Files structure should be:
- ``*.cuh`` files are for reusable device functions (using ``static inline __device__`` for definition)

```cpp
// file is kaolin/csrc/ops/add.cuh

#ifndef KAOLIN_OPS_ADD_CUH_
#define KAOLIN_OPS_ADD_CUH_

namespace kaolin {

static inline __device__ float add(float a, float b) {
  return a + b;
}

static inline __device__ double add(double a, double b) {
  return a + b;
}

}  // namespace kaolin

#endif  // KAOLIN_OPS_ADD_CUH_
```

- ``*.cpp`` files (except specific files like [bindings.cpp](kaolin/csrc/bindings.cpp)) should be used for defining the functions that will be directly binded to python, those functions should only be responsible for checking inputs device / memory layout / size, generating output (if possible) and call the base function in the ``_cuda.cu`` or ``_cpu.cpp``. The kernel launcher should be declared at the beginning of the file or from an included header (if reused).

```cpp
// file is kaolin/csrc/ops/foo.cpp

#include <ATen/ATen.h>

namespace kaolin {

#if WITH_CUDA
void foo_cuda_impl(
    at::Tensor lhs,
    at::Tensor rhs,
    at::Tensor output);
#endif  // WITH_CUDA

at::Tensor foo_cuda(
    at::Tensor lhs,
    at::Tensor rhs) {
  at::TensorArg lhs_arg{lhs, "lhs", 1}, rhs_arg{rhs, "rhs", 2};
  at::checkSameGPU("foo_cuda", lhs_arg, rhs_arg);
  at::checkAllContiguous("foo_cuda", {lhs_arg, rhs_arg});
  at::checkSameSize("foo_cuda", lhs_arg, rhs_arg);
  at::checkSameType("foo_cuda", lhs_arg, rhs_arg);

  at::Tensor output = at::zeros_like(lhs);

#if WITH_CUDA
  foo_cuda_impl(lhs, rhs, output);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
  return output;
}

}  // namespace kaolin
```

- ``*_cuda.cu`` files are for dispatching given the inputs types and implementing the operations on GPU, by using the Torch C++ API and/or launching a custom cuda kernel.

```cpp
// file is kaolin/csrc/ops/foo_cuda.cu

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include "./add.cuh"

namespace kaolin {

template<typename scalar_t>
__global__
void foo_cuda_kernel(
    const scalar_t* __restrict__ lhs,
    const scalar_t* __restrict__ rhs,
    const int numel,
    scalar_t* __restrict__ output) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < numel; i++) {
    output[i] = add(lhs[i], rhs[i]);
  }
}

void foo_cuda_impl(
    at::Tensor lhs,
    at::Tensor rhs,
    at::Tensor output) {
  const int threads = 1024;
  const int blocks = 64;
  AT_DISPATCH_FLOATING_TYPES(lhs.scalar_type(), "foo_cuda", [&] {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(output));
    auto stream = at::cuda::getCurrentCUDAStream();
    foo_cuda_kernel<<<blocks, threads, 0, stream>>>(
        lhs.data_ptr<scalar_t>(),
        rhs.data_ptr<scalar_t>(),
        lhs.numel(),
        output.data_ptr<scalar_t>());
  });
}

}  // namespace kaolin
```

- ``*.h`` files are for declaring functions that will be binded to Python, those header files are to be included in [bindings.cpp](kaolin/csrc/bindings.cpp).

```cpp
// file is kaolin/csrc/ops/foo.h

#ifndef KAOLIN_OPS_FOO_H_
#define KAOLIN_OPS_FOO_H_

#include <ATen/ATen.h>

namespace kaolin {

at::Tensor foo_cuda(
    at::Tensor lhs,
    at::Tensor rhs);

}  // namespace kaolin

#endif  // KAOLIN_OPS_FOO_H_
```


#### Python coding style
We follow [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/) with some exceptions listed in [flake8 config file](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat/.flake8) and generally follow PyTorch naming conventions.

It is enforced using [flake8](https://pypi.org/project/flake8/), with [flake8-bugbear](https://pypi.org/project/flake8-bugbear/), [flake8-comprehensions](https://pypi.org/project/flake8-comprehensions/), [flake8-mypy](https://pypi.org/project/flake8-mypy/) and [flake8-pyi](https://pypi.org/project/flake8-pyi/)

To run flake8 execute ``flake8 --config=.flake8 .`` from the [root of kaolin](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat).

On top of that we use prefixes (``packed\_``, ``padded\_``) to indicate that a module / op is specific to a layout, an , all ops of the same purpose for different layouts should be in the same file.

[tests/python/kaolin/](tests/python/kaolin) should follows the same directory structure of [kaolin/](kaolin/). E.g. each module kaolin/path/to/mymodule.py should have a corresponding tests/python/kaolin/path/to/test\_mymodule.py.


