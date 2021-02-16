// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef KAOLIN_OPS_SPC_CONVOLUTION_CUH_
#define KAOLIN_OPS_SPC_CONVOLUTION_CUH_

#include "../../spc_math.h"

#include <vector>

#include <cublas_v2.h>

constexpr int MAX_GRID = 65535;

template <typename Itype> struct pVector {
  Itype *ptr_;
  int size_;

  pVector(Itype *ptr, int size) : ptr_(ptr), size_(size) {}
  int size() const { return size_; };
  Itype *data() { return ptr_; };
  const Itype *data() const { return ptr_; };
};

// Input index to output index mapping in ptr, sise pair
// Used for device pointer and size
template <typename Itype> using pInOutMaps = std::vector<pVector<Itype>>;

#endif // KAOLIN_OPS_SPC_CONVOLUTION_CUH
