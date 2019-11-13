# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++
# cython: embedsignature = True

from libcpp.vector cimport vector

import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()




def nnsearch (A, B):
    closest = np.zeros(A.shape[0])
    cdef float min_val 
    for i, a in enumerate(A): 
        min_val = ((a-B[0])**2).sum() 
        for j, b in enumerate(B): 
            dist = ((a-b)**2).sum()
            if dist <= min_val: 
                closest[i] = j
                min_val = dist
    return closest
