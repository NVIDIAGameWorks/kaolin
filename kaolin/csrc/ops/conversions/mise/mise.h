// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

// Copyright 2019 Lars Mescheder, Michael Oechsle,
// Michael Niemeyer, Andreas Geiger, Sebastian Nowozin
//
// Permission is hereby granted, free of charge,
// to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to 
// in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <torch/custom_class.h>

namespace kaolin {

struct MISE : torch::CustomClassHolder {
private:
  struct Vector3D {
    int x = 0;
    int y = 0;
    int z = 0;

    Vector3D() = default;
    Vector3D(int in_x, int in_y, int in_z) : x(in_x), y(in_y), z(in_z) {}
  };

  struct Voxel {
    Vector3D loc{};
    unsigned int level = 0;
    bool is_leaf = true;
    unsigned long children[2][2][2]{};

    Voxel() = default;
    Voxel(const Vector3D& l, unsigned int lvl, bool leaf)
        : loc(l), level(lvl), is_leaf(leaf) {
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          for (int k = 0; k < 2; k++) {
            children[i][j][k] = 0;
          }
        }
      }
    }
  };

  struct GridPoint {
    Vector3D loc{};
    double value = 0.0;
    bool known = false;

    GridPoint() = default;
    GridPoint(const Vector3D& l, double v, bool k) : loc(l), value(v), known(k) {}
  };

  std::vector<Voxel> voxels;
  std::vector<GridPoint> grid_points;
  std::map<long, long> grid_point_hash;
  int resolution_0;
  int depth;
  double threshold;
  int voxel_size_0;
  int resolution;

  void subdivide_voxels();
  void subdivide_voxel(long idx);
  long get_voxel_idx(Vector3D loc);
  void add_grid_point(Vector3D loc);
  int get_grid_point_idx(Vector3D loc);

public:
  MISE(int32_t in_resolution_0, int32_t in_depth, double in_threshold);
  void update(at::Tensor points, at::Tensor values);
  at::Tensor query();
  at::Tensor to_dense();
  int32_t get_resolution();
};

}
