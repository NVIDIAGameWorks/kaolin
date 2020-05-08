# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from kaolin.transforms import Transform
from kaolin.transforms import transforms as tfs


class VoxelODMs(Transform):
    def __init__(self, resolutions: list = [32, 128], normalize: bool = True,
                 vertex_offset: float = 0.5, single_view: bool = True):
        self.transforms = {
            resolution: {
                'to_voxel': tfs.Compose([
                    tfs.TriangleMeshToVoxelGrid(resolution,
                                                normalize=normalize,
                                                vertex_offset=vertex_offset),
                    tfs.FillVoxelGrid(thresh=0.5),
                    tfs.ExtractProjectOdmsFromVoxelGrid()]),
                'to_odms': tfs.ExtractOdmsFromVoxelGrid(),
            } for resolution in resolutions
        }

    def __call__(self, mesh):
        data = {}
        for r in self.transforms.keys():
            voxels = self.transforms[r]['to_voxel'](mesh)
            odms = self.transforms[r]['to_odms'](voxels)
            data[r] = {
                'voxels': voxels,
                'odms': odms,
            }
        return data
