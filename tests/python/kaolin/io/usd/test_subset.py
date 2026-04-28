# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import os
import shutil

import torch
import pytest
from pxr import Usd

from kaolin.io import usd

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir, 'samples')


def io_data_path(fname):
    return os.path.join(__samples_path, 'io', fname)


@pytest.fixture(scope='module')
def subsets_sample_path():
    # subsets_sample.usda contains two prims with GeomSubsets across multiple families:
    #
    # /World/SimpleMesh  (UsdGeom.Mesh, tetrahedron, 4 faces)
    #   /group_a   elementType=face  familyName=part          indices=[0, 1]
    #   /group_b   elementType=face  familyName=part          indices=[2, 3]
    #   /mat_a     elementType=face  familyName=materialBind  indices=[0, 2]
    #
    # /World/SimpleGaussians  (ParticleField3DGaussianSplat, 4 splats)
    #   /region_a  elementType=''  familyName=part    indices=[0, 1]
    #   /region_b  elementType=''  familyName=region  indices=[2, 3]
    return io_data_path('subsets_sample.usda')


@pytest.fixture(scope='module')
def expected_subsets():
    # Ground truth for every subset in subsets_sample.usda, keyed by USD path.
    _all = {
        '/World/SimpleMesh/group_a': {'indices': torch.tensor([0, 1]), 'family_name': 'part'},
        '/World/SimpleMesh/group_b': {'indices': torch.tensor([2, 3]), 'family_name': 'part'},
        '/World/SimpleMesh/mat_a':   {'indices': torch.tensor([0, 2]), 'family_name': 'materialBind'},
        '/World/SimpleGaussians/region_a': {'indices': torch.tensor([0, 1]), 'family_name': 'part'},
        '/World/SimpleGaussians/region_b': {'indices': torch.tensor([2, 3]), 'family_name': 'region'},
    }

    def _get(prim_path, family_name=None):
        return {
            k: v for k, v in _all.items()
            if k.startswith(prim_path)
            and (family_name is None or v['family_name'] == family_name)
        }

    return _get


def assert_subsets_match(result, expected):
    assert set(result.keys()) == set(expected.keys())
    for path, info in result.items():
        assert torch.equal(info['indices'], expected[path]['indices']), \
            f"{path}: indices mismatch"
        assert info['family_name'] == expected[path]['family_name'], \
            f"{path}: family_name mismatch"


@pytest.fixture(scope='class')
def out_dir():
    d = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestImportSubsets:
    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_all_subsets_mesh(self, subsets_sample_path, expected_subsets, input_stage):
        path_or_stage = Usd.Stage.Open(subsets_sample_path) if input_stage else subsets_sample_path
        result = usd.import_subsets(path_or_stage, '/World/SimpleMesh')
        assert_subsets_match(result, expected_subsets('/World/SimpleMesh'))

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_subsets_part_family_mesh(self, subsets_sample_path, expected_subsets, input_stage):
        path_or_stage = Usd.Stage.Open(subsets_sample_path) if input_stage else subsets_sample_path
        result = usd.import_subsets(path_or_stage, '/World/SimpleMesh', family_name='part')
        assert_subsets_match(result, expected_subsets('/World/SimpleMesh', 'part'))

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_subsets_materialBind_family_mesh(self, subsets_sample_path, expected_subsets, input_stage):
        path_or_stage = Usd.Stage.Open(subsets_sample_path) if input_stage else subsets_sample_path
        result = usd.import_subsets(path_or_stage, '/World/SimpleMesh', family_name='materialBind')
        assert_subsets_match(result, expected_subsets('/World/SimpleMesh', 'materialBind'))

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_all_subsets_gaussians(self, subsets_sample_path, expected_subsets, input_stage):
        path_or_stage = Usd.Stage.Open(subsets_sample_path) if input_stage else subsets_sample_path
        result = usd.import_subsets(path_or_stage, '/World/SimpleGaussians')
        assert_subsets_match(result, expected_subsets('/World/SimpleGaussians'))

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_subsets_part_family_gaussians(self, subsets_sample_path, expected_subsets, input_stage):
        path_or_stage = Usd.Stage.Open(subsets_sample_path) if input_stage else subsets_sample_path
        result = usd.import_subsets(path_or_stage, '/World/SimpleGaussians', family_name='part')
        assert_subsets_match(result, expected_subsets('/World/SimpleGaussians', 'part'))

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_subsets_region_family_gaussians(self, subsets_sample_path, expected_subsets, input_stage):
        path_or_stage = Usd.Stage.Open(subsets_sample_path) if input_stage else subsets_sample_path
        result = usd.import_subsets(path_or_stage, '/World/SimpleGaussians', family_name='region')
        assert_subsets_match(result, expected_subsets('/World/SimpleGaussians', 'region'))

    def test_import_subsets_invalid_prim_path(self, subsets_sample_path):
        with pytest.raises(ValueError, match="No prim found at path"):
            usd.import_subsets(subsets_sample_path, '/World/DoesNotExist')

    def test_import_subsets_empty(self, subsets_sample_path):
        result = usd.import_subsets(subsets_sample_path, '/World')
        assert result == {}

    def test_import_subsets_prim_object(self, subsets_sample_path, expected_subsets):
        stage = Usd.Stage.Open(subsets_sample_path)
        prim = stage.GetPrimAtPath('/World/SimpleMesh')
        result = usd.import_subsets(stage, prim)
        assert_subsets_match(result, expected_subsets('/World/SimpleMesh'))


class TestAddSubset:
    def test_add_subset_mesh_file_path(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_file.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        usd.add_subset(path, '/World/Mesh', 'mysubset', torch.tensor([0, 2]))

        result = usd.import_subsets(path, '/World/Mesh')
        assert len(result) == 1
        info = next(iter(result.values()))
        assert info['family_name'] == 'part'
        assert torch.equal(info['indices'], torch.tensor([0, 2]))

    def test_add_subset_custom_family_name(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_family.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        usd.add_subset(path, '/World/Mesh', 'mysubset', torch.tensor([1, 2]), family_name='materialBind')

        result = usd.import_subsets(path, '/World/Mesh')
        assert len(result) == 1
        info = next(iter(result.values()))
        assert info['family_name'] == 'materialBind'
        assert torch.equal(info['indices'], torch.tensor([1, 2]))

    def test_add_subset_mesh_stage(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_stage.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        stage = Usd.Stage.Open(path)
        usd.add_subset(stage, '/World/Mesh', 'mysubset', torch.tensor([1, 3]))

        result = usd.import_subsets(stage, '/World/Mesh')
        assert len(result) == 1
        info = next(iter(result.values()))
        assert info['family_name'] == 'part'
        assert torch.equal(info['indices'], torch.tensor([1, 3]))

    def test_add_subset_returns_prim(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_prim.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        # With a stage object the prim stays valid (caller holds the stage reference)
        stage = Usd.Stage.Open(path)
        result = usd.add_subset(stage, '/World/Mesh', 'mysubset', torch.tensor([0]))
        assert isinstance(result, Usd.Prim)
        assert result.IsValid()

    def test_add_subset_prim_path_string(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_string_path.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        usd.add_subset(path, '/World/Mesh', 'strsubset', torch.tensor([3]))
        result = usd.import_subsets(path, '/World/Mesh')
        assert len(result) == 1

    def test_add_subset_prim_object(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_prim_obj.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        stage = Usd.Stage.Open(path)
        prim = stage.GetPrimAtPath('/World/Mesh')
        usd.add_subset(stage, prim, 'primsubset', torch.tensor([0, 1, 2]))

        result = usd.import_subsets(stage, '/World/Mesh')
        assert len(result) == 1

    def test_add_subset_indices_preserved(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_indices.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                  [0., 0., 1.], [1., 1., 0.], [0., 1., 1.],
                                  [1., 0., 1.], [1., 1., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                               [4, 5, 6], [4, 5, 7], [4, 6, 7], [5, 6, 7]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        # indices must be valid face indices (< 8), use [5, 3, 7] instead
        indices_in = torch.tensor([5, 3, 7], dtype=torch.long)
        usd.add_subset(path, '/World/Mesh', 'mysubset', indices_in)

        result = usd.import_subsets(path, '/World/Mesh')
        info = next(iter(result.values()))
        # USD sorts indices on write; compare as sets
        assert set(info['indices'].tolist()) == set(indices_in.tolist())

    def test_add_subset_duplicate_raises(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_dup.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        usd.add_subset(path, '/World/Mesh', 'mysubset', torch.tensor([0]))
        with pytest.raises(ValueError, match="already exists"):
            usd.add_subset(path, '/World/Mesh', 'mysubset', torch.tensor([1]))

    def test_add_subset_override(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_override.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        usd.add_subset(path, '/World/Mesh', 'mysubset', torch.tensor([0]), family_name='part')
        usd.add_subset(path, '/World/Mesh', 'mysubset', torch.tensor([2, 3]),
                       family_name='materialBind', override=True)

        result = usd.import_subsets(path, '/World/Mesh')
        assert len(result) == 1
        info = next(iter(result.values()))
        assert torch.equal(info['indices'], torch.tensor([2, 3]))
        assert info['family_name'] == 'materialBind'

    def test_add_subset_invalid_prim_path(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_invalid.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        with pytest.raises(ValueError, match="No prim found at path"):
            usd.add_subset(path, '/World/DoesNotExist', 'mysubset', torch.tensor([0]))

    def test_add_multiple_subsets_same_prim(self, out_dir):
        path = os.path.join(out_dir, 'mesh_subset_multi.usda')
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')

        indices_a = torch.tensor([0, 1])
        indices_b = torch.tensor([2, 3])
        usd.add_subset(path, '/World/Mesh', 'group_a', indices_a, family_name='part')
        usd.add_subset(path, '/World/Mesh', 'group_b', indices_b, family_name='part')

        result = usd.import_subsets(path, '/World/Mesh')
        assert len(result) == 2

        paths = list(result.keys())
        assert any('group_a' in p for p in paths)
        assert any('group_b' in p for p in paths)

        for p, info in result.items():
            assert info['family_name'] == 'part'
            if 'group_a' in p:
                assert torch.equal(info['indices'], indices_a)
            else:
                assert torch.equal(info['indices'], indices_b)
