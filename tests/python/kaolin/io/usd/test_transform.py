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

import pytest
import torch
from pxr import Usd, UsdGeom, Gf

import kaolin
from kaolin.io.usd.transform import get_local_to_world_transform, set_local_to_world_transform


@pytest.fixture(scope='class')
def out_dir():
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.parametrize('with_stage', [True, False])
class TestLocalToWorldTransform:
    def test_identity_returns_none(self, with_stage, out_dir):
        """A prim with no xform op has identity transform → returns None."""
        file_path = os.path.join(out_dir, f'test_identity_returns_none.usda')
        stage = kaolin.io.usd.create_stage(file_path)
        UsdGeom.Xform.Define(stage, '/World')
        
        if with_stage:
            result = get_local_to_world_transform(stage, '/World')
        else:
            stage.Save()
            del stage
            result = get_local_to_world_transform(file_path, '/World')
        assert result is None

    def test_non_identity_returns_tensor(self, with_stage, out_dir):
        """A prim with a pure translation returns a (4,4) tensor with correct values.

        Kaolin convention: translation in last column (result of transposing USD row-major matrix).
        """
        file_path = os.path.join(out_dir, 'test_non_identity_returns_tensor.usda')
        stage = kaolin.io.usd.create_stage(file_path)
        prim = UsdGeom.Xform.Define(stage, '/World').GetPrim()
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(1, 2, 3))
        
        if with_stage:
            result = get_local_to_world_transform(stage, '/World')
        else:
            stage.Save()
            del stage
            result = get_local_to_world_transform(file_path, '/World')
        expected = torch.eye(4, dtype=torch.float64)
        expected[0, 3] = 1.0
        expected[1, 3] = 2.0
        expected[2, 3] = 3.0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_with_time(self, with_stage, out_dir):
        """Animated xform: returns correct tensor at the authored time code."""
        file_path = os.path.join(out_dir, 'test_with_time.usda')
        stage = kaolin.io.usd.create_stage(file_path)
        prim = UsdGeom.Xform.Define(stage, '/World').GetPrim()
        xform_op = UsdGeom.Xformable(prim).MakeMatrixXform()
        # Translation (4, 5, 6) stored in USD row 3
        t_matrix_1 = Gf.Matrix4d(1, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 1, 0,
                                 4, 5, 6, 1)
        xform_op.Set(t_matrix_1, 1)

        t_matrix_2 = Gf.Matrix4d(1, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 1, 0,
                                 6, 5, 4, 1)
        xform_op.Set(t_matrix_2, 2)

        if with_stage:
            result = get_local_to_world_transform(stage, '/World', time=1.5)
        else:
            stage.Save()
            del stage
            result = get_local_to_world_transform(file_path, '/World', time=1.5)

        expected = torch.eye(4, dtype=torch.float64)
        expected[0, 3] = 5.0
        expected[1, 3] = 5.0
        expected[2, 3] = 5.0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_set_then_get_roundtrip(self, with_stage, out_dir):
        """set → get returns the same (4,4) matrix within float tolerance."""
        file_path = os.path.join(out_dir, 'test_set_then_get_roundtrip.usda')
        stage = kaolin.io.usd.create_stage(file_path)
        UsdGeom.Xform.Define(stage, '/World')
        target = torch.tensor([[1, 0, 0, 7],
                               [0, 1, 0, 8],
                               [0, 0, 1, 9],
                               [0, 0, 0, 1]], dtype=torch.float64)
        if with_stage:
            set_local_to_world_transform(stage, '/World', target)
        else:
            set_local_to_world_transform(file_path, '/World', target)
        stage.Save()
        if with_stage:
            result = get_local_to_world_transform(stage, '/World')
        else:
            stage.Save()
            del stage
            result = get_local_to_world_transform(file_path, '/World')

        assert torch.allclose(result, target, atol=1e-5)

    def test_set_with_time(self, with_stage, out_dir):
        """Animated: set at time=1, retrieve at time=1 returns the same matrix."""
        file_path = os.path.join(out_dir, 'test_set_with_time.usda')
        stage = kaolin.io.usd.create_stage(file_path)
        UsdGeom.Xform.Define(stage, '/World')
        target_1 = torch.tensor([[1, 0, 0, 1],
                                 [0, 1, 0, 2],
                                 [0, 0, 1, 3],
                                 [0, 0, 0, 1]], dtype=torch.float64)

        target_2 = torch.tensor([[1, 0, 0, 3],
                                 [0, 1, 0, 2],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 1]], dtype=torch.float64)

        expected = torch.tensor([[1, 0, 0, 2],
                                 [0, 1, 0, 2],
                                 [0, 0, 1, 2],
                                 [0, 0, 0, 1]], dtype=torch.float64)

        if with_stage:
            set_local_to_world_transform(stage, '/World', target_1, time=1)
            set_local_to_world_transform(stage, '/World', target_2, time=2)
            result = get_local_to_world_transform(stage, '/World', time=1.5)
        else:
            stage.Save()
            del stage
            set_local_to_world_transform(file_path, '/World', target_1, time=1)
            set_local_to_world_transform(file_path, '/World', target_2, time=2)
            result = get_local_to_world_transform(file_path, '/World', time=1.5)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_set_nested_hierarchy(self, with_stage, out_dir):
        """Child under a parent with a known xform: world transform is set correctly.

        Parent has translation (10, 0, 0). Setting child's world transform to (15, 0, 0)
        means the child's local transform should be (5, 0, 0) to compose correctly.
        """
        file_path = os.path.join(out_dir, 'test_set_nested_hierarchy.usda')
        stage = kaolin.io.usd.create_stage(file_path)
        parent = UsdGeom.Xform.Define(stage, '/World')
        child = UsdGeom.Xform.Define(stage, '/World/Child')
        UsdGeom.XformCommonAPI(parent).SetTranslate(Gf.Vec3d(10, 0, 0))

        target = torch.tensor([[1, 0, 0, 15],
                               [0, 1, 0,  0],
                               [0, 0, 1,  0],
                               [0, 0, 0,  1]], dtype=torch.float64)

        if with_stage:
            set_local_to_world_transform(stage, child.GetPrim(), target)
            result = get_local_to_world_transform(stage, child.GetPrim())
        else:
            stage.Save()
            del stage
            set_local_to_world_transform(file_path, '/World/Child', target)
            result = get_local_to_world_transform(file_path, '/World/Child')

        assert torch.allclose(result, target, atol=1e-5)
