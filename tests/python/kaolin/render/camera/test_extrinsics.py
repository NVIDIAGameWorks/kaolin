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

import math

import pytest
import itertools
import numpy as np
import torch

from kaolin.render.camera import CameraExtrinsics
from kaolin.utils.testing import FLOAT_TYPES, ALL_DEVICES

_LOOK_AT_DATA_IDX = (0, 1, 2)                       # Lookat data sample indices
_CAM_POS_DATA_IDX = (0, 1, 2)                       # Cam orientation / pos data sample indices
_VECTOR_DATA_IDX = (0, 1, 2)                        # Point / vectors data sample indices
_RAY_DIR_DATA_IDX = (0, 1, 2)                       # Rays data sample indices
_YPR_DATA_IDX = (0, 1)                              # Yaw-pitch-roll precomputed data sample indices
_YPR_ANGLES_IDX = range(3 * 5**3)                   # Yaw-pitch-roll angle combinations sample indices
_IS_BATCH_DIM = (False, True)                       # Add batch dim
_IS_ROW_DIM = (False, True)                         # Add row dim to vectors (i.e: (3,1) v.s. (3,))
_IS_NUMPY = (False, True)                           # Input is a numpy array


# data_idx, with_batch_dim, with_row_dim
@pytest.fixture(params=itertools.product(_LOOK_AT_DATA_IDX, _IS_BATCH_DIM, _IS_ROW_DIM, _IS_NUMPY))
def lookat_data(request):
    data = list()

    def _add_entry(at, eye, up, view_matrix):
        data.append({
            'at': torch.from_numpy(np.array(at)),
            'eye': torch.from_numpy(np.array(eye)),
            'up': torch.from_numpy(np.array(up)),
            'view_matrix': torch.from_numpy(np.array(view_matrix)).T.unsqueeze(0)
        })

    # View matrices here are row major, transposition required
    _add_entry(at=[1.0, 2.0, 3.0],
               eye=[4.0, 5.0, 6.0],
               up=[7.0, 8.0, 9.0],
               view_matrix=[[-0.40824827551841736, -0.7071067094802856, 0.5773502588272095, 0.0],
                            [0.8164965510368347, 0.0, 0.5773502588272095, 0.0],
                            [-0.40824827551841736, 0.7071067094802856, 0.5773502588272095, 0.0],
                            [-0.0, -1.4142136573791504, -8.660253524780273, 1.0]]
               )
    _add_entry(at=[0.0, 0.0, 0.0],
               eye=[0.0, 0.0, -1.0],
               up=[0.0, 1.0, 0.0],
               view_matrix=[[-1.0, 0.0, -0.0, 0.0],
                            [0.0, 1.0, -0.0, 0.0],
                            [0.0, -0.0, -1.0, 0.0],
                            [-0.0, -0.0, -1.0, 1.0]]
               )
    _add_entry(at=[2.0, 2.0, 2.0],
               eye=[0.0, -10.0, -1.0],
               up=[0.0, 0.001, 0.1],
               view_matrix=[[0.9863256812095642, -0.04103561490774155, -0.1596173793077469, 0.0],
                            [-0.16479960083961487, -0.2358890324831009, -0.9577043056488037, 0.0],
                            [0.0016479960177093744, 0.9709132313728333, -0.23942607641220093, 0.0],
                            [-1.646347999572754, -1.387977123260498, -9.816469192504883, 1.0]]
               )
    data_idx = request.param[0]
    is_batch_dim = request.param[1]
    is_row_dim = request.param[2]
    is_numpy = request.param[3]

    entry = data[data_idx]
    if is_batch_dim:  # Return a batch of 2 (same data twice)
        # second_entry = data[(data_idx + 1) % len(_LOOK_AT_DATA_IDX)]
        entry = {k: (torch.stack((v, v), dim=0) if k != 'view_matrix' else torch.cat((v, v), dim=0))
                 for k, v in entry.items()}
    if is_row_dim:  # Shape ([B], 3, 1)
        entry = {k: (v.unsqueeze(-1) if k != 'view_matrix' else v) for k, v in entry.items()}
    if is_numpy:
        entry = {k: v.numpy() if k != 'view_matrix' else v for k, v in entry.items()}

    return entry


# data_idx, with_batch_dim, with_row_dim
@pytest.fixture(params=itertools.product(_CAM_POS_DATA_IDX, _IS_BATCH_DIM, _IS_ROW_DIM, _IS_NUMPY))
def cam_pos_data(request):
    data = list()

    def _add_entry(cam_pos, cam_dir, view_matrix):
        data.append({
            'cam_pos': torch.from_numpy(np.array(cam_pos)),
            'cam_dir': torch.from_numpy(np.array(cam_dir)),
            'view_matrix': torch.from_numpy(np.array(view_matrix)).T.unsqueeze(0)
        })

    # View matrices here are row major, transposition required
    _add_entry(cam_pos=[4.0, 5.0, 6.0],
               cam_dir=[[-0.40824827551841736, -0.7071067094802856, 0.5773502588272095],
                        [0.8164965510368347, 0.0, 0.5773502588272095],
                        [-0.40824827551841736, 0.7071067094802856, 0.5773502588272095]],
               view_matrix=[[-0.40824827551841736, -0.7071067094802856, 0.5773502588272095, 0.0],
                            [0.8164965510368347, 0.0, 0.5773502588272095, 0.0],
                            [-0.40824827551841736, 0.7071067094802856, 0.5773502588272095, 0.0],
                            [-0.0, -1.4142136573791504, -8.660253524780273, 1.0]]
               )
    _add_entry(cam_pos=[0.0, 0.0, -1.0],
               cam_dir=[[-1.0, 0.0, -0.0],
                        [0.0, 1.0, -0.0],
                        [0.0, -0.0, -1.0]],
               view_matrix=[[-1.0, 0.0, -0.0, 0.0],
                            [0.0, 1.0, -0.0, 0.0],
                            [0.0, -0.0, -1.0, 0.0],
                            [-0.0, -0.0, -1.0, 1.0]]
               )
    _add_entry(cam_pos=[0.0, -10.0, -1.0],
               cam_dir=[[0.9863256812095642, -0.04103561490774155, -0.1596173793077469],
                        [-0.16479960083961487, -0.2358890324831009, -0.9577043056488037],
                        [0.0016479960177093744, 0.9709132313728333, -0.23942607641220093]],
               view_matrix=[[0.9863256812095642, -0.04103561490774155, -0.1596173793077469, 0.0],
                            [-0.16479960083961487, -0.2358890324831009, -0.9577043056488037, 0.0],
                            [0.0016479960177093744, 0.9709132313728333, -0.23942607641220093, 0.0],
                            [-1.646347999572754, -1.387977123260498, -9.816469192504883, 1.0]]
               )
    data_idx = request.param[0]
    is_batch_dim = request.param[1]
    is_row_dim = request.param[2]
    is_numpy = request.param[3]

    entry = data[data_idx]
    if is_batch_dim:  # Return a batch of 2 (same data twice)
        entry = {k: (torch.stack((v, v), dim=0) if k != 'view_matrix' else torch.cat((v, v), dim=0))
                 for k, v in entry.items()}
    if is_row_dim:  # Shape ([B], 3, 1)
        entry = {k: (v.unsqueeze(-1) if k == 'cam_pos' else v) for k, v in entry.items()}
    if is_numpy:
        entry = {k: v.numpy() if k != 'view_matrix' else v for k, v in entry.items()}
    return entry


@pytest.fixture(params=_VECTOR_DATA_IDX)
def vectors_data(request):
    data_idx = request.param
    if data_idx == 0:
        return torch.eye(3)
    elif data_idx == 1:
        return torch.Tensor([[0.1, 0.2, 0.3]])
    elif data_idx == 2:
        return torch.Tensor([0.1, 1.0, 10.0]).unsqueeze(0).repeat(10, 1)


@pytest.fixture(params=_RAY_DIR_DATA_IDX)
def ray_direction_data(request):
    data_idx = request.param
    if data_idx == 0:
        ray_orig = torch.eye(3)
        ray_dir = torch.eye(3)
    elif data_idx == 1:
        ray_orig = torch.Tensor([[0.1, 0.2, 0.3]])
        ray_dir = torch.Tensor([[0.5, 0.5, 0.5]])
    elif data_idx == 2:
        ray_orig = torch.Tensor([0.1, 1.0, 10.0]).unsqueeze(0).repeat(10, 1)
        ray_dir = torch.Tensor([
            [0.1, 1.0, 8.0],
            [0.2, 5.0, 0.3],
            [0.3, 2.0, 7.0],
            [0.4, 3.0, 10.0],
            [0.5, 7.0, 2.0],
            [0.6, -0.1, 0.0],
            [0.7, 4.0, 3.0],
            [0.8, 0.0, 10.0],
            [0.9, 0.9, 0.9],
            [1.0, 1.0, 10.0]
        ])
    ray_dir = torch.nn.functional.normalize(ray_dir, dim=1)
    return {'ray_orig': ray_orig, 'ray_dir': ray_dir}

# data_idx, with_batch_dim, with_row_dim
@pytest.fixture(params=_IS_BATCH_DIM)
def sample_lookat_data(request):

    def _make_entry(at, eye, up, view_matrix):
        return {
            'at': torch.from_numpy(np.array(at)),
            'eye': torch.from_numpy(np.array(eye)),
            'up': torch.from_numpy(np.array(up)),
            'view_matrix': torch.from_numpy(np.array(view_matrix)).T.unsqueeze(0)
        }

    # View matrices here are row major, transposition required
    entry = _make_entry(
        at=[1.0, 2.0, 3.0],
        eye=[4.0, 5.0, 6.0],
        up=[7.0, 8.0, 9.0],
        view_matrix=[[-0.40824827551841736, -0.7071067094802856, 0.5773502588272095, 0.0],
                     [0.8164965510368347, 0.0, 0.5773502588272095, 0.0],
                     [-0.40824827551841736, 0.7071067094802856, 0.5773502588272095, 0.0],
                     [-0.0, -1.4142136573791504, -8.660253524780273, 1.0]]
    )
    is_batch_dim = request.param
    if is_batch_dim:  # Return a batch of 2 (same data twice)
        entry = {k: (torch.stack((v, v), dim=0) if k != 'view_matrix' else torch.cat((v, v), dim=0))
                 for k, v in entry.items()}
    return entry


@pytest.fixture(params=itertools.product(_YPR_DATA_IDX, _IS_BATCH_DIM))
def rotation_data(request):
    data = list()
    rotation_amount = math.pi / 4

    def _add_entry(matrix_entries):
        entry = {k: torch.from_numpy(np.array(v)).unsqueeze(0) for k,v in matrix_entries.items()}
        data.append(entry)

    # View matrices here are row major, transposition required
    _add_entry({'view_matrix': [
                    [-0.40824828,  0.81649655,  -0.40824828, -0.],
                    [-0.70710671,  0.,          0.70710671,  -1.41421366],
                    [0.57735026,   0.57735026,  0.57735026,  -8.66025352],
                    [0.,            0.,         0.,          1.]
                ],
               'rotated_yaw': [
                   [-0.6967487335205078, 0.16918081045150757, -0.6967487335205078, 6.12641716003418],
                   [-0.70703125, 0.0, 0.70703125, -1.4140625],
                   [0.11946237087249756, 0.9853919148445129, 0.11946237087249756, -6.12641716003418],
                   [0.0, 0.0, 0.0, 1.0]
               ],
                'rotated_pitch': [
                    [-0.408203125, 0.81640625, -0.408203125, -0.0],
                    [-0.09184104204177856, 0.4081055521965027, 0.9080521464347839, -7.126310348510742],
                    [0.9080521464347839, 0.4081055521965027, -0.09184104204177856, -5.126523971557617],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                'rotated_roll': [
                    [-0.7885897755622864, 0.5772863626480103, 0.21130341291427612, -0.9998931884765625],
                    [-0.21130341291427612, -0.5772863626480103, 0.7885897755622864, -0.9998931884765625],
                    [0.5771484375, 0.5771484375, 0.5771484375, -8.6640625],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                'rotated_yawpitch': [
                    [-0.6967487335205078, 0.16918081045150757, -0.6967487335205078, 6.12641716003418],
                    [-0.41547393798828125, 0.6967772841453552, 0.5844192504882812,-5.3319244384765625],
                    [0.5844192504882812, 0.6967772841453552, -0.41547393798828125,-3.3321380615234375],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                'rotated_yawroll': [
                   [-0.9926223754882812, 0.1196288987994194, 0.007270842790603638, 3.3321380615234375],
                   [-0.007270842790603638, -0.1196288987994194, 0.9926223754882812, -5.3319244384765625],
                   [0.11946237087249756, 0.9853919148445129, 0.11946237087249756, -6.12641716003418],
                   [0.0, 0.0, 0.0, 1.0]
                ],
                'rotated_pitchroll': [
                    [-0.3535845875740051, 0.8658605813980103, 0.3534466624259949, -5.0390625],
                    [0.22370176017284393, -0.28871217370033264, 0.9307330250740051, -5.0390625],
                    [0.9080521464347839, 0.4081055521965027, -0.09184104204177856, -5.126523971557617],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                'rotated_ypr': [
                    [-0.7864601612091064, 0.6123248338699341, -0.07942894101142883, 0.5617914199829102],
                    [0.1988913118839264, 0.3730670213699341, 0.905922532081604, -8.10227108001709],
                    [0.5844192504882812, 0.6967772841453552, -0.41547393798828125, -3.3321380615234375],
                    [0.0, 0.0, 0.0, 1.0]
                ]})
    _add_entry({'view_matrix': [
                     [-1.0, 0.0, 0.0, -0.0],
                     [0.0, 1.0, -0.0, -0.0],
                     [-0.0, -0.0, -1.0, -1.0],
                     [0.0, 0.0, 0.0, 1.0]
                ],
                'rotated_yaw': [
                    [-0.7071067690849304, 0.0, 0.7071067690849304, 0.0],
                    [0.0, 1.0, -0.0, 0.0],
                    [-0.7071067690849304, 0.0, -0.7071067690849304, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ],
                'rotated_pitch': [
                    [-1.0, 0.0, -0.0, 0.0],
                    [0.0, 0.7071067690849304, -0.7071067690849304, 0.0],
                    [0.0, -0.7071067690849304, -0.7071067690849304, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ],
                'rotated_roll': [
                    [-0.7071067690849304, 0.7071067690849304, -0.0, 0.0],
                    [0.7071067690849304, 0.7071067690849304, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ],
                'rotated_yawpitch': [
                    [-0.7071067690849304, 0.0, 0.7071067690849304, 0.0],
                    [-0.4999999701976776, 0.7071067690849304, -0.4999999701976776, 0.0],
                    [-0.4999999701976776, -0.7071067690849304, -0.4999999701976776, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ],
                'rotated_yawroll': [
                    [-0.4999999701976776, 0.7071067690849304, 0.4999999701976776, 0.0],
                    [0.4999999701976776, 0.7071067690849304, -0.4999999701976776, 0.0],
                    [-0.7071067690849304, 0.0, -0.7071067690849304, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ],
                'rotated_pitchroll': [
                    [-0.7071067690849304, 0.4999999701976776, -0.4999999701976776, 0.0],
                    [0.7071067690849304, 0.4999999701976776, -0.4999999701976776, 0.0],
                    [0.0, -0.7071067690849304, -0.7071067690849304, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ],
                'rotated_ypr': [
                    [-0.853553295135498, 0.4999999701976776, 0.1464466154575348, 0.0],
                    [0.1464466154575348, 0.4999999701976776, -0.853553295135498, 0.0],
                    [-0.4999999701976776, -0.7071067690849304, -0.4999999701976776, 0.0],
                    [-0.0, -0.0, -1.0, 1.0]
                ]})
    data_idx = request.param[0]
    is_batch_dim = request.param[1]

    entry = data[data_idx]
    if is_batch_dim:  # Return a batch of 2 (same data twice)
        entry = {k: torch.cat((v, v), dim=0) for k, v in entry.items()}
    entry['rotation_amount'] = rotation_amount
    return entry


@pytest.fixture(params=_YPR_ANGLES_IDX)
def yaw_pitch_roll_data(request):
    data_idx = request.param

    tested_vals = (math.pi, -math.pi, 2 * math.pi, 0.0, 0.288 * math.pi)
    combos = list(itertools.product(tested_vals, tested_vals, tested_vals))
    vals = combos[data_idx]

    input_type = _YPR_DATA_IDX.stop // 3
    if data_idx // input_type == 0:
        pass    # input is float
    elif data_idx // input_type == 1:    # input is tensors
        vals = (torch.Tensor([vals[0]]), torch.Tensor([vals[1]]), torch.Tensor([vals[2]]))
    else:   # input is batched tensors
        vals = (torch.Tensor([[vals[0]]]), torch.Tensor([[vals[1]]]), torch.Tensor([[vals[2]]]))

    return dict(yaw=vals[0], pitch=vals[1], roll=vals[2])


def assert_transformed_tensors(expected_tensor, gt_tensor):
    dtype = expected_tensor.dtype
    if dtype == torch.half:
        assert torch.allclose(expected_tensor, gt_tensor, rtol=1e-3, atol=1e-2)
    elif dtype == torch.float32:
        assert torch.allclose(expected_tensor, gt_tensor, rtol=1e-3, atol=1e-3)
    else:
        assert torch.allclose(expected_tensor, gt_tensor)


def assert_view_matrix(expected_view_mat, gt_view_mat):
    assert_transformed_tensors(expected_view_mat, gt_view_mat)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES + [(None, torch.float), ('cuda', None)])
@pytest.mark.parametrize('requires_grad', (True, False))
class TestCameraExtrinsicsConstructors:

    def test_from_lookat(self, device, dtype, requires_grad, lookat_data):
        at, eye, up, view_matrix = lookat_data['at'], lookat_data['eye'], lookat_data['up'], lookat_data['view_matrix']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, dtype, device, requires_grad)
        view_matrix = view_matrix.to(device).to(dtype)

        assert_view_matrix(extrinsics.view_matrix(), view_matrix)
        assert (extrinsics.device.type == device) or device is None
        assert (extrinsics.dtype == dtype) or dtype is None
        assert extrinsics.requires_grad == requires_grad

    def test_from_camera_pose(self, device, dtype, requires_grad, cam_pos_data):
        cam_pos, cam_dir, view_matrix = cam_pos_data['cam_pos'], cam_pos_data['cam_dir'], cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir, dtype, device, requires_grad)
        view_matrix = view_matrix.to(device).to(dtype)

        assert_view_matrix(extrinsics.view_matrix(), view_matrix)
        assert (extrinsics.device.type == device) or device is None
        assert (extrinsics.dtype == dtype) or dtype is None
        assert extrinsics.requires_grad == requires_grad

    def test_from_view_matrix(self, device, dtype, requires_grad, cam_pos_data):
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype, device, requires_grad)
        target_view_matrix = view_matrix.to(device).to(dtype)

        assert_view_matrix(extrinsics.view_matrix(), target_view_matrix)
        assert (extrinsics.device.type == device) or device is None
        assert (extrinsics.dtype == dtype) or dtype is None
        assert extrinsics.requires_grad == requires_grad


@pytest.mark.parametrize('device', ALL_DEVICES)
@pytest.mark.parametrize('requires_grad', (True, False))
class TestCameraExtrinsicsBackends:

    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_build_with_backend(self, device, requires_grad, sample_lookat_data, backend):
        dtype = torch.float32
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        view_matrix = sample_lookat_data['view_matrix']
        view_matrix = view_matrix.to(device).to(dtype)

        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype,
                                                  requires_grad=requires_grad, backend=backend)
        assert torch.allclose(extrinsics.view_matrix(), view_matrix, rtol=1e-3, atol=1e-3)
        assert (extrinsics.device.type == device) or device is None
        assert (extrinsics.dtype == dtype) or dtype is None
        assert extrinsics.requires_grad == requires_grad

    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_switch_backend(self, device, requires_grad, sample_lookat_data, backend):
        dtype = torch.float32
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        view_matrix = sample_lookat_data['view_matrix']
        view_matrix = view_matrix.to(device).to(dtype)

        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype,
                                                  requires_grad=requires_grad, backend=backend)
        for backend_name in CameraExtrinsics.available_backends():
            extrinsics.switch_backend(backend_name)
            assert torch.allclose(extrinsics.view_matrix(), view_matrix, rtol=1e-3, atol=1e-3)
            assert (extrinsics.device.type == device) or device is None
            assert (extrinsics.dtype == dtype) or dtype is None
            assert extrinsics.requires_grad == requires_grad

    def test_register_backend(self, device, requires_grad, sample_lookat_data):
        from kaolin.render.camera.extrinsics import register_backend
        from kaolin.render.camera.extrinsics_backends import ExtrinsicsRep, ExtrinsicsParamsDefEnum

        @register_backend
        class _TestRep(ExtrinsicsRep):
            def __init__(self, params: torch.Tensor, dtype: torch.dtype = None, device: torch.device = None,
                         requires_grad: bool = False):
                super().__init__(params, dtype, device, requires_grad)

            def convert_to_mat(self):
                return self.params[:, 0, :].reshape(-1, 4, 4)   # Select one copy

            @classmethod
            def convert_from_mat(cls, mat: torch.Tensor):
                return torch.stack((mat.reshape(-1, 16), mat.reshape(-1, 16)), dim=1)  # Redundant representation

            def param_idx(self, param_idx: ExtrinsicsParamsDefEnum):
                if param_idx == ExtrinsicsParamsDefEnum.R:
                    return [0, 1, 2, 4, 5, 6, 8, 9, 10] + [16, 17, 18, 20, 21, 22, 24, 25, 26]

            @classmethod
            def backend_name(cls) -> str:
                return "test_rep"

        dtype = torch.float32
        device = 'cuda'
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        view_matrix = sample_lookat_data['view_matrix']
        view_matrix = view_matrix.to(device).to(dtype)

        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, dtype=dtype, device=device,
                                                  requires_grad=requires_grad, backend="test_rep")
        assert torch.allclose(extrinsics.view_matrix(), view_matrix, rtol=1e-3, atol=1e-3)
        assert (extrinsics.device.type == device) or device is None
        assert extrinsics.requires_grad == requires_grad
        assert extrinsics.backend_name == "test_rep"


@pytest.mark.parametrize('device', ALL_DEVICES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsDevice:
    def test_backend_to_device(self, device, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, backend=backend)
        extrinsics = extrinsics.to(device)
        assert extrinsics.to(device).device.type == device
        assert extrinsics.to(device).view_matrix().device.type == device


@pytest.mark.parametrize('dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsDtype:
    def test_backend_to_dtype(self, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, backend=backend)
        extrinsics = extrinsics.to(dtype)
        assert extrinsics.to(dtype).dtype == dtype
        assert extrinsics.to(dtype).view_matrix().dtype == dtype


@pytest.mark.parametrize('requires_grad', (True, False))
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsDtype:
    def test_backend_requires_grad(self, requires_grad, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, requires_grad=requires_grad, backend=backend)
        extrinsics.requires_grad = requires_grad
        assert extrinsics.requires_grad == requires_grad
        assert extrinsics.view_matrix().requires_grad == requires_grad


@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsLen:
    def test_backend_len(self, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        batch_size = sample_lookat_data['view_matrix'].shape[0]
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, backend=backend)
        assert len(extrinsics) == batch_size
        assert extrinsics.view_matrix().shape[0] == batch_size


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsMatrixComponents:
    def test_view_matrix_get_R(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_matrix = sample_lookat_data['view_matrix']
        gt_R = view_matrix[:, :3, :3].to(device, dtype)
        assert torch.allclose(extrinsics.R, gt_R, rtol=1e-3, atol=1e-3)

    def test_view_matrix_R_orthonormal(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        batch_size = sample_lookat_data['view_matrix'].shape[0]
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        R_mul_RT = extrinsics.R @ extrinsics.R.transpose(1, 2)
        eye_3 = torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3)
        assert torch.allclose(R_mul_RT, eye_3, rtol=1e-3, atol=1e-3)

    def test_view_matrix_set_R(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        batch_size = sample_lookat_data['view_matrix'].shape[0]
        gt_R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        extrinsics.R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        assert torch.allclose(extrinsics.R, gt_R, rtol=1e-4, atol=1e-3)

    def test_view_matrix_get_t(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_matrix = sample_lookat_data['view_matrix']
        gt_t = view_matrix[:, :3, -1:].to(device, dtype)
        assert torch.allclose(extrinsics.t, gt_t, rtol=1e-3, atol=1e-3)

    def test_view_matrix_set_t(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_matrix = sample_lookat_data['view_matrix']
        gt_t = view_matrix[:, :3, -1:].to(device, dtype) * 100
        extrinsics.t = view_matrix[:, :3, -1:].to(device, dtype) * 100
        assert torch.allclose(extrinsics.t, gt_t, rtol=1e-3, atol=1e-3)

    def test_view_matrix_update(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        batch_size = sample_lookat_data['view_matrix'].shape[0]
        gt_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        updated_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        extrinsics.update(updated_mat)
        assert torch.allclose(extrinsics.view_matrix(), gt_mat, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('device', ALL_DEVICES)
@pytest.mark.parametrize('dtype', (torch.float, torch.double))
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsInvViewMatrix:
    def test_inv_view_with_torch(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        assert torch.allclose(extrinsics.inv_view_matrix(), torch.inverse(extrinsics.view_matrix()),
                              rtol=1e-3, atol=1e-3)

    def test_inv_view_mul_to_identity(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_matrix = sample_lookat_data['view_matrix'].to(device, dtype)
        batch_size = view_matrix.shape[0]

        identity = extrinsics.inv_view_matrix() @ extrinsics.view_matrix()
        expected_identity = torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4)
        assert torch.allclose(identity, expected_identity, rtol=1e-4, atol=1e-3)

    def test_inv_twice(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_matrix = sample_lookat_data['view_matrix'].to(device, dtype)

        view_mat_reinverted = torch.inverse(extrinsics.inv_view_matrix())
        assert torch.allclose(view_mat_reinverted, view_matrix, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsTransform:
    def test_extrinsics_transform(self, device, dtype, backend, sample_lookat_data, vectors_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_mat = sample_lookat_data['view_matrix'].to(device, dtype)
        vectors_data = vectors_data.to(device, dtype)

        transformed_vecs = extrinsics.transform(vectors_data)
        assert transformed_vecs.shape == (view_mat.shape[0], vectors_data.shape[0], 3)

        homogeneous_vecs = torch.cat((vectors_data, vectors_data.new_ones(vectors_data.shape[0], 1)), dim=1)
        for cam_idx, single_view_mat in enumerate(view_mat):
            for vec_idx, single_vec in enumerate(homogeneous_vecs):
                gt_vec = single_view_mat @ single_vec
                gt_vec = gt_vec[:3]
                assert_transformed_tensors(transformed_vecs[cam_idx, vec_idx], gt_vec)

    def test_inv_transform_rays(self, device, dtype, backend, sample_lookat_data, ray_direction_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype, backend=backend)
        view_mat = sample_lookat_data['view_matrix'].to(device, dtype)

        world_ray_orig = ray_direction_data['ray_orig'].to(device, dtype)
        world_ray_dir = ray_direction_data['ray_dir'].to(device, dtype)
        num_cams = view_mat.shape[0]
        num_vecs = world_ray_orig.shape[0]

        cam_ray_orig = extrinsics.transform(world_ray_orig).squeeze(0)
        cam_vec_end = extrinsics.transform(world_ray_orig + world_ray_dir).squeeze(0)

        # Rigid transformations retain unit length..
        expected_world_ray_orig, expected_world_ray_dir = \
            extrinsics.inv_transform_rays(ray_orig=cam_ray_orig, ray_dir=cam_vec_end-cam_ray_orig)

        assert expected_world_ray_orig.shape == expected_world_ray_dir.shape == (num_cams, num_vecs, 3)
        assert torch.allclose(expected_world_ray_orig, world_ray_orig, rtol=1e-2, atol=1e-2)
        assert torch.allclose(expected_world_ray_dir, world_ray_dir, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestCameraExtrinsicsParams:

    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    @pytest.mark.parametrize('requires_grad', (True, False))
    def test_params(self, device, dtype, backend, requires_grad, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype,
                                                  requires_grad=requires_grad, backend=backend)
        view_mat = sample_lookat_data['view_matrix'].to(device, dtype)
        num_cams = view_mat.shape[0]
        params = extrinsics.parameters()
        assert params.shape[0] == num_cams and params.ndim == 2

    @pytest.mark.parametrize('backend', [x for x in CameraExtrinsics.available_backends() if x != 'matrix_se3'])
    def test_differentiable_params(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype,
                                                  requires_grad=True, backend=backend)
        num_cams = sample_lookat_data['view_matrix'].shape[0]
        original_matrix = extrinsics.view_matrix()
        target_matrix = torch.eye(4).to(device, dtype)

        optimizer = torch.optim.SGD([extrinsics.parameters()], lr=0.001, momentum=0.9)
        criterion = torch.nn.functional.mse_loss
        for i in range(50):
            optimizer.zero_grad()
            view_matrix = extrinsics.view_matrix()
            loss = criterion(view_matrix, target_matrix)
            loss.backward()
            optimizer.step()
            assert loss > 0

        # Sanity check: Ensure criteria is reduced
        assert criterion(original_matrix, target_matrix) > criterion(extrinsics.view_matrix(), target_matrix)

        # Sanity check: Ensure transformation stays rigid (rotation axes must be orthonormal)
        if not (device == 'cuda' and dtype == torch.half): # "lu_cuda" not implemented for 'Half'
            assert (torch.isclose(torch.det(extrinsics.R), torch.ones(num_cams, device=device, dtype=dtype))).all()

    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_non_differentiable_params(self, device, dtype, backend, sample_lookat_data):
        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype,
                                                  requires_grad=False, backend=backend)
        original_matrix = extrinsics.view_matrix()
        target_matrix = torch.eye(4).to(device, dtype)
        optimizer = torch.optim.SGD([extrinsics.parameters()], lr=0.001, momentum=0.9)
        criterion = torch.nn.functional.mse_loss
        for i in range(5):
            optimizer.zero_grad()
            view_matrix = extrinsics.view_matrix()
            loss = criterion(view_matrix, target_matrix)
            with pytest.raises(RuntimeError,
                               match="element 0 of tensors does not require grad and does not have a grad_fn"):
                loss.backward()
                optimizer.step()

        # Sanity check: Ensure criteria is not reduced
        assert (original_matrix == extrinsics.view_matrix()).all()

    @pytest.mark.parametrize('backend', [x for x in CameraExtrinsics.available_backends() if x != 'matrix_se3'])
    def test_differentiable_transform(self, device, dtype, backend, sample_lookat_data, vectors_data):
        vectors_data = vectors_data.to(device, dtype)
        expected_view_matrix = sample_lookat_data['view_matrix'].to(device, dtype)
        num_cams = expected_view_matrix.shape[0]

        at, eye, up = sample_lookat_data['at'], sample_lookat_data['eye'], sample_lookat_data['up']
        extrinsics = CameraExtrinsics.from_lookat(eye, at, up, device=device, dtype=dtype,
                                                  requires_grad=False, backend=backend)
        target_vecs = extrinsics.transform(vectors_data)

        torch.manual_seed(1337)
        noise = torch.randn_like(eye)
        diff_extrinsics = CameraExtrinsics.from_lookat(eye + noise, at + noise,
                                                       up + noise, device=device, dtype=dtype,
                                                       requires_grad=True, backend=backend)
        initial_vecs = diff_extrinsics.transform(vectors_data)
        original_matrix = diff_extrinsics.view_matrix()

        optimizer = torch.optim.SGD([diff_extrinsics.parameters()], lr=0.001, momentum=0.9)
        criterion = torch.nn.functional.mse_loss
        for i in range(50):
            optimizer.zero_grad()
            transformed_vecs = diff_extrinsics.transform(vectors_data)
            loss = criterion(transformed_vecs, target_vecs)
            loss.backward()
            optimizer.step()
            assert loss > 0

        # Sanity check: Ensure criteria is reduced
        assert criterion(initial_vecs, target_vecs) > criterion(transformed_vecs, target_vecs)

        # Sanity check: Ensure learned view matrix is closer to target view than initial, within SE(3)
        transformed_vecs = diff_extrinsics.transform(vectors_data)
        assert criterion(original_matrix, expected_view_matrix) > \
               criterion(diff_extrinsics.view_matrix(), expected_view_matrix)

        # Sanity check: Ensure transformation stays rigid (rotation axes must be orthonormal)
        if not (device == 'cuda' and dtype == torch.half):  # "lu_cuda" not implemented for 'Half'
            assert (torch.isclose(torch.det(extrinsics.R), torch.ones(num_cams, device=device, dtype=dtype))).all()


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsTranslate:

    def test_translate(self, device, dtype, backend, cam_pos_data):
        cam_pos, cam_dir, view_matrix = cam_pos_data['cam_pos'], cam_pos_data['cam_dir'], cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir, dtype=dtype, device=device, backend=backend)

        view_matrix = cam_pos_data['view_matrix'].to(device, dtype)
        translate_amnt = view_matrix.new_tensor([0.1, 1.0, 10.0])
        cam_pos = torch.tensor(cam_pos, device=device, dtype=dtype)
        shifted_pos = cam_pos.squeeze(-1) + translate_amnt.to(cam_pos.device)
        translated_extrinsics = CameraExtrinsics.from_camera_pose(shifted_pos, cam_dir,
                                                                  dtype=dtype, device=device, backend=backend)

        extrinsics.translate(translate_amnt)
        assert torch.allclose(extrinsics.t, translated_extrinsics.t, atol=1e-3, rtol=1e-3)


class TestCameraExtrinsicsRotate:

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_yaw(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(yaw=rotate_amount)
        expected_view_mat = rotation_data['rotated_yaw'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_pitch(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(pitch=rotate_amount)
        expected_view_mat = rotation_data['rotated_pitch'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_roll(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(roll=rotate_amount)
        expected_view_mat = rotation_data['rotated_roll'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_yaw_pitch(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(yaw=rotate_amount, pitch=rotate_amount)
        expected_view_mat = rotation_data['rotated_yawpitch'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_yaw_roll(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(yaw=rotate_amount, roll=rotate_amount)
        expected_view_mat = rotation_data['rotated_yawroll'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_pitch_roll(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(pitch=rotate_amount, roll=rotate_amount)
        expected_view_mat = rotation_data['rotated_pitchroll'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_yaw_pitch_roll(self, device, dtype, backend, rotation_data):
        view_matrix = rotation_data['view_matrix'].to(device, dtype)
        rotate_amount = rotation_data['rotation_amount']
        extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device, backend=backend)
        extrinsics.rotate(yaw=rotate_amount, pitch=rotate_amount, roll=rotate_amount)
        expected_view_mat = rotation_data['rotated_ypr'].to(device, dtype)
        torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_min_max_angles_float(self, device, dtype, backend):
        for angle in (math.pi, -math.pi, 2 * math.pi, 0.0):
            view_matrix = torch.eye(4, device=device, dtype=dtype)
            extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device)
            extrinsics.rotate(yaw=angle)
            a = torch.tensor((angle))
            expected_view_mat = torch.tensor([
                [torch.cos(a), 0, -torch.sin(a), 0],
                [0, 1, 0, 0],
                [torch.sin(a), 0, torch.cos(a), 0],
                [0, 0, 0, 1]
            ], dtype=dtype, device=device)
            torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_min_max_angles_float_batched_cam(self, device, dtype, backend):
        for angle in (math.pi, -math.pi, 2 * math.pi, 0.0):
            view_matrix = torch.eye(4, device=device, dtype=dtype).repeat(2, 1, 1)
            extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device)
            extrinsics.rotate(yaw=angle)
            a = torch.tensor((angle))
            expected_view_mat = torch.tensor([
                [torch.cos(a), 0, -torch.sin(a), 0],
                [0, 1, 0, 0],
                [torch.sin(a), 0, torch.cos(a), 0],
                [0, 0, 0, 1]
            ], dtype=dtype, device=device).repeat(2, 1, 1)
            torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_min_max_angles_tensor(self, device, dtype, backend):
        for angle in (math.pi, -math.pi, 2 * math.pi, 0.0):
            view_matrix = torch.eye(4, device=device, dtype=dtype)
            extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device)
            extrinsics.rotate(yaw=torch.tensor((angle), dtype=dtype, device=device))
            a = torch.tensor((angle))
            expected_view_mat = torch.tensor([
                [torch.cos(a), 0, -torch.sin(a), 0],
                [0, 1, 0, 0],
                [torch.sin(a), 0, torch.cos(a), 0],
                [0, 0, 0, 1]
            ], device=device, dtype=dtype)
            torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_min_max_angles_batched_tensor(self, device, dtype, backend):
        eps = math.pi * 0.1
        for angle in (math.pi, -math.pi, 2 * math.pi, 0.0):
            view_matrix = torch.eye(4, device=device, dtype=dtype).repeat(2, 1, 1)
            extrinsics = CameraExtrinsics.from_view_matrix(view_matrix, dtype=dtype, device=device)
            extrinsics.rotate(yaw=torch.tensor((angle, angle + eps), dtype=dtype, device=device))
            a = torch.tensor((angle))
            mat1 = torch.tensor([
                [torch.cos(a), 0, -torch.sin(a), 0],
                [0, 1, 0, 0],
                [torch.sin(a), 0, torch.cos(a), 0],
                [0, 0, 0, 1]
            ], dtype=dtype, device=device)
            b = torch.tensor((angle + eps))
            mat2 = torch.tensor([
                [torch.cos(b), 0, -torch.sin(b), 0],
                [0, 1, 0, 0],
                [torch.sin(b), 0, torch.cos(b), 0],
                [0, 0, 0, 1]
            ], dtype=dtype, device=device)
            expected_view_mat = torch.stack([mat1, mat2], dim=0)

            torch.allclose(extrinsics.view_matrix(), expected_view_mat, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsMoveCam:

    def test_move_right(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, dtype=dtype, device=device, backend=backend)
        amount = 1.0
        extrinsics.move_right(amount)

        expected_mat = cam_pos_data['view_matrix'].to(device, dtype)
        axis_idx = 0
        expected_mat[..., axis_idx, 3] -= amount
        assert_view_matrix(extrinsics.view_matrix(), expected_mat)

    def test_move_up(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, dtype=dtype, device=device, backend=backend)
        amount = 1.0
        extrinsics.move_up(amount)

        expected_mat = cam_pos_data['view_matrix'].to(device, dtype)
        axis_idx = 1
        expected_mat[..., axis_idx, 3] -= amount
        assert_view_matrix(extrinsics.view_matrix(), expected_mat)

    def test_move_forward(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, dtype=dtype, device=device, backend=backend)
        amount = 1.0
        extrinsics.move_forward(amount)

        expected_mat = cam_pos_data['view_matrix'].to(device, dtype)
        axis_idx = 2
        expected_mat[..., axis_idx, 3] -= amount
        assert_view_matrix(extrinsics.view_matrix(), expected_mat)

@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
@pytest.mark.parametrize('requires_grad', (True, False))
class TestCameraExtrinsicsBackendProperties:

    def test_backend_cls(self, backend, requires_grad, cam_pos_data):
        from kaolin.render.camera.extrinsics_backends import _REGISTERED_BACKENDS
        cam_pos, cam_dir, view_matrix = cam_pos_data['cam_pos'], cam_pos_data['cam_dir'], cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir,backend=backend, requires_grad=requires_grad)
        supported_backends = ("matrix_se3", "matrix_6dof_rotation")
        assert extrinsics.backend_name in supported_backends
        assert isinstance(extrinsics._backend, _REGISTERED_BACKENDS[extrinsics.backend_name])

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
@pytest.mark.parametrize('requires_grad', (True, False))
class TestCameraExtrinsicsGetItem:

    def test_get_item(self, device, dtype, backend, requires_grad, cam_pos_data):
        cam_pos, cam_dir, view_matrix = cam_pos_data['cam_pos'], cam_pos_data['cam_dir'], cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir, device=device, dtype=dtype, backend=backend,
                                                       requires_grad=requires_grad)
        single_extrinsics = extrinsics[0]

        assert single_extrinsics.parameters().ndim == 2
        assert len(single_extrinsics) == 1
        assert single_extrinsics.dtype == extrinsics.dtype
        assert single_extrinsics.device == extrinsics.device
        assert single_extrinsics.requires_grad == extrinsics.requires_grad


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('requires_grad', (True, False))
class TestCameraExtrinsicsAllClose:

    @pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
    def test_all_close(self, device, dtype, backend, requires_grad, cam_pos_data):
        cam_pos, cam_dir, view_matrix = cam_pos_data['cam_pos'], cam_pos_data['cam_dir'], cam_pos_data['view_matrix']
        eps = 1e-2
        extrinsics1 = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir, device=device, dtype=dtype, backend=backend,
                                                        requires_grad=requires_grad)
        extrinsics2 = CameraExtrinsics.from_camera_pose(cam_pos + eps, cam_dir, device=device, dtype=dtype,
                                                        backend=backend, requires_grad=requires_grad)
        assert torch.allclose(extrinsics1, extrinsics2, atol=1e-1)
        assert not torch.allclose(extrinsics1, extrinsics2, atol=1e-3, rtol=1e-3)

    def test_all_close_backend(self, device, dtype, requires_grad, cam_pos_data):
        cam_pos, cam_dir, view_matrix = cam_pos_data['cam_pos'], cam_pos_data['cam_dir'], cam_pos_data['view_matrix']
        extrinsics1 = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir, device=device, dtype=dtype,
                                                        backend="matrix_se3", requires_grad=requires_grad)
        extrinsics2 = CameraExtrinsics.from_camera_pose(cam_pos, cam_dir, device=device, dtype=dtype,
                                                        backend="matrix_6dof_rotation", requires_grad=requires_grad)
        assert not torch.allclose(extrinsics1, extrinsics2, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('backend', CameraExtrinsics.available_backends())
class TestCameraExtrinsicsCamPosDir:

    def test_cam_pos(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, device=device, dtype=dtype, backend=backend)
        num_cams = view_matrix.shape[0]
        expected = torch.tensor(cam_pos, device=device, dtype=dtype)
        expected = expected.reshape(num_cams, 3, 1)
        extrinsics_result = extrinsics.cam_pos()
        assert torch.allclose(extrinsics_result, expected, rtol=1e-3, atol=1e-2)
        assert extrinsics_result.shape == (num_cams, 3, 1)

    def test_cam_right(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, device=device, dtype=dtype, backend=backend)
        num_cams = view_matrix.shape[0]
        cam_dir = torch.tensor(cam_dir, device=device, dtype=dtype)
        if cam_dir.ndim == 2:
            cam_dir = cam_dir.unsqueeze(0)

        expected = cam_dir[:, :, 0]
        expected = expected.reshape(num_cams, 3, 1)
        extrinsics_result = extrinsics.cam_right()
        assert torch.allclose(extrinsics_result, expected, rtol=1e-3, atol=1e-3)
        assert extrinsics_result.shape == (num_cams, 3, 1)

    def test_cam_up(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, device=device, dtype=dtype, backend=backend)
        num_cams = view_matrix.shape[0]
        cam_dir = torch.tensor(cam_dir, device=device, dtype=dtype)
        if cam_dir.ndim == 2:
            cam_dir = cam_dir.unsqueeze(0)
        expected = cam_dir[:, :, 1]
        expected = expected.reshape(num_cams, 3, 1)
        extrinsics_result = extrinsics.cam_up()
        assert torch.allclose(extrinsics_result, expected, rtol=1e-3, atol=1e-3)
        assert extrinsics_result.shape == (num_cams, 3, 1)

    def test_cam_forward(self, device, dtype, backend, cam_pos_data):
        cam_pos = cam_pos_data['cam_pos']
        cam_dir = cam_pos_data['cam_dir']
        view_matrix = cam_pos_data['view_matrix']
        extrinsics = CameraExtrinsics.from_camera_pose(
            cam_pos, cam_dir, device=device, dtype=dtype, backend=backend)
        num_cams = view_matrix.shape[0]
        cam_dir = torch.tensor(cam_dir, device=device, dtype=dtype)
        if cam_dir.ndim == 2:
            cam_dir = cam_dir.unsqueeze(0)
        expected = cam_dir[:, :, 2]
        expected = expected.reshape(num_cams, 3, 1)
        extrinsics_result = extrinsics.cam_forward()
        assert torch.allclose(extrinsics_result, expected, rtol=1e-3, atol=1e-3)
        assert extrinsics_result.shape == (num_cams, 3, 1)

