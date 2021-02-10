# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import pytest
import random

import numpy as np
import torch
from PIL import Image

from kaolin.io import render
from kaolin.render.camera import generate_perspective_projection
from kaolin.utils.testing import FLOAT_TYPES

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir, os.pardir, 'samples', 'synthetic')

class TestImportView:
    @pytest.fixture(autouse=True)
    def expected_rgb(self):
        path = os.path.join(SAMPLE_DIR, '0_rgb.png')
        return torch.from_numpy(
            np.array(Image.open(path))
        )[:, :, :3].float() / 255.

    @pytest.fixture(autouse=True)
    def expected_depth_linear(self):
        path = os.path.join(SAMPLE_DIR, '0_depth_linear.npy')
        return torch.from_numpy(np.load(path))

    @pytest.fixture(autouse=True)
    def expected_semantic(self):
        path = os.path.join(SAMPLE_DIR, '0_semantic.npy')
        return torch.from_numpy(np.load(path))

    @pytest.fixture(autouse=True)
    def expected_instance(self):
        path = os.path.join(SAMPLE_DIR, '0_instance.npy')
        return torch.from_numpy(np.load(path))

    @pytest.fixture(autouse=True)
    def expected_normals(self):
        path = os.path.join(SAMPLE_DIR, '0_normals.png')
        return torch.from_numpy(
            np.array(Image.open(path))
        )[:, :, :3].float() / 255.

    @pytest.fixture(autouse=True)
    def expected_json(self):
        path = os.path.join(SAMPLE_DIR, '0_metadata.json')
        with open(path, 'r') as f:
            fjson = json.load(f)
        return fjson

    @pytest.fixture(autouse=True)
    def expected_metadata(self, expected_json):
        asset_transforms = torch.FloatTensor(expected_json['asset_transforms'][0][1])
        cam_transform = torch.FloatTensor(expected_json['camera_properties']['tf_mat'])
        aspect_ratio = (expected_json['camera_properties']['resolution']['width'] /
                        expected_json['camera_properties']['resolution']['height'])
        focal_length = expected_json['camera_properties']['focal_length']
        horizontal_aperture = expected_json['camera_properties']['horizontal_aperture']
        fov = 2 * math.atan(horizontal_aperture / (2 * focal_length))
        return {
            'cam_transform': cam_transform[:, :3],
            'asset_transforms': asset_transforms,
            'cam_proj': generate_perspective_projection(fov, aspect_ratio),
            'clipping_range': expected_json['camera_properties']['clipping_range']
        }

    @pytest.fixture(autouse=True)
    def expected_bbox_2d_tight(self, expected_json):
        return expected_json['bbox_2d_tight']

    @pytest.fixture(autouse=True)
    def expected_bbox_2d_loose(self, expected_json):
        return expected_json['bbox_2d_loose']

    @pytest.mark.parametrize('with_rgb', [True, False])
    @pytest.mark.parametrize('with_depth_linear', [True, False])
    @pytest.mark.parametrize('with_semantic', [True, False])
    @pytest.mark.parametrize('with_instance', [True, False])
    @pytest.mark.parametrize('with_normals', [True, False])
    @pytest.mark.parametrize('with_bbox_2d_tight', [True, False])
    @pytest.mark.parametrize('with_bbox_2d_loose', [True, False])
    def test_import_synthetic_view(self, expected_rgb, expected_depth_linear,
                                   expected_semantic, expected_instance,
                                   expected_normals, expected_bbox_2d_tight,
                                   expected_bbox_2d_loose, expected_metadata,
                                   with_rgb, with_depth_linear, with_semantic,
                                   with_instance, with_normals, with_bbox_2d_tight,
                                   with_bbox_2d_loose):
        output = render.import_synthetic_view(SAMPLE_DIR, 0,
                                              rgb=with_rgb,
                                              depth_linear=with_depth_linear,
                                              semantic=with_semantic,
                                              instance=with_instance,
                                              normals=with_normals,
                                              bbox_2d_tight=with_bbox_2d_tight,
                                              bbox_2d_loose=with_bbox_2d_loose)

        if with_rgb:
            assert torch.equal(output['rgb'], expected_rgb)
        else:
            assert 'rgb' not in output

        if with_depth_linear:
            assert torch.equal(output['depth_linear'], expected_depth_linear)
        else:
            assert 'depth_linear' not in output

        if with_semantic:
            assert torch.equal(output['semantic'], expected_semantic)
        else:
            assert 'semantic' not in output

        if with_instance:
            assert torch.equal(output['instance'], expected_instance)
        else:
            assert 'instance' not in output

        if with_normals:
            assert torch.equal(output['normals'], expected_normals)
        else:
            assert 'normals' not in output

        if with_bbox_2d_tight:
            assert output['bbox_2d_tight'] == expected_bbox_2d_tight
        else:
            assert 'bbox_2d_tight' not in output

        if with_bbox_2d_loose:
            assert output['bbox_2d_loose'] == expected_bbox_2d_loose
        else:
            assert 'bbox_2d_loose' not in output

        assert expected_metadata.keys() == output['metadata'].keys()
        assert torch.equal(expected_metadata['cam_transform'],
                           output['metadata']['cam_transform'])
        assert torch.equal(expected_metadata['cam_proj'],
                           output['metadata']['cam_proj'])
        assert (expected_metadata['clipping_range'] ==
                output['metadata']['clipping_range'])
