# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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

import json
import math
import os

import torch
import numpy as np
from PIL import Image
from ..render.camera import generate_perspective_projection


def import_synthetic_view(root_dir, idx, rgb=True, depth_linear=False,
                          semantic=False, instance=False, normals=False,
                          bbox_2d_tight=False, bbox_2d_loose=False):
    """Import views of synthetic data simulating sensors on 3D models,
    following the format output by the Data Generator extension in the `Omniverse Kaolin App`_.

    Args:
        root_dir (str): path to the root directory containin the views.
        idx (int): index of the view selected.
        rgb (bool, optional): if True, load RGB image. Default: True.
        depth_linear (bool, optional): if True, load depth map with linear scaling. Default: False.
        semantic (bool, optional): if True, load semantic segmentation map. Default: False.
        instance (bool, optional): if True, load instance segmentation map. Default: False.
        normals (bool, optional): if True, load normals map. Default: False.
        bbox_2d_tight (bool, optional): if True, load tight 2d bounding box. Default: False.
        bbox_2d_loose (bool, optional): if True, load loose 2d bounding box. Default: False.

    Returns:
        (dict):
            A dictionary of all the sensors selected depending on the arguments:

            - **rgb** (torch.FloatTensor): the RGB image, of shape :math:`(B, H, W, 3)`.
            - **depth_linear** (torch.FloatTensor):
              the depth map with linear scaling, of shape :math:`(B, H, W)`.
            - **semantic** (torch.IntTensor):
              the semantic segmentation map, of shape :math:`(B, H, W)`.
            - **instance** (torch.IntTensor):
              the instance segmentation map, of shape :math:`(B, H, W)`.
            - **bbox_2d_tight** (dict):
              the bounding box, as 4 floats (xmin, xmax, ymin, ymax).
            - **normals** (torch.FloatTensor):
              the normals map, of shape :math:`(B, H, W, 3)`.
            - And **metadata**, a dictionary containing:

              - **assets_transform** (torch.FloatTensor):
                the transformation matrix of the combined assets transformations.
              - **cam_transform** (torch.FloatTensor):
                the transformation matrix, of shape :math:`(4, 3)`.
              - **cam_proj** (torch.FloatTensor):
                the projection matrix, of shape :math:`(3, 1)`.
              - **clipping_range** (list of float):
                the range at which the object are seen, as a list of (min, max).

    .. _Omniverse Kaolin App:
        https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin/user_manual.html#data-generator
    """
    output = {}
    aspect_ratio = None

    def _import_npy(cat):
        path = os.path.join(root_dir, f'{idx}_{cat}.npy')
        if os.path.exists(path):
            output[cat] = torch.from_numpy(np.load(path))
        else:
            output[cat] = None

    def _import_png(cat):
        path = os.path.join(root_dir, f'{idx}_{cat}.png')
        if os.path.exists(path):
            output[cat] = torch.from_numpy(
                np.array(Image.open(path))
            )[:, :, :3].float() / 255.
        else:
            output[cat] = None

    if rgb:
        _import_png('rgb')

    if depth_linear:
        _import_npy('depth_linear')

    if semantic:
        _import_npy('semantic')

    if instance:
        _import_npy('instance')

    if normals:
        _import_png('normals')

    with open(os.path.join(root_dir, f'{idx}_metadata.json'), 'r') as f:
        fmetadata = json.load(f)
        asset_transforms = torch.FloatTensor(fmetadata['asset_transforms'][0][1])
        cam_transform = torch.FloatTensor(fmetadata['camera_properties']['tf_mat'])
        aspect_ratio = (fmetadata['camera_properties']['resolution']['width'] /
                        fmetadata['camera_properties']['resolution']['height'])
        focal_length = fmetadata['camera_properties']['focal_length']
        horizontal_aperture = fmetadata['camera_properties']['horizontal_aperture']
        fov = 2 * math.atan(horizontal_aperture / (2 * focal_length))
        output['metadata'] = {
            'cam_transform': cam_transform[:, :3],
            'asset_transforms': asset_transforms,
            'cam_proj': generate_perspective_projection(fov, aspect_ratio),
            'clipping_range': fmetadata['camera_properties']['clipping_range']
        }
        if bbox_2d_tight:
            output['bbox_2d_tight'] = fmetadata['bbox_2d_tight']
        if bbox_2d_loose:
            output['bbox_2d_loose'] = fmetadata['bbox_2d_loose']

    return output
