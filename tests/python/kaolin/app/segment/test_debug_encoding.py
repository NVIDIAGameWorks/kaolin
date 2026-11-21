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

import base64
import io
import torch
from PIL import Image

from kaolin.app.segment.handler import _encode_debug_images


def _fake_outputs(h, w, n):
    return [{'render': torch.rand(3, h, w)} for _ in range(n)]


def _fake_masks(h, w, n):
    return [torch.randint(0, 2, (h, w)).float() for _ in range(n)]


def test_count_and_list_lengths():
    result = _encode_debug_images(_fake_outputs(480, 640, 3), _fake_masks(480, 640, 3))
    assert result['count'] == 3
    assert len(result['renders']) == 3
    assert len(result['masks']) == 3


def test_renders_are_valid_jpeg_rgb():
    result = _encode_debug_images(_fake_outputs(480, 640, 2), _fake_masks(480, 640, 2))
    for b64 in result['renders']:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert img.mode == 'RGB'
        assert max(img.size) <= 320


def test_masks_are_valid_png_grayscale():
    result = _encode_debug_images(_fake_outputs(480, 640, 2), _fake_masks(480, 640, 2))
    for b64 in result['masks']:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert img.mode == 'L'
        assert max(img.size) <= 320


def test_no_upscale_for_small_inputs():
    result = _encode_debug_images(_fake_outputs(100, 80, 1), _fake_masks(100, 80, 1), max_dim=320)
    render_img = Image.open(io.BytesIO(base64.b64decode(result['renders'][0])))
    assert render_img.size == (80, 100)  # PIL size is (W, H)
    mask_img = Image.open(io.BytesIO(base64.b64decode(result['masks'][0])))
    assert mask_img.size == (80, 100)


def test_empty_inputs_return_zero_count():
    result = _encode_debug_images([], [])
    assert result == {'count': 0, 'renders': [], 'masks': []}
