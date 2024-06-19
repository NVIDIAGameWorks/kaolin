# Copyright (c) 2019,20-22, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from PIL import Image
import pytest
import numpy as np
import shutil
import torch

from kaolin.io import utils
from kaolin.utils.testing import contained_torch_equal, check_allclose

__test_dir = os.path.dirname(os.path.realpath(__file__))
_textures_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, 'samples', 'io', 'textures')


@pytest.fixture(scope='function')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)

class TestUtils:
    @pytest.mark.parametrize(
        'handler', [utils.heterogeneous_mesh_handler_naive_homogenize, utils.mesh_handler_naive_triangulate])
    @pytest.mark.parametrize(
        'face_assignment_mode', [0, 1, 2])
    def test_mesh_handler_naive_triangulate(self, handler, face_assignment_mode):
        N = 15
        vertices = torch.rand((N, 3), dtype=torch.float32)
        face_vertex_counts = torch.LongTensor([3, 4, 5, 3, 6])
        faces = torch.LongTensor(
            [0, 1, 2,                  # Face 0 -> 1 face idx [0]
             2, 1, 3, 4,               # Face 1 -> 2 faces idx [1, 2]
             4, 5, 6, 7, 8,            # Face 2 -> 3 faces idx [3, 4, 5]
             3, 4, 6,                  # Face 3 -> 1 face idx [6]
             8, 9, 10, 11, 12, 13])    # Face 4 -> 4 faces idx [7, 8, 9, 10]
        expected_faces = torch.LongTensor(
            [[0, 1, 2],
             [2, 1, 3],   [2, 3, 4],
             [4, 5, 6],   [4, 6, 7],   [4, 7, 8],
             [3, 4, 6],
             [8, 9, 10],  [8, 10, 11], [8, 11, 12],   [8, 12, 13]])
        expected_num_faces = 11
        expected_face_vertex_counts = torch.LongTensor([3 for _ in range(expected_num_faces)])
        face_uvs_idx = torch.LongTensor(
            [0, 1, 2,                  # UVs for face 0
             10, 11, 12, 13,           # UVs for face 1
             20, 21, 22, 23, 24,       # UVs for face 2
             30, 31, 32,               # UVs for face 3
             40, 41, 42, 43, 44, 45])  # UVs for face 4
        expected_face_uvs_idx = torch.LongTensor(
            [[0, 1, 2],
             [10, 11, 12],   [10, 12, 13],
             [20, 21, 22],   [20, 22, 23],    [20, 23, 24],
             [30, 31, 32],
             [40, 41, 42],   [40, 42, 43],   [40, 43, 44],   [40, 44, 45]])

        # assignments to faces
        face_assignments = None
        expected_face_assignments = None
        with_assignments = face_assignment_mode > 0
        if with_assignments:
            if face_assignment_mode == 1:   # 1D tensors for face assignemtns replaced with new face indices
                face_assignments = {
                    '1': torch.LongTensor([0, 2]),
                    '2': torch.LongTensor([1, 3, 4])}
                expected_face_assignments = {
                    '1': torch.LongTensor([0, 3, 4, 5]),
                    '2': torch.LongTensor([1, 2, 6, 7, 8, 9, 10])}
            else:  # 2D tensors of start and end face_idx, replaced with new start and end face_idx
                face_assignments = {
                    'cat': torch.LongTensor([[0, 2], [3, 4], [2, 5]]),
                    'dog': torch.LongTensor([[1, 3]])}
                expected_face_assignments = {
                    'cat': torch.LongTensor([[0, 3], [6, 7], [3, 11]]),
                    'dog': torch.LongTensor([[1, 6]])}

        res = handler(
            vertices, face_vertex_counts, faces, face_uvs_idx, face_assignments=face_assignments)
        assert len(res) == (5 if with_assignments else 4)
        new_vertices = res[0]
        new_face_vertex_counts = res[1]
        new_faces = res[2]
        new_face_uvs_idx = res[3]

        assert torch.allclose(new_vertices, vertices)
        assert torch.equal(new_face_vertex_counts, expected_face_vertex_counts)
        assert torch.equal(new_faces, expected_faces)
        assert torch.equal(new_face_uvs_idx, expected_face_uvs_idx)

        if with_assignments:
            new_face_assignments = res[4]
            assert contained_torch_equal(new_face_assignments, expected_face_assignments)

    # No need to do all permutations
    @pytest.mark.parametrize("device_dtype_format", zip(["cuda", "cpu"], ["uint8", "float"], ["chw", "hwc"]))
    @pytest.mark.parametrize("random_image", [False])
    def test_read_write_image(self, out_dir, device_dtype_format, random_image):
        device, dtype, channel_format = device_dtype_format
        if random_image:
            image_orig = torch.rand((30, 50, 3)).to(device)
        else:
            image_orig = torch.from_numpy(np.array(Image.open(os.path.join(_textures_path, 'toppings.png'))))
            image_orig = image_orig.to(torch.float32).to(device) / 255.0
        image = image_orig
        if dtype == "uint8":
            image = (image * 255).clamp(0, 255).to(torch.uint8)
        if channel_format == "chw":
            image = image.permute(2, 0, 1)

        path = os.path.join(out_dir, 'newdir', 'image.png')
        utils.write_image(image, path)
        image_read = utils.read_image(path)  # chw
        check_allclose(image_orig.cpu(), image_read)

        # Try also batched version
        utils.write_image(image.unsqueeze(0), path)
        image_read = utils.read_image(path)  # chw
        check_allclose(image_orig.cpu(), image_read)

        # Also try 1 channel
        utils.write_image(image[..., :1] if channel_format == 'hwc' else image[:1, ...], path)
        image_read = utils.read_image(path)  # chw
        check_allclose(image_orig[..., :1].cpu(), image_read)

        # Also try 0 channel
        utils.write_image(image[..., 0] if channel_format == 'hwc' else image[0, ...], path)
        image_read = utils.read_image(path)  # chw
        check_allclose(image_orig[..., :1].cpu(), image_read)


def get_test_image(filename):
    return utils.read_image(os.path.join(_textures_path, filename))


class TestTextureExporter:
    @pytest.fixture(scope='class')
    def fox_img(self):
        return get_test_image('fox.jpg')

    @pytest.fixture(scope='class')
    def toppings_img(self):
        return get_test_image('toppings.png')

    def test_defaults(self, out_dir, fox_img, toppings_img):
        export_fn = utils.TextureExporter(out_dir, 'textures')
        path_toppings = export_fn(toppings_img, 'image')
        assert path_toppings == os.path.join('textures', 'image.png')
        path_fox = export_fn(fox_img, 'image')  # should not overwrite
        assert path_fox != path_toppings

        check_allclose(toppings_img, utils.read_image(os.path.join(out_dir, path_toppings)))
        check_allclose(fox_img, utils.read_image(os.path.join(out_dir, path_fox)))

        # Try exporting without relative dir
        export_fn = utils.TextureExporter(out_dir)
        path_toppings = export_fn(toppings_img, 'image')
        assert path_toppings == 'image.png'
        check_allclose(toppings_img, utils.read_image(os.path.join(out_dir, path_toppings)))

    @pytest.mark.parametrize("withprefix_extension", zip([True, False], ['.jpg', '.png']))
    def test_basics(self, withprefix_extension, out_dir, fox_img, toppings_img):
        with_prefix, extension = withprefix_extension
        args = {'image_extension': extension}
        if with_prefix:
            prefix = 'galactic_'
            args['file_prefix'] = prefix
        else:
            prefix = ''
        rel_dir = f'{prefix}textures_{extension[1:]}'
        export_fn = utils.TextureExporter(out_dir, rel_dir, **args)

        def _check_images_simiar(img1, img2):
            if extension == '.jpg':  # cannot guarantee all pixels are similar with jpg
                assert torch.abs(img1 - img2).mean() < 0.01, 'Fail'
            else:
                check_allclose(img1, img2)

        path_fox = export_fn(fox_img, 'fox')  # rgb
        assert path_fox == os.path.join(rel_dir, f'{prefix}fox{extension}')
        _check_images_simiar(fox_img, utils.read_image(os.path.join(out_dir, path_fox)))

        if extension == '.jpg':
            with pytest.raises(Exception):  # toppings_img is rgbs
                path_toppings = export_fn(toppings_img, 'pizza')  # cannot write RGBA with '.jpg' extension
            path_toppings = export_fn(toppings_img, 'pizza', extension='.png')
            assert path_toppings == os.path.join(rel_dir, f'{prefix}pizza.png')
            _check_images_simiar(toppings_img, utils.read_image(os.path.join(out_dir, path_toppings)))
        else:
            path_toppings = export_fn(toppings_img, 'pizza')
            assert path_toppings == os.path.join(rel_dir, f'{prefix}pizza{extension}')
            _check_images_simiar(toppings_img, utils.read_image(os.path.join(out_dir, path_toppings)))

    def test_checks_base_dir(self, out_dir, fox_img):
        export_fn = utils.TextureExporter(out_dir, 'abracadabra')
        path = export_fn(fox_img, 'fox')

        export_fn = utils.TextureExporter(out_dir + 'abracadabra')  # base dir must exist, to avoid misconfigurations
        with pytest.raises(ValueError):
            path = export_fn(fox_img, 'fox')

    def test_creates_subdir_only_on_export(self, out_dir, fox_img):
        subdir = 'magic_folder'
        abs_dir = os.path.join(out_dir, subdir)
        export_fn = utils.TextureExporter(out_dir, subdir)
        assert not os.path.exists(abs_dir)
        path = export_fn(fox_img, 'fox')
        assert os.path.exists(abs_dir)
        assert os.path.isdir(abs_dir)
        assert os.path.exists(os.path.join(out_dir, path))

    @pytest.mark.parametrize("overwrite", [True, False])
    def test_overwrite(self, overwrite, out_dir, fox_img, toppings_img):
        export_fn = utils.TextureExporter(out_dir, 'textures', overwrite_files=overwrite)
        fname = 'pretty_image'
        path1 = export_fn(fox_img, fname)
        assert path1 == os.path.join('textures', f'{fname}.png')
        path2 = export_fn(toppings_img, fname)  # write with same basename

        if overwrite:
            assert path1 == path2
            # fox got overwritten by toppings
            check_allclose(toppings_img, utils.read_image(os.path.join(out_dir, path1)))
        else:
            assert path1 != path2
            # both images got saved to their respective paths
            check_allclose(fox_img, utils.read_image(os.path.join(out_dir, path1)))
            check_allclose(toppings_img, utils.read_image(os.path.join(out_dir, path2)))
