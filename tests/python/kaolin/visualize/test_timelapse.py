# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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


from kaolin import io
from kaolin.visualize import timelapse
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids


@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)

@pytest.fixture(scope='class')
def instancer_out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_instancer_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture(scope='module')
def voxelgrid(meshes):
    resolution = 64
    voxelgrid = trianglemeshes_to_voxelgrids(meshes[0].vertices.unsqueeze(0), meshes[0].faces,
                                             resolution)
    return voxelgrid[0].bool()


@pytest.fixture(scope='module')
def pointcloud():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud = io.usd.import_pointcloud(os.path.join(cur_dir, os.pardir, os.pardir,
                                          os.pardir, 'samples/rocket_pointcloud_GeomPoints.usda'),
                                          '/World/pointcloud').points
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_color():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = io.usd.import_pointcloud(os.path.join(cur_dir, os.pardir, os.pardir,
                                                          os.pardir, 'samples/golden/pointcloud_GeomPoints_colors.usda'),
                                                          '/World/pointcloud')
    return pointcloud, color

@pytest.fixture(scope='module')
def meshes():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    meshes = io.usd.import_meshes(os.path.join(cur_dir, os.pardir, os.pardir,
                                  os.pardir, 'samples/rocket_hetero.usd'),
                                  with_normals=True,
                                  heterogeneous_mesh_handler=io.utils.mesh_handler_naive_triangulate)
    return meshes

@pytest.fixture(scope='class')
def material_values():
    params = {
        'diffuse_color': (0., 1., 0.),
        'roughness_value': 0.1,
        'metallic_value': 1.,
        'specular_color': (1., 0., 0.),
        'is_specular_workflow': True,
    }
    material = io.materials.PBRMaterial(**params)
    yield material


@pytest.fixture(scope='class')
def material_textures():
    params = {
        'diffuse_texture': torch.rand((3, 256, 256)),
        'roughness_texture': torch.rand((1, 256, 256)),
        'metallic_texture': torch.rand((1, 256, 256)),
        'specular_texture': torch.rand((3, 256, 256)),
        'is_specular_workflow': True,
    }
    material = io.materials.PBRMaterial(**params)
    yield material


class TestTimelapse:
    def test_add_mesh_batch(self, out_dir, meshes, material_values, material_textures):
        writer = timelapse.Timelapse(out_dir)
        data = {
            0: {
                'vertices_list': [m.vertices for m in meshes],
                'faces_list': [m.faces for m in meshes],
                'uvs_list': [m.uvs for m in meshes],
                'face_uvs_idx_list': [m.face_uvs_idx for m in meshes],
                'face_normals_list': [m.face_normals for m in meshes],
                'materials_list': [{'values': material_values, 'textures': material_textures}]
            },
            10: {
                'vertices_list': [m.vertices / 2. for m in meshes],
                'faces_list': [m.faces for m in meshes],
                'materials_list': [{'values': material_values, 'textures': material_textures}]
            },
        }
        for iteration, params in data.items():
            writer.add_mesh_batch(iteration=iteration, category='test', **params)

        # Check that category directory is created
        assert os.path.exists(os.path.join(out_dir, 'test'))

        # Check that data at each iteration is correct
        texture_dir = os.path.join(out_dir, 'test', 'textures')
        assert os.path.exists(texture_dir) 
        for iteration in data.keys():
            filename = os.path.join(out_dir, 'test', 'mesh_0.usd')
            mesh_in = io.usd.import_mesh(filename, time=iteration, with_normals=True)
            # Verify mesh properties
            assert torch.allclose(data[iteration]['vertices_list'][0], mesh_in.vertices)
            assert torch.equal(data[iteration]['faces_list'][0], mesh_in.faces)
            if not data[iteration].get('face_uvs_idx_list'):
                i = 0
            else:
                i = iteration
            assert torch.allclose(data[i]['uvs_list'][0].view(-1, 2), mesh_in.uvs.view(-1, 2))
            # assert torch.equal(data[i]['face_uvs_idx_list'][0], mesh_in.face_uvs_idx)
            assert torch.allclose(data[i]['face_normals_list'][0], mesh_in.face_normals)

            materials = data[iteration]['materials_list'][0]
            # Verify materials textures exist
            for attr in ['diffuse', 'specular', 'roughness', 'metallic']:
                assert os.path.exists(os.path.join(texture_dir, f'mesh_0_textures_{iteration}_{attr}.png'))

            # Verify material properties
            for variant_name, material_data in materials.items():
                mat = io.materials.PBRMaterial().read_from_usd(filename, f'/mesh_0/{variant_name}', time=iteration)
                assert pytest.approx(mat.diffuse_color, 1e-5) == material_data.diffuse_color
                assert pytest.approx(mat.specular_color, 1e-5) == material_data.specular_color
                assert pytest.approx(mat.roughness_value, 1e-5) == material_data.roughness_value
                assert pytest.approx(mat.metallic_value, 1e-5) == material_data.metallic_value

                if material_data.diffuse_texture is not None:
                    assert torch.allclose(mat.diffuse_texture, material_data.diffuse_texture, atol=1e-2)
                    assert torch.allclose(mat.specular_texture, material_data.specular_texture, atol=1e-2)
                    assert torch.allclose(mat.roughness_texture, material_data.roughness_texture, atol=1e-2)
                    assert torch.allclose(mat.metallic_texture, material_data.metallic_texture, atol=1e-2)

    def test_add_voxelgrid_batch(self, out_dir, voxelgrid):
        writer = timelapse.Timelapse(out_dir)

        data = {
            0: {'voxelgrid_list': [voxelgrid]},
            10: {'voxelgrid_list': [voxelgrid * (torch.rand_like(voxelgrid.float()) < 0.5)]},
        }
        for iteration, params in data.items():
            writer.add_voxelgrid_batch(iteration=iteration, category='test', **params)

        # Verify
        filename = os.path.join(out_dir, 'test', 'voxelgrid_0.usd')
        for iteration, params in data.items():
            voxelgrid_in = io.usd.import_voxelgrid(filename, scene_path='/voxelgrid_0', time=iteration)

            assert torch.equal(voxelgrid_in, params['voxelgrid_list'][0])

    def test_add_pointcloud_batch(self, out_dir, pointcloud):
        writer = timelapse.Timelapse(out_dir)

        data = {
            0: {'pointcloud_list': [pointcloud], 'colors': None, 'points_type': 'usd_geom_points'},
            10: {'pointcloud_list': [pointcloud + 100.], 'colors': None, 'points_type': 'usd_geom_points'},
        }
        for iteration, params in data.items():
            writer.add_pointcloud_batch(iteration=iteration, category='test', **params)

        # Verify
        filename = os.path.join(out_dir, 'test', 'pointcloud_0.usd')
        for iteration, params in data.items():
            pointcloud_in = io.usd.import_pointcloud(filename, scene_path='/pointcloud_0', time=iteration)[0]

            assert torch.allclose(pointcloud_in, params['pointcloud_list'][0])

    def test_add_pointcloud_batch_color(self, out_dir, pointcloud_color):
        writer = timelapse.Timelapse(out_dir)

        pointcloud, color = pointcloud_color

        data = {
            0: {'pointcloud_list': [pointcloud], 'colors': [color], 'points_type': 'usd_geom_points'},
            10: {'pointcloud_list': [pointcloud + 100.], 'colors': [color], 'points_type': 'usd_geom_points'},
        }
        for iteration, params in data.items():
            writer.add_pointcloud_batch(iteration=iteration, category='test', **params)

        # Verify
        filename = os.path.join(out_dir, 'test', 'pointcloud_0.usd')
        for iteration, params in data.items():
            pointcloud_in, color_in, normals_in = io.usd.import_pointcloud(filename, scene_path='/pointcloud_0', time=iteration)

            assert torch.allclose(pointcloud_in, params['pointcloud_list'][0])

            assert torch.allclose(color_in, params['colors'][0])

    def test_add_pointcloud_batch_instancer(self, instancer_out_dir, pointcloud):
        writer = timelapse.Timelapse(instancer_out_dir)

        data = {
            0: {'pointcloud_list': [pointcloud], 'colors': None},
            10: {'pointcloud_list': [pointcloud + 100.], 'colors': None},
        }
        for iteration, params in data.items():
            writer.add_pointcloud_batch(iteration=iteration, category='test', **params)

        # Verify
        filename = os.path.join(instancer_out_dir, 'test', 'pointcloud_0.usd')
        for iteration, params in data.items():
            pointcloud_in = io.usd.import_pointcloud(filename, scene_path='/pointcloud_0', time=iteration)[0]

            assert torch.allclose(pointcloud_in, params['pointcloud_list'][0])


class TestTimelapseParser:
    @pytest.fixture(scope='class')
    def timelapse_sample_dir(self):
        # To regenerate run:
        # 'examples/tutorial/visualize_main.py \
        #    --checkpoint_interval=10 --iterations=101 --skip_normalization  \
        #    --test_objs=test/samples/rocket.obj,test/samples/model.obj --output_dir=<CLEARED_OUTPUT_DIR>
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, 'samples',
                            'timelapse', 'notexture')

    @pytest.fixture(scope='class')
    def output_dir2(self):
        # Create temporary output directory
        out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_viz_out')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        yield out_dir
        # shutil.rmtree(out_dir)  # Note: comment to keep output directory

    def test_parsing(self, timelapse_sample_dir, output_dir2, meshes):
        shutil.copytree(timelapse_sample_dir, output_dir2)

        parser = timelapse.TimelapseParser(output_dir2)
        expected_keys = [('mesh', 'ground_truth', 0),
                         ('mesh', 'ground_truth', 1),
                         ('mesh', 'output', 0),
                         ('mesh', 'output', 1),
                         ('pointcloud', 'input', 0),
                         ('pointcloud', 'input', 1),
                         ('pointcloud', 'output', 0),
                         ('pointcloud', 'output', 1),
                         ('voxelgrid', 'output', 0),
                         ('voxelgrid', 'output', 1)]
        expected_keys.sort()
        assert sorted(parser.filepaths.keys()) == expected_keys
        for k in expected_keys:
            assert os.path.exists(parser.filepaths[k])

        assert parser.num_mesh_categories() == 2
        assert parser.num_pointcloud_categories() == 2
        assert parser.num_voxelgrid_categories() == 1
        assert parser.num_mesh_items() == 4
        assert parser.num_pointcloud_items() == 4
        assert parser.num_voxelgrid_items() == 2

        expected_categories = {
            "mesh": [
                timelapse.TimelapseParser.CategoryInfo(
                    'ground_truth', ids=[0, 1], end_time=0).serializable(),
                timelapse.TimelapseParser.CategoryInfo(
                    'output', ids=[0, 1], end_time=100).serializable()],
            "pointcloud": [
                timelapse.TimelapseParser.CategoryInfo(
                    'input', ids=[0, 1], end_time=0).serializable(),
                timelapse.TimelapseParser.CategoryInfo(
                    'output', ids=[0, 1], end_time=100).serializable()],
            "voxelgrid": [
                timelapse.TimelapseParser.CategoryInfo(
                    'output', ids=[0, 1], end_time=100).serializable()]
        }
        assert set(expected_categories.keys()) == set(parser.dir_info.keys())
        for k, v in expected_categories.items():
            expected = v
            actual = parser.dir_info[k]
            assert len(expected) == len(actual)
            for i in range(len(expected)):
                for ck, cv in expected[i].items():  # Only check expected properties
                    assert (ck in actual[i])
                    assert cv == actual[i][ck]

        # Now we add another iteration
        writer = timelapse.Timelapse(output_dir2)
        writer.add_mesh_batch(iteration=200, category='output',
                              vertices_list=[m.vertices for m in meshes],
                              faces_list=[m.faces for m in meshes])
        assert parser.check_for_updates()
        assert parser.get_category_info('mesh', 'output')['end_time'] == 200

        # Now let's delete a category
        shutil.rmtree(os.path.join(output_dir2, 'output'))
        assert parser.check_for_updates()
        assert parser.num_mesh_categories() == 1
        assert parser.num_pointcloud_categories() == 1
