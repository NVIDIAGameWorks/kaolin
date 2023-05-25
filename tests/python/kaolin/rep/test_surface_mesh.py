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

import copy
import logging
import os
import pytest
import random

import torch

from kaolin.io import obj, usd
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.rep import SurfaceMesh
from kaolin.utils.testing import check_tensor_attribute_shapes, contained_torch_equal


__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, 'samples')


def data_path(fname):
    """ Return path relative to tests/samples/rep"""
    return os.path.join(__samples_path, 'rep', fname)


def rand_float_vec(shp, device):
    return torch.rand(shp, dtype=torch.float32, device=device)


_core_attr = ["vertices", "normals", "uvs", "faces", "face_uvs_idx",
              "face_normals_idx", "material_assignments", "vertex_normals",
              "face_normals", "face_uvs", "face_vertices"]


def random_mesh_attr(num_vertices=None,
                     num_normals=None,
                     num_uvs=None,
                     num_faces=None,
                     num_materials=None,
                     add_face_uvs_idx=False,
                     add_face_normals_idx=False,
                     add_material_assignments=False,
                     add_vertex_normals=False,
                     add_face_normals=False,
                     add_face_uvs=False,
                     add_face_vertices=False,
                     face_size=3,
                     float_dtype=torch.float32,
                     device="cpu"):
    """Generate random mesh attributes for testing."""
    def _val_or_rand(val):
        return val if val is not None else random.randint(30, 200)

    attr = {}
    _num_vertices = None
    if num_vertices is not None:
        attr["vertices"] = torch.rand((num_vertices, 3), dtype=float_dtype)
    if num_normals is not None:
        attr["normals"] = torch.rand((num_normals, 3), dtype=float_dtype)
    if num_uvs is not None:
        attr["uvs"] = torch.rand((num_uvs, 2), dtype=float_dtype)
    if num_faces is not None:
        _num_vertices = _val_or_rand(num_vertices)
        attr["faces"] = torch.randint(0, _num_vertices, (num_faces, face_size))
    if num_materials is not None:
        attr["materials"] = [{"material_name": f"material_{i}"} for i in range(num_faces)]

    _num_faces = _val_or_rand(num_faces)
    if add_face_uvs_idx:
        attr["face_uvs_idx"] = torch.randint(0, _val_or_rand(num_uvs), (_num_faces, face_size)).long()
    if add_face_normals_idx:
        attr["face_normals_idx"] = torch.randint(0, _val_or_rand(num_normals), (_num_faces, face_size)).long()
    if add_material_assignments:
        attr["material_assignments"] = torch.randint(0, _val_or_rand(num_materials), (_num_faces,)).short()
    if add_vertex_normals:
        _num_vertices = _val_or_rand(_num_vertices)
        attr["vertex_normals"] = torch.rand((_num_vertices, 3), dtype=float_dtype)
    if add_face_normals:
        attr["face_normals"] = torch.rand((_num_faces, face_size, 3), dtype=float_dtype)
    if add_face_uvs:
        attr["face_uvs"] = torch.rand((_num_faces, face_size, 2), dtype=float_dtype)
    if add_face_vertices:
        attr["face_vertices"] = torch.rand((_num_faces, face_size, 3), dtype=float_dtype)

    for k, v in attr.items():
        if torch.is_tensor(v):
            attr[k] = v.to(device)

    return attr


def two_squares_mesh_attr(device='cpu', float_dtype=torch.float32, quad=False):
    attr = {}
    attr['vertices'] = torch.FloatTensor([
        [-1., -1., -1.],
        [1., -1., -1.],
        [1., 1., -1.],
        [-1., 1., -1.],
        [1., 1., 1.],
        [1, -1., 1.]]).to(float_dtype)
    if quad:
        # one square at z=-1, spanning [-1,1] for x and y
        # one square at x=1, spanning [-1, 1] for z and y
        attr['faces'] = torch.LongTensor([[0, 1, 2, 3], [1, 5, 4, 2]])
        attr['normals'] = torch.FloatTensor([[0, 0, -1], [1, 0, 0]]).to(float_dtype)
        attr['face_normals_idx'] = torch.LongTensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        attr['face_normals'] = torch.FloatTensor([[[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]],
                                                 [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]).to(float_dtype)
    else:
        # same, but squares are split into triangles
        attr['faces'] = torch.LongTensor([[0, 1, 2], [0, 2, 3], [1, 5, 4], [1, 4, 2]])
        # Note: custom normals use right-hand rule, unlike default Kaolin face normal computation
        # (to allow testing both indexed and computed face normals)
        attr['normals'] = torch.FloatTensor([[0, 0, -1], [1, 0, 0]]).to(float_dtype)
        attr['face_normals_idx'] = torch.LongTensor([[0, 0, 0], [0, 0, 0], [1, 1, 1],  [1, 1, 1]])
        attr['face_normals'] = torch.FloatTensor([[[0, 0, -1], [0, 0, -1], [0, 0, -1]],
                                                  [[0, 0, -1], [0, 0, -1], [0, 0, -1]],
                                                  [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                                                  [[1, 0, 0], [1, 0, 0], [1, 0, 0]]]).to(float_dtype)
        # Compute vertex normals for vertices that share faces with different normals
        n0 = attr['normals'][0, ...]
        n1 = attr['normals'][1, ...]
        n_v2 = (n0 + n0 + n1) / 3.0
        n_v1 = (n0 + n1 + n1) / 3.0
        attr['vertex_normals'] = torch.stack([n0, n_v1, n_v2, n0, n1, n1])

    for k, v in attr.items():
        if torch.is_tensor(v):
            attr[k] = v.to(device)

    return attr


def make_default_unbatched_input(device, nfaces=4):
    V, N, U, M = 9, 12, 6, 3
    return random_mesh_attr(num_vertices=V,
                            num_normals=N,
                            num_uvs=U,
                            num_faces=nfaces,
                            num_materials=M,
                            add_face_uvs_idx=True,
                            add_face_normals_idx=True,
                            add_material_assignments=True,
                            add_vertex_normals=True,
                            add_face_normals=True,
                            add_face_uvs=True,
                            add_face_vertices=True,
                            face_size=3,
                            device=device)


def make_default_fixed_input(device, batchsize=3):
    attr_dicts = [make_default_unbatched_input(device) for i in range(batchsize)]
    res = {}
    for attr in attr_dicts[0].keys():
        if attr == 'faces':
            val = attr_dicts[0]['faces']  # just use first item's faces
        elif attr == 'materials':
            val = [attr_dicts[i][attr] for i in range(batchsize)]
        else:
            val = torch.stack([attr_dicts[i][attr] for i in range(batchsize)])
        res[attr] = val
    return res


def make_default_list_input(device, batchsize=3, fixed_topology=False):
    nfaces = [4 if fixed_topology else random.randint(3, 7) for i in range(batchsize)]

    attr_dicts = [make_default_unbatched_input(device, nfaces=nfaces[i]) for i in range(batchsize)]
    res = {}
    for attr in attr_dicts[0].keys():
        if attr == 'faces' and fixed_topology:
            val = [attr_dicts[0]['faces'] for i in range(batchsize)]
        else:
            val = [attr_dicts[i][attr] for i in range(batchsize)]
        res[attr] = val
    return res


def make_default_input(device, batching, fixed_topology_if_list=False):
    if batching == SurfaceMesh.Batching.NONE:
        return make_default_unbatched_input(device)
    elif batching == SurfaceMesh.Batching.FIXED:
        return make_default_fixed_input(device)
    elif batching == SurfaceMesh.Batching.LIST:
        return make_default_list_input(device, fixed_topology=fixed_topology_if_list)
    else:
        raise RuntimeError(f'Bug; implement for batching {batching}')


def make_batched_attribute(attr, val, batching):
    if batching == SurfaceMesh.Batching.NONE:
        return val

    if attr == 'faces':
        return [val] if batching == SurfaceMesh.Batching.LIST else val
    elif attr == 'materials':
        return [val]
    else:
        return [val] if batching == SurfaceMesh.Batching.LIST else val.unsqueeze(0)


def make_batched_attributes(attr_dict, batching):
    if batching == SurfaceMesh.Batching.NONE:
        return attr_dict

    res = {}
    for k, v in attr_dict.items():
        res[k] = make_batched_attribute(k, v, batching)
    return res


# Construct with a realistic set of attributes
def construct_mesh_default(in_attr):
    return SurfaceMesh(vertices=in_attr['vertices'],
                       faces=in_attr['faces'],
                       normals=in_attr['normals'],
                       uvs=in_attr['uvs'],
                       face_uvs_idx=in_attr['face_uvs_idx'],
                       face_normals_idx=in_attr['face_normals_idx'],
                       material_assignments=in_attr['material_assignments'],
                       materials=in_attr['materials'])


@pytest.mark.parametrize("batching", [x for x in SurfaceMesh.Batching])
class TestBasics:
    def assert_string_includes_attributes(self, in_str, required_attributes=None):
        if required_attributes is None:
            required_attributes = _core_attr

        incl_attr_superset = [rec.split(':')[0].strip() for rec in in_str.split('\n')]
        for attr in required_attributes:
            assert attr in incl_attr_superset

    def test_attribute_info_string(self, batching):
        res = SurfaceMesh.attribute_info_string(batching)
        self.assert_string_includes_attributes(res)

    def test_to_string(self, batching):
        input_attr = make_default_input("cpu", batching)
        mesh = construct_mesh_default(input_attr)
        str_default = mesh.to_string()
        self.assert_string_includes_attributes(str_default)
        str_default2 = str(mesh)
        assert str_default == str_default2

        str_detailed = mesh.to_string(detailed=True, print_stats=True)
        self.assert_string_includes_attributes(str_detailed)
        assert len(str_detailed) > len(str_default)
        assert len('\n'.split(str_detailed)) == len('\n'.split(str_default))

    def test_describe_attribute(self, batching):
        input_attr = make_default_input("cpu", batching)
        mesh = construct_mesh_default(input_attr)

        for attr in _core_attr:
            res = mesh.describe_attribute(attr)
            if mesh.has_attribute(attr):
                assert len(res) > 5

    def test_expected_shape(self, batching):
        # TODO
        pass


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("batching", [x for x in SurfaceMesh.Batching])
class TestAttributes:

    @pytest.mark.parametrize('unset_attributes_return_none', [False, True])
    @pytest.mark.parametrize('strict_checks', [False, True])
    @pytest.mark.parametrize('allow_auto_compute', [False, True])
    def test_construct_access_attributes(self, device, batching,
                                         strict_checks, unset_attributes_return_none, allow_auto_compute):
        common_args = {'strict_checks': strict_checks,
                       'unset_attributes_return_none': unset_attributes_return_none,
                       'allow_auto_compute': allow_auto_compute}
        input_attr = make_default_input(device, batching)
        expected_batch_size = 1 if batching == SurfaceMesh.Batching.NONE else 3

        input_vertices = input_attr['vertices']
        input_faces = input_attr['faces']
        if batching == SurfaceMesh.Batching.NONE:
            expected_face_vertices = index_vertices_by_faces(input_vertices.unsqueeze(0), input_faces).squeeze(0)

        # Construct with just faces and vertices
        mesh = SurfaceMesh(vertices=input_vertices, faces=input_faces, **common_args)
        mesh0_str = str(mesh)
        assert len(mesh) == expected_batch_size
        assert mesh.batching == batching   # Batching auto-detect succeeded
        assert contained_torch_equal(mesh.vertices, input_vertices, approximate=True)
        assert contained_torch_equal(mesh.faces, input_faces)
        assert contained_torch_equal(mesh.get_attribute('vertices'), input_vertices, approximate=True)
        assert contained_torch_equal(mesh.get_attribute('faces'), input_faces)
        assert set(mesh.get_attributes(only_tensors=True)) == {'vertices', 'faces'}
        with pytest.raises(AttributeError):
            tmp = mesh.abracadabra
        with pytest.raises(AttributeError):
            tmp = mesh.get_attribute('abracadabra')
        with pytest.raises(AttributeError):
            tmp = mesh.get_or_compute_attribute('abracadabra')

        if unset_attributes_return_none:
            # Convenience API
            assert mesh.uvs is None
            assert mesh.face_uvs_idx is None
            assert mesh.face_normals_idx is None
            assert mesh.material_assignments is None
            assert mesh.face_uvs is None

            # Explicit API
            for attr in ['uvs', 'face_uvs_idx', 'face_normals_idx', 'material_assignments', 'face_uvs']:
                assert mesh.get_attribute(attr) is None
                assert mesh.get_or_compute_attribute(attr) is None
        else:
            # Convenience API
            with pytest.raises(AttributeError):
                tmp = mesh.uvs
            with pytest.raises(AttributeError):
                tmp = mesh.face_uvs_idx
            with pytest.raises(AttributeError):
                tmp = mesh.face_normals_idx
            with pytest.raises(AttributeError):
                tmp = mesh.material_assignments
            with pytest.raises(AttributeError):
                tmp = mesh.face_uvs

            # Explicit API
            for attr in ['uvs', 'face_uvs_idx', 'face_normals_idx', 'material_assignments', 'face_uvs']:
                with pytest.raises(AttributeError):
                    tmp = mesh.get_attribute(attr)
                with pytest.raises(AttributeError):
                    tmp = mesh.get_or_compute_attribute(attr)
        if allow_auto_compute:
            assert mesh.vertex_normals is not None  # Can compute
            assert mesh.face_normals is not None  # Can compute
            if batching == SurfaceMesh.Batching.NONE:
                assert torch.allclose(mesh.face_vertices, expected_face_vertices)
            else:
                assert mesh.face_vertices is not None
            assert set(mesh.get_attributes(only_tensors=True)) == \
                   {'vertices', 'faces', 'face_normals', 'vertex_normals', 'face_vertices'}

            # Test that we can delete these attributes
            mesh.vertex_normals = None
            mesh.face_normals = None
            mesh.face_vertices = None
            assert set(mesh.get_attributes(only_tensors=True)) == {'vertices', 'faces'}
            for attr in ['vertex_normals', 'face_normals', 'face_vertices']:
                assert not mesh.has_attribute(attr)
                assert mesh.probably_can_compute_attribute(attr)
                assert mesh.has_or_can_compute_attribute(attr)

                # Can compute
                assert mesh.get_or_compute_attribute(attr, should_cache=False) is not None

                # But is not cached
                assert not mesh.has_attribute(attr)

                if unset_attributes_return_none:
                    assert mesh.get_attribute(attr) is None
                else:
                    with pytest.raises(AttributeError):
                        tmp = mesh.get_attribute(attr)

            for attr in ['vertex_normals', 'face_normals', 'face_vertices']:
                # Can compute and cache
                assert mesh.get_or_compute_attribute(attr, should_cache=True) is not None
                assert mesh.get_attribute(attr) is not None
                assert mesh.has_attribute(attr)
        elif unset_attributes_return_none:
            assert mesh.vertex_normals is None
            assert mesh.face_normals is None
            assert mesh.face_vertices is None

            # But we can compute these with explicit API
            for attr in ['vertex_normals', 'face_vertices', 'face_normals']:
                assert not mesh.has_attribute(attr), attr
                assert mesh.probably_can_compute_attribute(attr)
                assert mesh.has_or_can_compute_attribute(attr)
                assert mesh.get_attribute(attr) is None
                assert mesh.get_or_compute_attribute(attr, should_cache=False) is not None

            # Not set b/c caching was false
            assert mesh.vertex_normals is None
            assert mesh.face_normals is None
            assert mesh.face_vertices is None

            # Now we cache
            assert mesh.get_or_compute_attribute('vertex_normals', should_cache=True) is not None

            # Now these are set, b/c we force computed them
            assert mesh.vertex_normals is not None
            assert mesh.face_normals is not None
            assert mesh.face_vertices is not None
        else:
            with pytest.raises(AttributeError):
                tmp = mesh.vertex_normals
            with pytest.raises(AttributeError):
                tmp = mesh.face_normals
            with pytest.raises(AttributeError):
                tmp = mesh.face_vertices

        # Construct with all the attributes (note: we allow conflicts between indexed and explicit normals)
        mesh = SurfaceMesh(**input_attr, **common_args)
        assert len(mesh) == expected_batch_size
        assert set(mesh.get_attributes(only_tensors=True)) == set([k for k in input_attr.keys() if k != 'materials'])
        assert contained_torch_equal(mesh.vertices, input_vertices, approximate=True)
        assert contained_torch_equal(mesh.faces, input_faces)
        assert contained_torch_equal(mesh.normals, input_attr['normals'], approximate=True)
        assert contained_torch_equal(mesh.uvs, input_attr['uvs'], approximate=True)
        assert contained_torch_equal(mesh.face_uvs_idx, input_attr['face_uvs_idx'])
        assert contained_torch_equal(mesh.face_normals_idx, input_attr['face_normals_idx'])
        assert contained_torch_equal(mesh.material_assignments, input_attr['material_assignments'])
        assert contained_torch_equal(mesh.vertex_normals, input_attr['vertex_normals'], approximate=True)
        assert contained_torch_equal(mesh.face_normals, input_attr['face_normals'], approximate=True)
        assert contained_torch_equal(mesh.face_uvs, input_attr['face_uvs'], approximate=True)
        assert contained_torch_equal(mesh.face_vertices, input_attr['face_vertices'], approximate=True)

        # Construct with more realistic set of attributes
        mesh = SurfaceMesh(vertices=input_vertices,
                           faces=input_faces,
                           normals=input_attr['normals'],
                           uvs=input_attr['uvs'],
                           face_uvs_idx=input_attr['face_uvs_idx'],
                           face_normals_idx=input_attr['face_normals_idx'],
                           material_assignments=input_attr['material_assignments'],
                           materials=input_attr['materials'],
                           **common_args)
        mesh2_str = str(mesh)
        assert len(mesh0_str) < len(mesh2_str)
        assert len(mesh) == expected_batch_size
        assert set(mesh.get_attributes(only_tensors=True)) == \
               {'vertices', 'faces', 'normals', 'uvs', 'face_uvs_idx', 'face_normals_idx', 'material_assignments'}
        assert contained_torch_equal(mesh.vertices, input_vertices, approximate=True)
        assert contained_torch_equal(mesh.faces, input_faces)
        assert contained_torch_equal(mesh.normals, input_attr['normals'], approximate=True)
        assert contained_torch_equal(mesh.uvs, input_attr['uvs'], approximate=True)
        assert contained_torch_equal(mesh.face_uvs_idx, input_attr['face_uvs_idx'])
        assert contained_torch_equal(mesh.face_normals_idx, input_attr['face_normals_idx'])
        assert contained_torch_equal(mesh.material_assignments, input_attr['material_assignments'])
        if allow_auto_compute:
            assert mesh.vertex_normals is not None  # can compute
            assert mesh.face_normals is not None  # can compute
            assert mesh.face_uvs is not None  # can compute
            if batching == SurfaceMesh.Batching.NONE:
                assert torch.allclose(mesh.face_vertices, expected_face_vertices)
            else:
                assert mesh.face_vertices is not None
            assert set(mesh.get_attributes(only_tensors=True)) == \
                   {'vertices', 'faces', 'normals', 'uvs', 'face_uvs_idx', 'face_normals_idx', 'material_assignments',
                    'vertex_normals', 'face_normals', 'face_uvs', 'face_vertices'}
        elif unset_attributes_return_none:
            assert mesh.vertex_normals is None
            assert mesh.face_normals is None
            assert mesh.face_uvs is None
            assert mesh.face_vertices is None
        else:
            with pytest.raises(AttributeError):
                tmp = mesh.vertex_normals
            with pytest.raises(AttributeError):
                tmp = mesh.face_normals
            with pytest.raises(AttributeError):
                tmp = mesh.face_uvs
            with pytest.raises(AttributeError):
                tmp = mesh.face_vertices

        # Let's try to construct with an unexpected type of input
        if batching == SurfaceMesh.Batching.LIST:
            input_faces = [torch.randint(0, 15, (1, 15, 3)).long() for i in range(len(input_faces))]
        else:
            input_attr['face_normals_idx'] = input_attr['face_normals_idx'][..., :-1, :]  # face number does not match
        if not strict_checks:
            # should succeed
            mesh = SurfaceMesh(vertices=input_vertices,
                               faces=input_faces,
                               normals=input_attr['normals'],
                               face_normals_idx=input_attr['face_normals_idx'],
                               **common_args)
            assert len(mesh) == expected_batch_size
        else:
            with pytest.raises(ValueError):
                mesh = SurfaceMesh(vertices=input_vertices,
                                   faces=input_faces,
                                   normals=input_attr['normals'],
                                   face_normals_idx=input_attr['face_normals_idx'],
                                   **common_args)

    @pytest.mark.parametrize('deep', [False, True])
    def test_copy(self, device, batching, deep):
        input_attr = make_default_input(device, batching)
        mesh = construct_mesh_default(input_attr)
        mesh_str = str(mesh)
        orig_attr = sorted(mesh.get_attributes())

        print(f'Copied mesh')
        if deep:
            mesh1 = copy.deepcopy(mesh)
        else:
            mesh1 = copy.copy(mesh)
        mesh1_str = str(mesh1)
        copied_attr = sorted(mesh1.get_attributes())

        assert contained_torch_equal(orig_attr, copied_attr)
        for attr in orig_attr:
            orig_val = getattr(mesh, attr)
            new_val = getattr(mesh1, attr)
            assert contained_torch_equal(orig_val, new_val)
            if torch.is_tensor(orig_val):
                if deep:
                    assert orig_val is not new_val  # different references
                else:
                    assert orig_val is new_val  # same reference

        # Also check printed summaries
        assert mesh_str == mesh1_str

    @pytest.mark.parametrize('method', ['set', 'function'])  # How to set batching
    @pytest.mark.parametrize('convert_to_batching', [x for x in SurfaceMesh.Batching])
    def test_set_batching(self, device, batching, convert_to_batching, method):
        """ Convert to another batching strategy and test that the output is as expected. """
        input_attr = make_default_input(device, batching)
        mesh = construct_mesh_default(input_attr)
        assert mesh.batching == batching

        def _apply_batching(in_mesh, in_batching):
            if method == 'set':
                in_mesh.batching = in_batching
            else:
                in_mesh.set_batching(in_batching)

        with pytest.raises(ValueError):
            mesh_tmp = copy.copy(mesh)
            _apply_batching(mesh_tmp, 1022)  # bogus value

        if batching == convert_to_batching:
            mesh_orig = copy.copy(mesh)
            _apply_batching(mesh, convert_to_batching)
            assert contained_torch_equal(mesh, mesh_orig, approximate=True)
        elif batching == SurfaceMesh.Batching.NONE:  # can convert to any batching
            mesh_copy = copy.copy(mesh)
            batched_attr = make_batched_attributes(input_attr, convert_to_batching)  # TODO: add expected_face_vertices
            _apply_batching(mesh, convert_to_batching)
            assert contained_torch_equal(mesh.faces, batched_attr['faces'])
            assert contained_torch_equal(mesh.face_uvs_idx, batched_attr['face_uvs_idx'])
            assert contained_torch_equal(mesh.face_normals_idx, batched_attr['face_normals_idx'])
            assert contained_torch_equal(mesh.material_assignments, batched_attr['material_assignments'])
            assert contained_torch_equal(mesh.vertices, batched_attr['vertices'], approximate=True)
            assert contained_torch_equal(mesh.normals, batched_attr['normals'], approximate=True)
            assert contained_torch_equal(mesh.uvs, batched_attr['uvs'], approximate=True)
            assert contained_torch_equal(mesh.materials, batched_attr['materials'], approximate=True)

            assert contained_torch_equal(mesh.faces, mesh_copy.getattr_batched('faces', convert_to_batching))
            assert contained_torch_equal(mesh.face_uvs_idx, mesh_copy.getattr_batched('face_uvs_idx', convert_to_batching))
            assert contained_torch_equal(mesh.face_normals_idx, mesh_copy.getattr_batched('face_normals_idx', convert_to_batching))
            assert contained_torch_equal(mesh.material_assignments, mesh_copy.getattr_batched('material_assignments', convert_to_batching))
            assert contained_torch_equal(mesh.vertices, mesh_copy.getattr_batched('vertices', convert_to_batching), approximate=True)
            assert contained_torch_equal(mesh.normals, mesh_copy.getattr_batched('normals', convert_to_batching), approximate=True)
            assert contained_torch_equal(mesh.uvs, mesh_copy.getattr_batched('uvs', convert_to_batching), approximate=True)
            assert contained_torch_equal(mesh.materials, mesh_copy.getattr_batched('materials', convert_to_batching), approximate=True)

            # Check that for unbatched meshes result equivalent for to_batched and set_batching(FIXED)
            if convert_to_batching == SurfaceMesh.Batching.FIXED:
                mesh_copy.to_batched()
                assert contained_torch_equal(mesh, mesh_copy)
        elif batching == SurfaceMesh.Batching.FIXED:
            if convert_to_batching == SurfaceMesh.Batching.NONE:
                with pytest.raises(ValueError):
                    _apply_batching(mesh, convert_to_batching)

                # Let's try making fixed size mesh with size 1 -- that one is convertible
            elif convert_to_batching == SurfaceMesh.Batching.LIST:
                batch_size = input_attr['vertices'].shape[0]
                _apply_batching(mesh, convert_to_batching)
                assert contained_torch_equal(
                    mesh.faces, [input_attr['faces'] for i in range(batch_size)])
                assert contained_torch_equal(
                    mesh.face_uvs_idx, [input_attr['face_uvs_idx'][i, ...] for i in range(batch_size)])
                assert contained_torch_equal(
                    mesh.face_normals_idx, [input_attr['face_normals_idx'][i, ...] for i in range(batch_size)])
                assert contained_torch_equal(
                    mesh.material_assignments, [input_attr['material_assignments'][i, ...] for i in range(batch_size)])
                assert contained_torch_equal(
                    mesh.vertices, [input_attr['vertices'][i, ...] for i in range(batch_size)], approximate=True)
                assert contained_torch_equal(
                    mesh.normals, [input_attr['normals'][i, ...] for i in range(batch_size)], approximate=True)
                assert contained_torch_equal(
                    mesh.uvs, [input_attr['uvs'][i, ...] for i in range(batch_size)], approximate=True)
            else:
                raise RuntimeError(f'Bug conversion {batching} --> {convert_to_batching} not tested')
        elif batching == SurfaceMesh.Batching.LIST:
            # Default list mesh is not convertible to other batchings
            with pytest.raises(ValueError):
                _apply_batching(mesh, convert_to_batching)

            # Try to make a list batch that's fixed topology -- convertible to FIXED
            input_attr = make_default_list_input(device, batchsize=3, fixed_topology=True)
            mesh = construct_mesh_default(input_attr)
            assert mesh.batching == batching
            if convert_to_batching == SurfaceMesh.Batching.FIXED:
                _apply_batching(mesh, convert_to_batching)
                assert torch.equal(input_attr['faces'][0], mesh.faces)
                assert contained_torch_equal(input_attr['vertices'], [mesh.vertices[i, ...] for i in range(len(mesh))])
            else:
                with pytest.raises(ValueError):
                    _apply_batching(mesh, convert_to_batching)

            # Try to make a list batch of size 1 -- that's convertible to either
            input_attr = make_default_list_input(device, batchsize=1, fixed_topology=True)
            mesh = construct_mesh_default(input_attr)
            _apply_batching(mesh, convert_to_batching)
        else:
            raise RuntimeError(f'Bug conversion from batching {batching} not tested')

    def test_set_attributes(self, device, batching):
        input_attr = make_default_input(device, batching)
        mesh = construct_mesh_default(input_attr)
        assert mesh.batching == batching

        # Make new default input
        input_attr2 = make_default_input(device, batching)

        # User is allowed to set any value; we do not disallow checks
        mesh.vertices = input_attr2['vertices']
        mesh.faces = input_attr2['faces']
        mesh.uvs = input_attr2['uvs']
        mesh.face_uvs_idx = input_attr2['face_uvs_idx']
        mesh. face_normals_idx = input_attr2['face_normals_idx']
        mesh.material_assignments = input_attr2['material_assignments']
        mesh.vertex_normals = input_attr2['vertex_normals']
        mesh.face_normals = input_attr2['face_normals']
        mesh.face_uvs = input_attr2['face_uvs']
        mesh.face_vertices = input_attr2['face_vertices']
        mesh.materials = input_attr2['materials']

    def test_check_sanity(self, device, batching):
        """ Check that unexpected tensor sizes are detected (valid arguments already tested in the construction test).
        """
        input_attr = make_default_input(device, batching)
        mesh = construct_mesh_default(input_attr)
        assert mesh.batching == batching
        assert mesh.check_sanity()

        # Now let's try changing a few attributes
        if batching == SurfaceMesh.Batching.LIST:
            mesh_copy = copy.copy(mesh)
            mesh_copy.vertices = [torch.rand((10, 3)) for _ in range((len(mesh_copy)) - 1)]  # wrong batch size
            assert mesh.check_sanity()  # orig mech unchanged
            assert not mesh_copy.check_sanity()
            mesh_copy = copy.copy(mesh)
            mesh_copy.faces = torch.randint(0, 100, (5, 100, 3))  # not list
            assert mesh.check_sanity()  # orig mech unchanged
            assert not mesh_copy.check_sanity()

    def test_as_dict(self, device, batching):
        input_attr = make_default_input(device, batching)
        input_attr = {k: v for k, v in input_attr.items()
                      if k in ['vertices', 'faces', 'normals', 'uvs', 'face_uvs_idx', 'face_normals_idx',
                               'material_assignments', 'materials']}
        input_attr['allow_auto_compute'] = True
        mesh = SurfaceMesh(**input_attr)
        input_attr['batching'] = mesh.batching
        input_attr['unset_attributes_return_none'] = mesh.unset_attributes_return_none
        assert contained_torch_equal(input_attr, mesh.as_dict(), print_error_context='')

        assert mesh.face_vertices is not None  # auto compute
        assert mesh.face_normals is not None  # auto compute
        input_attr['face_vertices'] = mesh.face_vertices
        input_attr['face_normals'] = mesh.face_normals
        assert contained_torch_equal(input_attr, mesh.as_dict(), print_error_context='')

    def test_empty_faces(self, device, batching):
        input_attr = make_default_input(device, batching)
        if batching == SurfaceMesh.Batching.LIST:
            empty_faces = [torch.zeros((0, 3)).long().to(device) for i in range(len(input_attr['faces']))]
        else:
            empty_faces = torch.zeros((0, 3)).long().to(device)
        input_attr['faces'] = empty_faces

        mesh = construct_mesh_default(input_attr)
        assert mesh.check_sanity()
        assert mesh.face_vertices is not None
        if batching == SurfaceMesh.Batching.NONE:
            assert mesh.face_vertices.shape[0] == 0
        elif batching == SurfaceMesh.Batching.FIXED:
            assert list(mesh.face_vertices.shape) == [len(mesh), 0, 3, 3]
        elif batching == SurfaceMesh.Batching.LIST:
            for i in range(len(mesh)):
                assert mesh.face_vertices[i].shape[0] == 0
        else:
            raise RuntimeError(f'Error not tested for batching {batching}')

    def test_empty_vertices(self, device, batching):
        input_attr = make_default_input(device, batching)
        if batching == SurfaceMesh.Batching.LIST:
            empty_vertices = [torch.zeros((0, 3)).float().to(device) for i in range(len(input_attr['vertices']))]
        elif batching == SurfaceMesh.Batching.NONE:
            empty_vertices = torch.zeros((0, 3)).float().to(device)
        elif batching == SurfaceMesh.Batching.FIXED:
            empty_vertices = torch.zeros((input_attr['vertices'].shape[0], 0, 3)).float().to(device)
        else:
            raise RuntimeError(f'Error not tested for batching {batching}')
        input_attr['vertices'] = empty_vertices

        mesh = construct_mesh_default(input_attr)
        assert mesh.check_sanity()
        assert mesh.face_vertices is None  # Fails to compute

    @pytest.mark.parametrize('to_attr', ['vertices', 'normals'])
    @pytest.mark.parametrize('through_attr', ['face_vertices', 'face_normals', 'vertex_normals'])
    def test_backprop_to_vertices(self, device, batching, to_attr, through_attr):
        """
        Test that we can backprop to optimize to_attr (e.g. mesh.vertices) when the loss is
        computed using auto-computed through_attr (e.g. mesh.face_vertices).
        This is testing that the mesh's internal caching is off when a gradient is required.
        """
        if to_attr == 'normals' and through_attr == 'face_vertices':
            # The only combination that is not related
            return

        input_attr = make_default_input(device, batching, fixed_topology_if_list=True)
        extra_args = {}
        if to_attr == 'normals':
            extra_args['normals'] = input_attr['normals']
            extra_args['face_normals_idx'] = input_attr['face_normals_idx']
        mesh = SurfaceMesh(faces=input_attr['faces'], vertices=input_attr['vertices'], **extra_args)
        mesh.to_batched()
        if to_attr == 'vertices':
            mesh.vertices.requires_grad = True
            params = [mesh.vertices]
        elif to_attr == 'normals':
            mesh.normals.requires_grad = True
            params = [mesh.normals]
        else:
            raise RuntimeError(f'Not tested backpropagation to attr {to_attr}')

        crit = torch.nn.MSELoss()
        def compute_loss():
            if through_attr == 'vertex_normals':
                vn = mesh.vertex_normals
                # Let's say want all normals equal to 0, 1, 0
                return crit(mesh.vertex_normals,
                            torch.tensor([0, 1, 0]).float().to(device).reshape((1, 1, 3)).repeat(1, vn.shape[1], 1))
            else:
                if through_attr == 'face_vertices':
                    face_attr = mesh.face_vertices
                elif through_attr == 'face_normals':
                    face_attr = mesh.face_normals
                else:
                    raise RuntimeError(f'Bug, not tested for {through_attr}')
            # Let's say we want first and second vertex to be exactly 1 unit apart
            distances = torch.sqrt(torch.pow(face_attr[..., 0, :] - face_attr[..., 1, :], 2).sum(dim=-1))
            return crit(distances, torch.ones_like(distances))
        optim = torch.optim.Adam(params=params, lr=5e-4)

        loss = None
        for i in range(3):
            optim.zero_grad()
            prior_loss = loss

            loss = compute_loss()
            if prior_loss is not None:
                assert not torch.allclose(loss, prior_loss)

            loss.backward()
            if to_attr == 'vertices':
                prior_value = torch.clone(mesh. vertices)
                assert not torch.allclose(mesh.vertices.grad, torch.zeros_like(mesh.vertices.grad))
            elif to_attr == 'normals':
                prior_value = torch.clone(mesh.normals)
                assert not torch.allclose(mesh.normals.grad, torch.zeros_like(mesh.normals.grad))
            else:
                raise RuntimeError(f'Bug, not tested for {to_attr}')

            optim.step()
            if to_attr == 'vertices':
                assert not torch.allclose(mesh.vertices, prior_value)
            elif to_attr == 'normals':
                assert not torch.allclose(mesh.normals, prior_value)
            else:
                raise RuntimeError(f'Bug, not tested for {to_attr}')

    def test_backprop_face_uvs(self, device, batching):
        """
        Test that we can backprop to optimize mesh.uvs when the loss is
        computed using auto-computed mesh.face_uvs.
        This is testing that the mesh's internal caching is off when a gradient is required.
        """
        input_attr = make_default_input(device, batching, fixed_topology_if_list=True)
        mesh = construct_mesh_default(input_attr)
        mesh.to_batched()
        mesh.uvs.requires_grad = True

        # Let's say we want first and second uvs to be exactly 1 unit apart
        crit = torch.nn.MSELoss()

        def compute_loss():
            face_uvs = mesh.face_uvs
            distances = torch.sqrt(torch.pow(face_uvs[:, 0, :] - face_uvs[:, 1, :], 2).sum(dim=-1))
            return crit(distances, torch.ones_like(distances))

        optim = torch.optim.Adam(params=[mesh.uvs], lr=5e-4)

        loss = None
        for i in range(3):
            optim.zero_grad()
            prior_loss = loss
            prior_uvs = torch.clone(mesh.uvs)
            loss = compute_loss()
            if prior_loss is not None:
                assert not torch.allclose(loss, prior_loss)

            loss.backward()
            assert not torch.allclose(mesh.uvs.grad, torch.zeros_like(mesh.uvs.grad))
            optim.step()
            assert not torch.allclose(mesh.uvs, prior_uvs)

    def test_vertex_normals(self, device, batching):
        attr = two_squares_mesh_attr(device=device, quad=False)
        # Indexed normals use right-hand rule, but auto-computed normals use left-hand rule
        flipped_vertex_normals = make_batched_attribute('vertex_normals', attr['vertex_normals'] * -1, batching)

        # Test can conpute vertex normal by computing face normals and then vertex normals
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'])
        mesh.set_batching(batching)
        attr = make_batched_attributes(attr, batching)
        assert contained_torch_equal(mesh.vertex_normals, flipped_vertex_normals, approximate=True, atol=1e-4)

        # Test can compute vertex normals from indexed face normals
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'],
                           normals=attr['normals'], face_normals_idx=attr['face_normals_idx'])
        mesh.set_batching(batching)
        assert contained_torch_equal(mesh.vertex_normals, attr['vertex_normals'], approximate=True)

        # Test when vertex_normals are provided as input
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'],
                           vertex_normals=attr['vertex_normals'])
        mesh.set_batching(batching)
        assert mesh.normals is None
        assert mesh.face_normals_idx is None
        assert contained_torch_equal(mesh.vertex_normals, attr['vertex_normals'], approximate=True)  # provided

    def test_face_normals_tri(self, device, batching):
        # Test that can compute face normals, given just vertices, for tri meshes
        attr = two_squares_mesh_attr(device=device, quad=False)
        # Indexed normals use right-hand rule, but auto-computed normals use left-hand rule
        flipped_face_normals = make_batched_attribute('face_normals', attr['face_normals'] * -1, batching)

        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'])
        mesh.set_batching(batching)
        attr = make_batched_attributes(attr, batching)
        assert mesh.normals is None
        assert mesh.face_normals_idx is None
        assert contained_torch_equal(mesh.face_normals, flipped_face_normals, approximate=True)

        # Test that can index normals correctly
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'],
                           normals=attr['normals'], face_normals_idx=attr['face_normals_idx'])
        mesh.set_batching(batching)
        assert contained_torch_equal(mesh.normals, attr['normals'], approximate=True)
        assert contained_torch_equal(mesh.face_normals_idx, attr['face_normals_idx'])
        assert contained_torch_equal(mesh.face_normals, attr['face_normals'], approximate=True)
        assert contained_torch_equal(mesh.vertex_normals, attr['vertex_normals'], approximate=True)

    def test_face_normals_quad(self, device, batching):
        # Test that can gracefully not compute for quad meshes
        attr = two_squares_mesh_attr(device=device, quad=True)
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'])
        mesh.set_batching(batching)
        assert mesh.face_normals is None

        # Test that can compute correctly given indexing
        attr = make_batched_attributes(attr, batching)
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'],
                           normals=attr['normals'], face_normals_idx=attr['face_normals_idx'])
        mesh.set_batching(batching)
        assert contained_torch_equal(mesh.normals, attr['normals'], approximate=True)
        assert contained_torch_equal(mesh.face_normals_idx, attr['face_normals_idx'])
        assert contained_torch_equal(mesh.face_normals, attr['face_normals'], approximate=True)

        # Test that nothing bad happens if only normals or face_normals_idx are set
        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'],
                           normals=attr['normals'])
        mesh.set_batching(batching)
        assert contained_torch_equal(mesh.normals, attr['normals'], approximate=True)
        assert mesh.face_normals_idx is None
        assert mesh.face_normals is None

        mesh = SurfaceMesh(vertices=attr['vertices'], faces=attr['faces'],
                           face_normals_idx=attr['face_normals_idx'])
        mesh.set_batching(batching)
        assert contained_torch_equal(mesh.face_normals_idx, attr['face_normals_idx'])
        assert mesh.normals is None
        assert mesh.face_normals is None

    def test_face_uvs(self, device, batching):
        input_attr = make_default_input(device, batching)
        expected_batch_size = 1 if batching == SurfaceMesh.Batching.NONE else 3
        mesh = SurfaceMesh(faces=input_attr['faces'],
                           vertices=input_attr['vertices'],
                           uvs=input_attr['uvs'],
                           face_uvs_idx=input_attr['face_uvs_idx'])
        assert not mesh.has_attribute('face_uvs')  # Does not have it to start
        assert mesh.face_uvs is not None           # Is auto-computed
        assert mesh.has_attribute('face_uvs')      # It's cached

        mesh.set_batching(SurfaceMesh.Batching.LIST)
        assert len(mesh) == expected_batch_size
        for b in range(len(mesh)):
            if batching == SurfaceMesh.Batching.NONE:
                uvs = input_attr['uvs']
                face_uvs_idx = input_attr['face_uvs_idx']
            elif batching == SurfaceMesh.Batching.LIST:
                uvs = input_attr['uvs'][b]
                face_uvs_idx = input_attr['face_uvs_idx'][b]
            elif batching == SurfaceMesh.Batching.FIXED:
                uvs = input_attr['uvs'][b, ...]
                face_uvs_idx = input_attr['face_uvs_idx'][b, ...]
            else:
                raise RuntimeError(f'Not tested for batching {batching}')
            actual_face_uvs = mesh.face_uvs[b]
            for f in range(face_uvs_idx.shape[0]):
                for v in range(face_uvs_idx.shape[1]):
                    assert torch.allclose(uvs[face_uvs_idx[f, v], :], actual_face_uvs[f, v, :]), f'Failed for f {f}, v {v}'

    def test_face_vertices(self, device, batching):
        input_attr = make_default_input(device, batching)
        expected_batch_size = 1 if batching == SurfaceMesh.Batching.NONE else 3
        mesh = SurfaceMesh(faces=input_attr['faces'],
                           vertices=input_attr['vertices'])
        assert not mesh.has_attribute('face_vertices')  # Does not have it to start
        assert mesh.face_vertices is not None  # Is auto-computed
        assert mesh.has_attribute('face_vertices')  # It's cached

        mesh.set_batching(SurfaceMesh.Batching.LIST)
        assert len(mesh) == expected_batch_size
        for b in range(len(mesh)):
            if batching == SurfaceMesh.Batching.NONE:
                vertices = input_attr['vertices']
                faces = input_attr['faces']
            elif batching == SurfaceMesh.Batching.LIST:
                vertices = input_attr['vertices'][b]
                faces = input_attr['faces'][b]
            elif batching == SurfaceMesh.Batching.FIXED:
                vertices = input_attr['vertices'][b, ...]
                faces = input_attr['faces']
            else:
                raise RuntimeError(f'Not tested for batching {batching}')
            actual_face_vertices = mesh.face_vertices[b]
            for f in range(faces.shape[0]):
                for v in range(faces.shape[1]):
                    assert torch.allclose(vertices[faces[f, v], :],
                                          actual_face_vertices[f, v, :]), f'Failed for f {f}, v {v}'

    @pytest.mark.parametrize("method_to_test", ["to", "named"])  # if to test mesh.to or mesh.cpu/mesh.cuda
    @pytest.mark.parametrize("to_device", ["cuda", "cpu"])
    def test_device_convert(self, device, batching, to_device, method_to_test):
        input_attr = make_default_input(device, batching, fixed_topology_if_list=True)
        mesh = construct_mesh_default(input_attr)
        if method_to_test == "to":
            mesh_converted = mesh.to(to_device, attributes=["vertices", "faces"])
        elif to_device == "cuda":
            mesh_converted = mesh.cuda(attributes=["vertices", "faces"])
        elif to_device == "cpu":
            mesh_converted = mesh.cpu(["vertices", "faces"])
        else:
            raise RuntimeError(f'Bug; unknown test condition.')

        # Convert to fixed batching for easier tests
        mesh.set_batching(SurfaceMesh.Batching.FIXED)
        mesh_converted.set_batching(SurfaceMesh.Batching.FIXED)

        # orig mesh, orig device
        assert mesh.vertices.device.type == device
        assert mesh.faces.device.type == device
        assert mesh.normals.device.type == device

        # converted mesh, converted device
        assert mesh_converted.vertices.device.type == to_device     # converted
        assert mesh_converted.faces.device.type == to_device        # converted
        assert mesh_converted.normals.device.type == device      # not converted
        assert mesh_converted.uvs.device.type == device  # not converted
        assert mesh_converted.face_uvs_idx.device.type == device  # not converted
        assert mesh_converted.face_normals_idx.device.type == device  # not converted
        assert mesh_converted.material_assignments.device.type == device  # not converted

        # Now we convert all tensors
        if method_to_test == "to":
            mesh_converted = mesh.to(to_device)
        elif to_device == "cuda":
            mesh_converted = mesh.cuda()
        elif to_device == "cpu":
            mesh_converted = mesh.cpu()
        else:
            raise RuntimeError(f'Bug; unknown test condition.')
        assert mesh_converted.vertices.device.type == to_device
        assert mesh_converted.faces.device.type == to_device
        assert mesh_converted.normals.device.type == to_device
        assert mesh_converted.uvs.device.type == to_device
        assert mesh_converted.face_uvs_idx.device.type == to_device
        assert mesh_converted.face_normals_idx.device.type == to_device
        assert mesh_converted.material_assignments.device.type == to_device

    def test_type_convert(self, device, batching):
        input_attr = make_default_input(device, batching, fixed_topology_if_list=True)
        mesh = construct_mesh_default(input_attr)
        to_dtype = torch.float64
        mesh_converted = mesh.float_tensors_to(to_dtype)

        # Convert to fixed batching for easier tests
        mesh.set_batching(SurfaceMesh.Batching.FIXED)
        mesh_converted.set_batching(SurfaceMesh.Batching.FIXED)

        orig_dtype = torch.float32
        assert mesh.vertices.dtype == orig_dtype
        assert mesh.normals.dtype == orig_dtype
        assert mesh.uvs.dtype == orig_dtype
        assert mesh.vertex_normals.dtype == orig_dtype
        assert mesh.face_normals.dtype == orig_dtype
        assert mesh.face_uvs.dtype == orig_dtype
        assert mesh.face_vertices.dtype == orig_dtype

        assert mesh_converted.vertices.dtype == to_dtype
        assert mesh_converted.normals.dtype == to_dtype
        assert mesh_converted.uvs.dtype == to_dtype
        assert mesh_converted.vertex_normals.dtype == to_dtype
        assert mesh_converted.face_normals.dtype == to_dtype
        assert mesh_converted.face_uvs.dtype == to_dtype
        assert mesh_converted.face_vertices.dtype == to_dtype

    def test_cat_fixed_topology(self, device, batching):
        """Test concatenation of fixed topology meshes, which succeeds even when fixed_topology=False,
        creating a list batching instead."""
        debug_print = False
        import_args = {'with_materials': True, 'with_normals': True}
        log_args = {'detailed': True, 'print_stats': True}

        # 1. Load two fixed topology meshes
        flat_mesh = obj.import_mesh(data_path('ico_flat.obj'), **import_args)
        if debug_print:
            print('FLAT mesh')
            print(flat_mesh.to_string(**log_args))
        flat_mesh = flat_mesh.set_batching(batching).to(device)

        smooth_mesh = obj.import_mesh(data_path('ico_smooth.obj'), **import_args)
        # print('SMOOTH mesh')
        # print(smooth_mesh.to_string(**log_args))
        smooth_mesh = smooth_mesh.set_batching(batching).to(device)

        source_meshes = [flat_mesh, smooth_mesh]

        # 2. Concatenate meshes, asking for FIXED batching
        expected_shapes = {'vertices': [2, 42, 3], 'faces': [80,  3], 'material_assignments': [2, 80],
                           'face_normals': [2, 80, 3, 3], 'uvs': [2, 63, 2], 'face_uvs_idx': [2, 80, 3]}
        result = SurfaceMesh.cat(source_meshes, fixed_topology=True)
        if debug_print:
            print('Result')
            print(result.to_string(**log_args))
        assert result.batching == SurfaceMesh.Batching.FIXED
        assert result.check_sanity()
        assert len(result) == 2
        assert check_tensor_attribute_shapes(result, **expected_shapes)

        # Because normals have different shapes across two meshes, we expect indexed normals to be removed
        # (Note that face_normals are batched instead)
        assert result.normals is None
        assert result.face_normals_idx is None
        assert type(result.materials) == list
        assert len(result.materials) == 2

        # Check value of all the items
        for idx in range(len(source_meshes)):
            source_meshes[idx].set_batching(SurfaceMesh.Batching.NONE)

            assert torch.allclose(result.vertices[idx, ...], source_meshes[idx].vertices)
            assert torch.equal(result.faces, source_meshes[idx].faces)
            assert torch.equal(result.face_uvs_idx[idx, ...], source_meshes[idx].face_uvs_idx)
            assert torch.allclose(result.uvs[idx, ...], source_meshes[idx].uvs)
            assert torch.allclose(result.face_normals[idx, ...], source_meshes[idx].face_normals, atol=1e-4)
            assert torch.allclose(result.vertex_normals[idx, ...], source_meshes[idx].vertex_normals, atol=1e-4)

        # 3. Concatenate meshes, asking for LIST batching (should still work)
        result_list = SurfaceMesh.cat(source_meshes, fixed_topology=False)
        if debug_print:
            print('Result')
            print(result_list.to_string(**log_args))
        assert result_list.batching == SurfaceMesh.Batching.LIST
        assert result_list.check_sanity()
        assert len(result_list) == 2
        # set_batching is less smart than cat; it will simply ignore un-stackable items
        assert result_list.vertex_normals is not None
        assert result_list.face_normals is not None
        assert result_list.face_uvs is not None
        result_list.set_batching(SurfaceMesh.Batching.FIXED, skip_errors=True)
        assert contained_torch_equal(result.vertices, result_list.vertices, approximate=True)
        assert contained_torch_equal(result.faces, result_list.faces, approximate=True)
        assert contained_torch_equal(result.face_normals, result_list.face_normals, approximate=True)
        assert contained_torch_equal(result.face_uvs, result_list.face_uvs, approximate=True)
        assert contained_torch_equal(result.vertex_normals, result_list.vertex_normals, approximate=True)

    def test_cat_variable_topology(self, device, batching):
        """Test concatenation of fixed topology meshes, which succeeds even when fixed_topology=False,
        creating a list batching instead."""
        debug_print = False
        import_args = {'with_materials': True, 'with_normals': True}

        # Load multiple meshes from USD and concatenate them
        amsterdam_meshes = usd.import_meshes(data_path('amsterdam.usd'), **import_args)
        input_meshes = amsterdam_meshes
        num_amsterdam = 18
        assert len(amsterdam_meshes) == num_amsterdam   # list
        amsterdam = SurfaceMesh.cat(amsterdam_meshes, fixed_topology=False)
        assert len(amsterdam) == num_amsterdam   # SurfaceMesh
        assert amsterdam.batching == SurfaceMesh.Batching.LIST

        # Load and cat two fixed topology meshes
        flat_mesh = obj.import_mesh(data_path('ico_flat.obj'), **import_args)
        smooth_mesh = obj.import_mesh(data_path('ico_smooth.obj'), **import_args)
        input_meshes.extend([flat_mesh, smooth_mesh])
        fixed_meshes = SurfaceMesh.cat([flat_mesh, smooth_mesh], fixed_topology=True)
        if debug_print:
            print(f'\nFixed meshes {fixed_meshes}')

        # Load another mesh and convert it to any batching strategy
        pizza_mesh = usd.import_mesh(data_path('pizza.usd'), **import_args)
        input_meshes.append(copy.deepcopy(pizza_mesh))
        pizza_mesh.set_batching(batching)

        # Now, let's concatenate all of these together
        mesh = SurfaceMesh.cat([amsterdam, fixed_meshes, pizza_mesh], fixed_topology=False)
        if debug_print:
            print(f'\nTotal mesh {mesh}')
        expected_len = num_amsterdam + 3
        assert len(mesh) == expected_len
        expected_attrs = {'vertices', 'faces', 'uvs', 'face_normals', 'face_uvs_idx', 'material_assignments'}
        assert set(mesh.get_attributes(only_tensors=True)) == expected_attrs
        assert mesh.has_attribute('materials')
        assert len(mesh.materials) == expected_len
        for attr in expected_attrs:
            assert len(getattr(mesh, attr)) == expected_len
        for i in range(len(mesh)):
            for attr in expected_attrs:
                assert torch.allclose(getattr(mesh, attr)[i], getattr(input_meshes[i], attr))

        # Now try concatenating with fixed topology
        with pytest.raises(ValueError):
            mesh = SurfaceMesh.cat([amsterdam, fixed_meshes, pizza_mesh], fixed_topology=True)

        # Will fail even with skip_errors, as faces and vertices must be concatenatable
        with pytest.raises(ValueError):
            mesh = SurfaceMesh.cat([amsterdam, fixed_meshes, pizza_mesh], fixed_topology=True, skip_errors=True)

    def test_can_compute_attribute(self, device, batching):
        input_attr = make_default_input(device, batching)

        mesh = SurfaceMesh(vertices=input_attr['vertices'],
                           faces=input_attr['faces'])
        for existing in ['vertices', 'faces']:
            assert mesh.has_attribute(existing)
            assert mesh.has_or_can_compute_attribute(existing)
            assert existing in mesh.get_attributes()
        for computable in ['face_normals', 'face_vertices', 'vertex_normals']:
            assert not mesh.has_attribute(computable), computable
            assert mesh.probably_can_compute_attribute(computable), computable
            assert mesh.has_or_can_compute_attribute(computable), computable
        for missing in ['uvs', 'normals', 'face_uvs', 'face_uvs_idx', 'face_normals_idx']:
            assert not mesh.has_attribute(missing), missing
            assert not mesh.probably_can_compute_attribute(missing), missing
            assert not mesh.has_or_can_compute_attribute(missing), missing
