# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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

"""Export common 3D representations to USD format. Mesh (homogeneous), Voxelgrid
and PointCloud representations are currently supported.
"""

import itertools
import os
import re
import warnings
from collections import namedtuple
import numpy as np

import torch

try:
    from pxr import Usd, UsdGeom, Vt, Sdf, UsdShade
except ImportError:
    warnings.warn('Warning: module pxr not found', ImportWarning)

from kaolin.io import materials as usd_materials
from kaolin.io import utils

mesh_return_type = namedtuple('mesh_return_type', ['vertices', 'faces',
                                                   'uvs', 'face_uvs_idx', 'face_normals', 'materials_order',
                                                   'materials'])
pointcloud_return_type = namedtuple('pointcloud_return_type', ['points', 'colors', 'normals'])

class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.NonHomogeneousMeshError`.

    Attributes:
        message (str)
    """
    __slots__ = ['message']

    def __init__(self, message):
        warnings.warn("NonHomogeneousMeshError is deprecated, "
                      "please use kaolin.io.utils.NonHomogeneousMeshError instead", 
                      DeprecationWarning, stacklevel=2)
        self.message = message

def _get_stage_next_free_path(stage, scene_path):
    """Get next free path in the stage."""
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        scene_path = default_prim.GetPath().pathString + scene_path

    match = re.search(r'_(\d+)$', scene_path)
    i = int(match.group(1)) if match else 0

    while stage.GetPrimAtPath(scene_path):
        scene_path = f'{scene_path}_{i:02d}'

    return scene_path


def _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time):
    """Return mesh attributes flattened into a single mesh."""
    stage_dir = os.path.dirname(str(stage.GetRootLayer().realPath))
    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')

    attrs = {}

    def _process_mesh(mesh_prim, ref_path, attrs):
        cur_first_idx_faces = sum([len(v) for v in attrs.get('vertices', [])])
        cur_first_idx_uvs = sum([len(u) for u in attrs.get('uvs', [])])
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh_vertices = mesh.GetPointsAttr().Get(time=time)
        mesh_face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time=time)
        mesh_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time=time)
        mesh_st = mesh.GetPrimvar('st')
        mesh_subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
        mesh_material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]

        # Parse mesh UVs
        if mesh_st:
            mesh_uvs = mesh_st.Get(time=time)
            mesh_uv_indices = mesh_st.GetIndices(time=time)
            mesh_uv_interpolation = mesh_st.GetInterpolation()
        mesh_face_normals = mesh.GetNormalsAttr().Get(time=time)

        # Parse mesh geometry
        if mesh_vertices:
            attrs.setdefault('vertices', []).append(torch.from_numpy(np.array(mesh_vertices, dtype=np.float32)))
        if mesh_vertex_indices:
            attrs.setdefault('face_vertex_counts', []).append(torch.from_numpy(
                np.array(mesh_face_vertex_counts, dtype=np.int64)))
            vertex_indices = torch.from_numpy(np.array(mesh_vertex_indices, dtype=np.int64)) + cur_first_idx_faces
            attrs.setdefault('vertex_indices', []).append(vertex_indices)
        if with_normals and mesh_face_normals:
            attrs.setdefault('face_normals', []).append(torch.from_numpy(np.array(mesh_face_normals, dtype=np.float32)))
        if mesh_st and mesh_uvs:
            attrs.setdefault('uvs', []).append(torch.from_numpy(np.array(mesh_uvs, dtype=np.float32)))
            if mesh_uv_interpolation in ['vertex', 'varying']:
                if not mesh_uv_indices:
                    # for vertex and varying interpolation, length of mesh_uv_indices should match
                    # length of mesh_vertex_indices
                    mesh_uv_indices = list(range(len(mesh_uvs)))
                mesh_uv_indices = torch.tensor(mesh_uv_indices) + cur_first_idx_uvs
                face_uvs_idx = mesh_uv_indices[torch.from_numpy(np.array(mesh_vertex_indices, dtype=np.int64))]
                attrs.setdefault('face_uvs_idx', []).append(face_uvs_idx)
            elif mesh_uv_interpolation == 'faceVarying':
                if not mesh_uv_indices:
                    # for faceVarying interpolation, length of mesh_uv_indices should match
                    # num_faces * face_size
                    # TODO implement default behaviour
                    mesh_uv_indices = [i for i, c in enumerate(mesh_face_vertex_counts) for _ in range(c)]
                else:
                    attrs.setdefault('face_uvs_idx', []).append(torch.tensor(mesh_uv_indices) + cur_first_idx_uvs)
            # elif mesh_uv_interpolation == 'uniform':
            else:
                raise NotImplementedError(f'Interpolation type {mesh_uv_interpolation} is '
                                          'not currently supported')

        # Parse mesh materials
        if with_materials:
            subset_idx_map = {}
            attrs.setdefault('materials', []).append(None)
            attrs.setdefault('material_idx_map', {})
            if mesh_material:
                mesh_material_path = str(mesh_material.GetPath())
                if mesh_material_path in attrs['material_idx_map']:
                    material_idx = attrs['material_idx_map'][mesh_material_path]
                else:
                    try:
                        material = usd_materials.MaterialManager.read_usd_material(mesh_material, stage_dir, time)
                        material_idx = len(attrs['materials'])
                        attrs['materials'].append(material)
                        attrs['material_idx_map'][mesh_material_path] = material_idx
                    except usd_materials.MaterialNotSupportedError as e:
                        warnings.warn(e.args[0])
                    except usd_materials.MaterialReadError as e:
                        warnings.warn(e.args[0])
            if mesh_subsets:
                for subset in mesh_subsets:
                    subset_material, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
                    subset_material_metadata = subset_material.GetPrim().GetMetadata('references')
                    mat_ref_path = ""
                    if ref_path:
                        mat_ref_path = ref_path
                    if subset_material_metadata:
                        asset_path = subset_material_metadata.GetAddedOrExplicitItems()[0].assetPath
                        mat_ref_path = os.path.join(ref_path, os.path.dirname(asset_path))
                    if not os.path.isabs(mat_ref_path):
                        mat_ref_path = os.path.join(stage_dir, mat_ref_path)
                    try:
                        kal_material = usd_materials.MaterialManager.read_usd_material(subset_material, mat_ref_path,
                                                                                       time)
                    except usd_materials.MaterialNotSupportedError as e:
                        warnings.warn(e.args[0])
                        continue
                    except usd_materials.MaterialReadError as e:
                        warnings.warn(e.args[0])

                    subset_material_path = str(subset_material.GetPath())
                    if subset_material_path not in attrs['material_idx_map']:
                        attrs['material_idx_map'][subset_material_path] = len(attrs['materials'])
                        attrs['materials'].append(kal_material)
                    subset_indices = np.array(subset.GetIndicesAttr().Get())
                    subset_idx_map[attrs['material_idx_map'][subset_material_path]] = subset_indices
            # Create material face index list
            if mesh_face_vertex_counts:
                for face_idx in range(len(mesh_face_vertex_counts)):
                    is_in_subsets = False
                    for subset_idx in subset_idx_map:
                        if face_idx in subset_idx_map[subset_idx]:
                            is_in_subsets = True
                            attrs.setdefault('materials_face_idx', []).extend(
                                [subset_idx] * mesh_face_vertex_counts[face_idx]
                            )
                    if not is_in_subsets:
                        if mesh_material:
                            attrs.setdefault('materials_face_idx', []).extend(
                                [material_idx] * mesh_face_vertex_counts[face_idx]
                            )
                        else:
                            # Assign to `None` material (ie. index 0)
                            attrs.setdefault('materials_face_idx', []).extend([0] * mesh_face_vertex_counts[face_idx])

    def _traverse(cur_prim, ref_path, attrs):
        metadata = cur_prim.GetMetadata('references')
        if metadata:
            ref_path = os.path.dirname(metadata.GetAddedOrExplicitItems()[0].assetPath)
        if UsdGeom.Mesh(cur_prim):
            _process_mesh(cur_prim, ref_path, attrs)
        for child in cur_prim.GetChildren():
            _traverse(child, ref_path, attrs)

    _traverse(stage.GetPrimAtPath(scene_path), '', attrs)

    if not attrs.get('vertices'):
        warnings.warn(f'Scene object at {scene_path} contains no vertices.', UserWarning)

    # Only import vertices if they are defined for the entire mesh
    if all([v is not None for v in attrs.get('vertices', [])]) and len(attrs.get('vertices', [])) > 0:
        attrs['vertices'] = torch.cat(attrs.get('vertices'))
    else:
        attrs['vertices'] = None
    # Only import vertex index and counts if they are defined for the entire mesh
    if all([vi is not None for vi in attrs.get('vertex_indices', [])]) and len(attrs.get('vertex_indices', [])) > 0:
        attrs['face_vertex_counts'] = torch.cat(attrs.get('face_vertex_counts', []))
        attrs['vertex_indices'] = torch.cat(attrs.get('vertex_indices', []))
    else:
        attrs['face_vertex_counts'] = None
        attrs['vertex_indices'] = None
    # Only import UVs if they are defined for the entire mesh
    if not all([uv is not None for uv in attrs.get('uvs', [])]) or len(attrs.get('uvs', [])) == 0:
        if len(attrs.get('uvs', [])) > 0:
            warnings.warn('UVs are missing for some child meshes for prim at '
                          f'{scene_path}. As a result, no UVs were imported.')
        attrs['uvs'] = None
        attrs['face_uvs_idx'] = None
    else:
        attrs['uvs'] = torch.cat(attrs['uvs'])
        if attrs.get('face_uvs_idx', None):
            attrs['face_uvs_idx'] = torch.cat(attrs['face_uvs_idx'])
        else:
            attrs['face_uvs_idx'] = None

    # Only import face_normals if they are defined for the entire mesh
    if not all([n is not None for n in attrs.get('face_normals', [])]) or len(attrs.get('face_normals', [])) == 0:
        if len(attrs.get('face_normals', [])) > 0:
            warnings.warn('Face normals are missing for some child meshes for '
                          f'prim at {scene_path}. As a result, no Face Normals were imported.')
        attrs['face_normals'] = None
    else:
        attrs['face_normals'] = torch.cat(attrs['face_normals'])

    if attrs.get('materials_face_idx') is None or max(attrs.get('materials_face_idx', [])) == 0:
        attrs['materials_face_idx'] = None
    else:
        attrs['materials_face_idx'] = torch.LongTensor(attrs['materials_face_idx'])

    if all([m is None for m in attrs.get('materials', [])]):
        attrs['materials'] = None
    return attrs


def get_root(file_path):
    r"""Return the root prim scene path.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

    Returns:
        (str): Root scene path.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve root scene path
        >>> root_prim = get_root('./new_stage.usd')
        >>> mesh = import_mesh('./new_stage.usd', root_prim)
        >>> mesh.vertices.shape
        torch.Size([9, 3])
        >>> mesh.faces.shape
        torch.Size([3, 3])
    """
    stage = Usd.Stage.Open(file_path)
    return stage.GetPseudoRoot().GetPath()


def get_pointcloud_scene_paths(file_path):
    r"""Returns all point cloud scene paths contained in specified file. Assumes that point
    clouds are exported using this API.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

    Returns:
        (list of str): List of filtered scene paths.
    """
    # TODO(mshugrina): is passing prim_types='PointInstancer' the same as UsdGeom.PointInstancer(p) ?
    geom_points_paths = get_scene_paths(file_path, prim_types=['Points'])
    point_instancer_paths = get_scene_paths(file_path, prim_types=['PointInstancer'])
    return geom_points_paths + point_instancer_paths


def get_scene_paths(file_path, scene_path_regex=None, prim_types=None, conditional=lambda x: True):
    r"""Return all scene paths contained in specified file. Filter paths with regular
    expression in `scene_path_regex` if provided.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path_regex (str, optional): Optional regular expression used to select returned scene paths.
        prim_types (list of str, optional): Optional list of valid USD Prim types used to
            select scene paths.
        conditional (function path: Bool): Custom conditionals to check

    Returns:
        (list of str): List of filtered scene paths.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve scene paths
        >>> get_scene_paths('./new_stage.usd', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0'), Sdf.Path('/World/Meshes/mesh_1'), Sdf.Path('/World/Meshes/mesh_2')]
        >>> get_scene_paths('./new_stage.usd', scene_path_regex=r'.*_0', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0')]
    """
    stage = Usd.Stage.Open(file_path)
    if scene_path_regex is None:
        scene_path_regex = '.*'
    if prim_types is not None:
        prim_types = [pt.lower() for pt in prim_types]

    scene_paths = []
    for p in stage.Traverse():
        is_valid_prim_type = prim_types is None or p.GetTypeName().lower() in prim_types
        is_valid_scene_path = re.match(scene_path_regex, str(p.GetPath()))
        passes_conditional = conditional(p)
        if is_valid_prim_type and is_valid_scene_path and passes_conditional:
            scene_paths.append(p.GetPath())
    return scene_paths


def get_authored_time_samples(file_path):
    r"""
    Returns *all* authored time samples within the USD, aggregated across all primitives.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

    Returns:
        (list)
    """
    stage = Usd.Stage.Open(file_path)
    scene_paths = get_scene_paths(file_path)
    res = set()
    for scene_path in scene_paths:
        prim = stage.GetPrimAtPath(scene_path)
        attr = prim.GetAttributes()
        res.update(set(itertools.chain.from_iterable([x.GetTimeSamples() for x in attr])))
    return sorted(res)


def create_stage(file_path, up_axis='Y'):
    r"""Create a new USD file and return an empty stage.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        up_axis (['Y', 'Z']): Specify the stage up axis. Choose from ``['Y', 'Z']``.

    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd', up_axis='Z')
        >>> type(stage)
        <class 'pxr.Usd.Stage'>
    """
    assert os.path.exists(os.path.dirname(file_path)), f'Directory {os.path.dirname(file_path)} not found.'
    stage = Usd.Stage.CreateNew(str(file_path))
    world = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    return stage


# Mesh Functions
# TODO(jlafleche) Support mesh subgroups for materials
def heterogeneous_mesh_handler_skip(*args):
    r"""Skip heterogeneous meshes.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.heterogeneous_mesh_handler_skip`.

    """
    warnings.warn("heterogeneous_mesh_handler_skip is deprecated, "
                  "please use kaolin.io.utils.heterogeneous_mesh_handler_skip instead", 
                  DeprecationWarning, stacklevel=2)

    return None

def heterogeneous_mesh_handler_empty(*args):
    """Return empty tensors for vertices and faces of heterogeneous meshes.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.heterogeneous_mesh_handler_empty`.

    """
    warnings.warn("heterogeneous_mesh_handler_empty is deprecated, "
                  "please use kaolin.io.utils.heterogeneous_mesh_handler_empty instead", 
                  DeprecationWarning, stacklevel=2)

    return (torch.FloatTensor(size=(0, 3)), torch.LongTensor(size=(0,)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 2)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 3, 3)),
            torch.LongTensor(size=(0,)))


def heterogeneous_mesh_handler_naive_homogenize(vertices, face_vertex_counts, *args):
    r"""Homogenize list of faces containing polygons of varying number of edges to triangles using fan
    triangulation.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.heterogeneous_mesh_handler_naive_homogenize`.

    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        *args: Variable length features that need to be handled. For example, faces and uvs.

    Returns:
        (list of torch.tensor): Homogeneous list of attributes.
    """

    warnings.warn("heterogeneous_mesh_handler_naive_homogenize is deprecated, "
                  "please use kaolin.io.utils.heterogeneous_mesh_handler_naive_homogenize instead", 
                  DeprecationWarning, stacklevel=2)

    def _homogenize(attr, face_vertex_counts):
        if attr is not None:
            attr = attr.tolist()
            idx = 0
            new_attr = []
            for face_vertex_count in face_vertex_counts:
                attr_face = attr[idx:(idx + face_vertex_count)]
                idx += face_vertex_count
                while len(attr_face) >= 3:
                    new_attr.append(attr_face[:3])
                    attr_face.pop(1)
            return torch.tensor(new_attr)
        else:
            return None

    new_attrs = [_homogenize(a, face_vertex_counts) for a in args]
    new_counts = torch.ones(vertices.size(0), dtype=torch.long).fill_(3)
    return (vertices, new_counts, *new_attrs)


def import_mesh(file_path, scene_path=None, with_materials=False, with_normals=False,
                heterogeneous_mesh_handler=None, time=None):
    r"""Import a single mesh from a USD file in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face).
    All sub-meshes found under the `scene_path` are flattened to a single mesh. The following
    interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    Returns an unbatched representation.

    Args:
        file_path (str): Path to usd file (`\*.usd`, `\*.usda`).
        scene_path (str, optional): Scene path within the USD file indicating which primitive to import.
            If not specified, the all meshes in the scene will be imported and flattened into a single mesh.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes.
            The function's input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
            ``uvs`` (torch.FloatTensor), ``face_uvs_idx`` (torch.LongTensor), and ``face_normals`` (torch.FloatTensor).
            If the function returns ``None``, the mesh will be skipped. If no function is specified,
            an error will be raised when attempting to import a heterogeneous mesh.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

    Returns:

    namedtuple of:
        - **vertices** (torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties

    Example:
        >>> # Create a stage with some meshes
        >>> stage = export_mesh('./new_stage.usd', vertices=torch.rand(3, 3), faces=torch.tensor([[0, 1, 2]]),
        ... scene_path='/World/mesh1')
        >>> # Import meshes
        >>> mesh = import_mesh(file_path='./new_stage.usd', scene_path='/World/mesh1')
        >>> mesh.vertices.shape
        torch.Size([3, 3])
        >>> mesh.faces
        tensor([[0, 1, 2]])
    """
    # TODO  add arguments to selectively import UVs and normals
    if scene_path is None:
        scene_path = get_root(file_path)
    if time is None:
        time = Usd.TimeCode.Default()
    meshes_list = import_meshes(file_path, [scene_path],
                                heterogeneous_mesh_handler=heterogeneous_mesh_handler, with_materials=with_materials,
                                with_normals=with_normals, times=[time])
    return mesh_return_type(*meshes_list[0])


def import_meshes(file_path, scene_paths=None, with_materials=False, with_normals=False,
                  heterogeneous_mesh_handler=None, times=None):
    r"""Import one or more meshes from a USD file in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face). Custom handling of
    heterogeneous meshes can be achieved by passing a function through the ``heterogeneous_mesh_handler`` argument.
    The following interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    For each scene path specified in `scene_paths`, sub-meshes (if any) are flattened to a single mesh.
    Returns unbatched meshes list representation. Prims with no meshes or with heterogenous faces are skipped.

    Args:
        file_path (str): Path to usd file (`\*.usd`, `\*.usda`).
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, all prims of type `Mesh` will be imported.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes.
            The function's input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
            ``uvs`` (torch.FloatTensor), ``face_uvs_idx`` (torch.LongTensor), and ``face_normals`` (torch.FloatTensor).
            If the function returns ``None``, the mesh will be skipped. If no function is specified,
            an error will be raised when attempting to import a heterogeneous mesh.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:

    list of namedtuple of:
        - **vertices** (list of torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (list of torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (list of torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (list of torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (list of torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Import meshes
        >>> meshes = import_meshes(file_path='./new_stage.usd')
        >>> len(meshes)
        3
        >>> meshes[0].vertices.shape
        torch.Size([3, 3])
        >>> [m.faces for m in meshes]
        [tensor([[0, 1, 2]]), tensor([[0, 1, 2]]), tensor([[0, 1, 2]])]
    """
    # TODO  add arguments to selectively import UVs and normals
    assert os.path.exists(file_path)
    stage = Usd.Stage.Open(file_path)
    if scene_paths is None:
        scene_paths = get_scene_paths(file_path, prim_types=['Mesh'])
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    vertices_list, faces_list, uvs_list, face_uvs_idx_list, face_normals_list = [], [], [], [], []
    materials_order_list, materials_list = [], []
    for scene_path, time in zip(scene_paths, times):
        mesh_attr = _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time=time)
        vertices = mesh_attr['vertices']
        face_vertex_counts = mesh_attr['face_vertex_counts']
        faces = mesh_attr['vertex_indices']
        uvs = mesh_attr['uvs']
        face_uvs_idx = mesh_attr['face_uvs_idx']
        face_normals = mesh_attr['face_normals']
        materials_face_idx = mesh_attr['materials_face_idx']
        materials = mesh_attr['materials']
        # TODO(jlafleche) Replace tuple output with mesh class

        if faces is not None:
            if not torch.all(face_vertex_counts == face_vertex_counts[0]):
                if heterogeneous_mesh_handler is None:
                    raise utils.NonHomogeneousMeshError(f'Mesh at {scene_path} is non-homogeneous '
                                                        f'and cannot be imported from {file_path}.')
                else:
                    mesh = heterogeneous_mesh_handler(vertices, face_vertex_counts, faces, uvs,
                                                      face_uvs_idx, face_normals, materials_face_idx)
                    if mesh is None:
                        continue
                    else:
                        vertices, face_vertex_counts, faces, uvs, face_uvs_idx, face_normals, materials_face_idx = mesh
            if faces.size(0) > 0:
                faces = faces.view(-1, face_vertex_counts[0])

        if face_uvs_idx is not None and faces is not None and face_uvs_idx.size(0) > 0:
            uvs = uvs.reshape(-1, 2)
            face_uvs_idx = face_uvs_idx.reshape(-1, faces.size(1))
        if face_normals is not None and faces is not None and face_normals.size(0) > 0:
            face_normals = face_normals.reshape(-1, faces.size(1), 3)
        if faces is not None and materials_face_idx is not None:            # Create material order list
            materials_face_idx.view(-1, faces.size(1))
            cur_mat_idx = -1
            materials_order = []
            for idx in range(len(materials_face_idx)):
                mat_idx = materials_face_idx[idx][0].item()
                if cur_mat_idx != mat_idx:
                    cur_mat_idx = mat_idx
                    materials_order.append([idx, mat_idx])
        else:
            materials_order = None

        vertices_list.append(vertices)
        faces_list.append(faces)
        uvs_list.append(uvs)
        face_uvs_idx_list.append(face_uvs_idx)
        face_normals_list.append(face_normals)
        materials_order_list.append(materials_order)
        materials_list.append(materials)

    params = [vertices_list, faces_list, uvs_list, face_uvs_idx_list,
              face_normals_list, materials_order_list, materials_list]
    return [mesh_return_type(v, f, uv, fuv, fn, mo, m) for v, f, uv, fuv, fn, mo, m in zip(*params)]


def add_mesh(stage, scene_path, vertices=None, faces=None, uvs=None, face_uvs_idx=None, face_normals=None,
             materials_order=None, materials=None, time=None):
    r"""Add a mesh to an existing USD stage.

    Add a mesh to the USD stage. The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the mesh.
        scene_path (str): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        materials_order (torch.LongTensor): of shape (N, 2)
          showing the order in which materials are used over **face_uvs_idx** and the first indices
          in which they start to be used. A material can be used multiple times.
        materials (list of Material): a list of materials
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = create_stage('./new_stage.usd')
        >>> mesh = add_mesh(stage, '/World/mesh', vertices, faces)
        >>> stage.Save()
    """
    if time is None:
        time = Usd.TimeCode.Default()

    usd_mesh = UsdGeom.Mesh.Define(stage, scene_path)

    if faces is not None:
        num_faces = faces.size(0)
        face_vertex_counts = [faces.size(1)] * num_faces
        faces_list = faces.view(-1).cpu().long().numpy()
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces_list, time=time)
    if vertices is not None:
        vertices_list = vertices.detach().cpu().float().numpy()
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices_list), time=time)
    if uvs is not None:
        uvs_list = uvs.view(-1, 2).detach().cpu().float().numpy()
        pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            'st', Sdf.ValueTypeNames.Float2Array)
        pv.Set(uvs_list, time=time)

        if vertices is not None and uvs.size(0) == vertices.size(0):
            pv.SetInterpolation('vertex')
        elif faces is not None and uvs.view(-1, 2).size(0) == faces.size(0):
            pv.SetInterpolation('uniform')
        else:
            pv.SetInterpolation('faceVarying')

    if face_uvs_idx is not None:
        if uvs is not None:
            pv.SetIndices(Vt.IntArray.FromNumpy(face_uvs_idx.view(-1).cpu().long().numpy()), time=time)
        else:
            warnings.warn('If providing "face_uvs_idx", "uvs" must also be provided.')

    if face_normals is not None:
        face_normals = face_normals.view(-1, 3).cpu().float().numpy()
        usd_mesh.GetNormalsAttr().Set(face_normals, time=time)
        UsdGeom.PointBased(usd_mesh).SetNormalsInterpolation('faceVarying')

    if faces is not None and materials_order is not None and materials is not None:
        stage.DefinePrim(f'{scene_path}/Looks', 'Scope')
        subsets = {}
        for i in range(len(materials_order)):
            first_face_idx, mat_idx = materials_order[i]
            if materials[mat_idx] is None:
                continue
            last_face_idx = materials_order[i + 1][0] if (i + 1) < len(materials_order) else faces.size(0)
            for face_idx in range(first_face_idx, last_face_idx):
                subsets.setdefault(mat_idx, []).append(face_idx)

        # Create submeshes
        for i, subset in enumerate(subsets):
            subset_prim = stage.DefinePrim(f'{scene_path}/subset_{i}', 'GeomSubset')
            subset_prim.GetAttribute('indices').Set(subsets[subset])
            materials[subset]._write_usd_preview_surface(stage, f'{scene_path}/Looks/material_{subset}',
                                                         [subset_prim], time, texture_dir=f'material_{subset}',
                                                         texture_file_prefix='')    # TODO file path

    return usd_mesh.GetPrim()


def export_mesh(file_path, scene_path='/World/Meshes/mesh_0', vertices=None, faces=None,
                uvs=None, face_uvs_idx=None, face_normals=None, materials_order=None, materials=None,
                up_axis='Y', time=None):
    r"""Export a single mesh to USD.

    Export a single mesh defined by vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str, optional): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
            If no path is provided, a default path is used.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        materials_order (torch.LongTensor): of shape (N, 2)
          showing the order in which materials are used over **face_uvs_idx** and the first indices
          in which they start to be used. A material can be used multiple times.
        materials (list of Material): a list of materials
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
       (Usd.Stage)

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = export_mesh('./new_stage.usd', vertices=vertices, faces=faces)
    """
    assert isinstance(scene_path, str)
    if time is None:
        time = Usd.TimeCode.Default()
    if os.path.exists(file_path):
        stage = Usd.Stage.Open(file_path)
        UsdGeom.SetStageUpAxis(stage, up_axis)
    else:
        stage = create_stage(file_path, up_axis)
    add_mesh(stage, scene_path, vertices, faces, uvs, face_uvs_idx, face_normals, materials_order, materials, time=time)
    stage.Save()

    return stage


def export_meshes(file_path, scene_paths=None, vertices=None, faces=None,
                  uvs=None, face_uvs_idx=None, face_normals=None, materials_order=None, materials=None,
                  up_axis='Y', times=None):
    r"""Export multiple meshes to a new USD stage.

    Export multiple meshes defined by lists vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Absolute paths of meshes within the USD file scene. Must have the same
            number ofpaths as the number of meshes ``N``. Must be a valid Sdf.Path. If no path is provided, a default
            path is used.
        vertices (list of torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (list of torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (list of torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (list of torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs`
            must also be specified.
        face_normals (list of torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        materials_order (torch.LongTensor): of shape (N, 2)
          showing the order in which materials are used over **face_uvs_idx** and the first indices
          in which they start to be used. A material can be used multiple times.
        materials (list of Material): a list of materials
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        times (list of int, optional): Positive integers defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
    """
    stage = create_stage(file_path, up_axis)
    mesh_parameters = {'vertices': vertices, 'faces': faces, 'uvs': uvs,
                       'face_uvs_idx': face_uvs_idx, 'face_normals': face_normals,
                       'materials_order': materials_order, 'materials': materials}
    supplied_parameters = {k: p for k, p in mesh_parameters.items() if p is not None}
    length = len(list(supplied_parameters.values())[0])
    assert all([len(p) == length for p in supplied_parameters.values()])
    if scene_paths is None:
        if not stage.GetPrimAtPath('/World/Meshes'):
            stage.DefinePrim('/World/Meshes', 'Xform')
        scene_paths = [f'/World/Meshes/mesh_{i}' for i in range(len(vertices))]
    assert len(scene_paths) == length
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    for i, scene_path in enumerate(scene_paths):
        mesh_params = {k: p[i] for k, p in supplied_parameters.items()}
        add_mesh(stage, scene_path, **mesh_params)
    stage.Save()

    return stage


# Pointcloud functions
def import_pointcloud(file_path, scene_path, time=None):
    r"""Import a single pointcloud from a USD file.

    Assumes that the USD pointcloud is interpreted using a point instancer or UsdGeomPoints. Converts the coordinates
    of each point instance to a point within the output pointcloud.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str): Scene path within the USD file indicating which primitive to import.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.
    Returns:
        namedtuple of:
            - **points** (torch.FloatTensor): of shape (num_points, 3)
            - **colors** (torch.FloatTensor): of shape (num_points, 3)
            - **normals** (torch.FloatTensor): of shape (num_points, 3) (not yet implemented)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points, scene_path='/World/pointcloud')
        >>> points_imp = import_pointcloud(file_path='./new_stage.usd',
        ...                                scene_path='/World/pointcloud')[0]
        >>> points_imp.shape
        torch.Size([100, 3])
    """
    if time is None:
        time = Usd.TimeCode.Default()

    pointcloud_list = import_pointclouds(file_path, [scene_path], times=[time])

    return pointcloud_return_type(*pointcloud_list[0])


def import_pointclouds(file_path, scene_paths=None, times=None):
    r"""Import one or more pointclouds from a USD file.

    Assumes that pointclouds are interpreted using point instancers or UsdGeomPoints. Converts the coordinates
    of each point instance to a point within the output pointcloud.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, will return all pointclouds found based on PointInstancer or UsdGeomPoints prims with
            `kaolin_type` primvar set to `PointCloud`.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:
        list of namedtuple of:
            - **points** (list of torch.FloatTensor): of shape (num_points, 3)
            - **colors** (list of torch.FloatTensor): of shape (num_points, 3)
            - **normals** (list of torch.FloatTensor): of shape (num_points, 2) (not yet implemented)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointclouds('./new_stage.usd', [points, points, points])
        >>> pointclouds = import_pointclouds(file_path='./new_stage.usd')[0]
        >>> len(pointclouds)
        3
        >>> pointclouds[0].shape
        torch.Size([100, 3])
    """
    assert os.path.exists(file_path)

    if scene_paths is None:
        scene_paths = get_pointcloud_scene_paths(file_path)
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    pointclouds = []
    colors = []
    normals = []
    stage = Usd.Stage.Open(file_path)
    for scene_path, time in zip(scene_paths, times):
        prim = stage.GetPrimAtPath(scene_path)
        assert prim, f'The prim at {scene_path} does not exist.'

        if UsdGeom.Points(prim):
            geom_points = UsdGeom.Points(prim)
            pointclouds.append(torch.tensor(geom_points.GetPointsAttr().Get(time=time)))

            color = geom_points.GetDisplayColorAttr().Get(time=time)

            if color is None:
                colors.append(color)
            else:
                colors.append(torch.tensor(color))
        elif UsdGeom.PointInstancer(prim):
            instancer = UsdGeom.PointInstancer(prim)
            pointclouds.append(torch.tensor(instancer.GetPositionsAttr().Get(time=time)))
            colors.append(None)
        else:
            raise TypeError('The prim is neither UsdGeomPoints nor UsdGeomPointInstancer.')

    # TODO: place holders for normals for now
    normals = [None] * len(colors)

    params = [pointclouds, colors, normals]
    return [pointcloud_return_type(p, c, n) for p, c, n in zip(*params)]


def get_pointcloud_bracketing_time_samples(stage, scene_path, target_time):
    """Returns two time samples that bracket ``target_time`` for point cloud
    attributes at a specified scene_path.

    Args:
        stage (Usd.Stage)
        scene_path (str)
        target_time (Number)

    Returns:
        (iterable of 2 numbers)
    """
    # Note: can also get usd_attr.GetTimeSamples()
    prim = stage.GetPrimAtPath(scene_path)

    if UsdGeom.Points(prim):
        geom_points = UsdGeom.Points(prim)
        result = geom_points.GetPointsAttr().GetBracketingTimeSamples(target_time)
    elif UsdGeom.PointInstancer(prim):
        instancer = UsdGeom.PointInstancer(prim)
        result = instancer.GetPositionsAttr().GetBracketingTimeSamples(target_time)
    else:
        raise TypeError('The prim is neither UsdGeomPoints nor UsdGeomPointInstancer.')
    return result


def add_pointcloud(stage, points, scene_path, colors=None, time=None, points_type='point_instancer'):
    r"""Add a pointcloud to an existing USD stage.

    Create a pointcloud represented by point instances of a sphere centered at each point coordinate.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the pointcloud.
        points (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        scene_path (str): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
        colors (torch.FloatTensor, optional): Color tensor corresponding each point in the pointcloud
            tensor of shape ``(N, 3)``. colors only works if points_type is 'usd_geom_points'.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
            'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
            Please refer here for UsdGeomPoints:
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> points = torch.rand(100, 3)
        >>> stage = add_pointcloud(stage, points, '/World/PointClouds/pointcloud_0')
        >>> stage.Save()
    """
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    if stage.GetPrimAtPath(scene_path):
        points_prim = stage.GetPrimAtPath(scene_path)
    else:
        if points_type == 'point_instancer':
            points_prim = stage.DefinePrim(scene_path, 'PointInstancer')
        elif points_type == 'usd_geom_points':
            points_prim = stage.DefinePrim(scene_path, 'Points')
        else:
            raise ValueError('Expected points_type to be "usd_geom_points" or "point_instancer", '
                             f'but got "{points_type}".')

    if points_type == 'point_instancer':
        geom_points = UsdGeom.PointInstancer(points_prim)
        sphere = UsdGeom.Sphere.Define(stage, f'{scene_path}/sphere')
        sphere.GetRadiusAttr().Set(0.5)
        geom_points.CreatePrototypesRel().SetTargets([sphere.GetPath()])
    elif points_type == 'usd_geom_points':
        geom_points = UsdGeom.Points(points_prim)

    # Calculate default point scale
    bounds = points.max(dim=0)[0] - points.min(dim=0)[0]
    min_bound = min(bounds)
    scale = (min_bound / points.size(0) ** (1 / 3)).item()

    # Generate instancer parameters
    positions = points.detach().cpu().tolist()
    scales = np.asarray([scale, ] * points.size(0))

    if points_type == 'point_instancer':
        indices = [0] * points.size(0)
        # Populate point instancer
        geom_points.GetProtoIndicesAttr().Set(indices, time=time)
        geom_points.GetPositionsAttr().Set(positions, time=time)
        scales = [(scale,) * 3] * points.size(0)
        geom_points.GetScalesAttr().Set(scales, time=time)
    elif points_type == 'usd_geom_points':
        # Populate UsdGeomPoints
        geom_points.GetPointsAttr().Set(points.numpy(), time=time)
        geom_points.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(scales), time=time)

    if colors is not None and points_type == 'usd_geom_points':
        assert colors.shape == points.shape, 'Colors and points must have the same shape.'
        geom_points.GetDisplayColorAttr().Set(colors.numpy(), time=time)

    return stage


def export_pointcloud(file_path, pointcloud, scene_path='/World/PointClouds/pointcloud_0',
                      color=None, time=None, points_type='point_instancer'):
    r"""Export a single pointcloud to a USD scene.

    Export a single pointclouds to USD. The pointcloud will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        pointcloud (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        scene_path (str, optional): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        color (torch.FloatTensor, optional): Color tensor corresponding each point in the pointcloud
            tensor of shape ``(N, 3)``. colors only works if points_type is 'usd_geom_points'.
        time (convertible to float): Positive integer defining the time at which the supplied parameters correspond to.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
                'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
                Please refer here for UsdGeomPoints:
                https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
                https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
    Returns:
        (Usd.Stage)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points)
    """
    stage = export_pointclouds(file_path, [pointcloud], [scene_path], colors=[color], times=[time],
                               points_type=points_type)
    return stage


def export_pointclouds(file_path, pointclouds, scene_paths=None, colors=None, times=None,
                       points_type='point_instancer'):
    r"""Export one or more pointclouds to a USD scene.

    Export one or more pointclouds to USD. The pointclouds will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        pointclouds (list of torch.FloatTensor): List of pointcloud tensors of length ``N`` defining N pointclouds.
        scene_paths (list of str, optional): Absolute path(s) of pointcloud(s) within the USD file scene.
            Must be a valid Sdf.Path. If no path is provided, a default path is used.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
        colors (list of tensors, optional): Lits of RGB colors of length ``N``, each corresponding to a pointcloud
            in the pointcloud list. colors only works if points_type is 'usd_geom_points'.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
            'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
            Please refer here for UsdGeomPoints:
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
    Returns:
        (Usd.Stage)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points)
    """
    if scene_paths is None:
        scene_paths = [f'/World/PointClouds/pointcloud_{i}' for i in range(len(pointclouds))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)
    if colors is None:
        colors = [None] * len(scene_paths)

    assert len(pointclouds) == len(scene_paths)
    stage = create_stage(file_path)
    for scene_path, points, color, time in zip(scene_paths, pointclouds, colors, times):
        add_pointcloud(stage, points, scene_path, color, time=time, points_type=points_type)
    stage.Save()

    return stage


# VoxelGrid Functions
def import_voxelgrid(file_path, scene_path, time=None):
    r"""Import a single voxelgrid from a USD file.

    Assumes that the USD voxelgrid is defined by a point instancer. Converts the coordinates
    of each point instance to an occupied voxel. The output grid size is determined by the `grid_size`
    primvar. If not specified, grid size will be determined by the axis with the largest number of occupied
    voxels. The output voxelgrid will be of shape ``[grid_size, grid_size, grid_size]``.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str): Scene path within the USD file indicating which PointInstancer primitive
            to import as a voxelgrid.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.
    Returns:
        torch.BoolTensor

    Example:
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrid('./new_stage.usd', voxelgrid, scene_path='/World/voxelgrid')
        >>> voxelgrid_imp = import_voxelgrid(file_path='./new_stage.usd',
        ...                                  scene_path='/World/voxelgrid')
        >>> voxelgrid_imp.shape
        torch.Size([32, 32, 32])
    """
    if time is None:
        time = Usd.TimeCode.Default()
    voxelgrid_list = import_voxelgrids(file_path, [scene_path], times=[time])
    return voxelgrid_list[0]


def import_voxelgrids(file_path, scene_paths=None, times=None):
    r"""Import one or more voxelgrids from a USD file.

    Assumes that the USD voxelgrid is defined by a point instancer. Converts the coordinates
    of each point instance to an occupied voxel. The output grid size is determined from the `grid_size`
    primvar. If not specified, grid size will be determined by the axis with the largest number of occupied
    voxels. The output voxelgrid will be of shape ``[grid_size, grid_size, grid_size]``.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which PointInstancer
            primitive(s) to import. If None, will return all pointclouds found based on PointInstancer
            prims with `kaolin_type` primvar set to `VoxelGrid`.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:
        (list of torch.BoolTensor)

    Example:
        >>> voxelgrid_1 = torch.rand(32, 32, 32) > 0.5
        >>> voxelgrid_2 = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrids('./new_stage.usd', [voxelgrid_1, voxelgrid_2])
        >>> voxelgrid_imp = import_voxelgrids(file_path='./new_stage.usd')
        >>> len(voxelgrid_imp)
        2
        >>> voxelgrid_imp[0].shape
        torch.Size([32, 32, 32])
    """
    assert os.path.exists(file_path)
    stage = Usd.Stage.Open(file_path)

    # If scene path not specified, find all point clouds
    if scene_paths is None:
        scene_paths = []
        for p in stage.Traverse():
            is_point_instancer = UsdGeom.PointInstancer(p)
            if is_point_instancer and p.GetAttribute('primvars:kaolin_type').Get() == 'VoxelGrid':
                scene_paths.append(p.GetPath())
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    voxelgrids = []
    for scene_path, time in zip(scene_paths, times):
        prim = stage.GetPrimAtPath(scene_path)
        assert prim, f'The prim at {scene_path} does not exist.'

        instancer = UsdGeom.PointInstancer(prim)
        assert instancer   # Currently only support pointclouds from point instancers

        voxel_indices = torch.tensor(instancer.GetPositionsAttr().Get(time=time), dtype=torch.long)
        bounds = voxel_indices.max(dim=0)[0]
        max_bound = bounds.max()
        grid_size = prim.GetAttribute('primvars:grid_size').Get(time=time)
        if grid_size is not None:
            assert max_bound < grid_size
        else:
            grid_size = max_bound
        voxelgrid = torch.zeros([grid_size, grid_size, grid_size], dtype=torch.bool)
        voxelgrid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.
        voxelgrids.append(voxelgrid)

    return voxelgrids


def add_voxelgrid(stage, voxelgrid, scene_path, time=None):
    r"""Add a voxelgrid to an existing USD stage.

    Add a voxelgrid where occupied voxels are defined by non-zero values. The voxelgrid is
    represented by point instances of a cube centered at each occupied index coordinate and scaled.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the voxelgrid.
        voxelgrid (torch.BoolTensor): Binary voxelgrid of shape ``(N, N, N)``.
        scene_path (str): Absolute path of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = add_voxelgrid(stage, voxelgrid, '/World/VoxelGrids/voxelgrid_0')
        >>> stage.Save()
    """
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    if stage.GetPrimAtPath(scene_path):
        instancer_prim = stage.GetPrimAtPath(scene_path)
    else:
        instancer_prim = stage.DefinePrim(scene_path, 'PointInstancer')
    instancer = UsdGeom.PointInstancer(instancer_prim)
    assert instancer
    cube = UsdGeom.Cube.Define(stage, f'{scene_path}/cube')
    cube.GetSizeAttr().Set(1.0)
    instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])

    # Get each occupied voxel's centre coordinate
    points = torch.nonzero(voxelgrid, as_tuple=False).float()

    # Generate instancer parameters
    indices = [0] * points.shape[0]
    positions = points.detach().cpu().tolist()
    scales = [(1.,) * 3] * points.size(0)

    # Populate point instancer
    instancer.GetProtoIndicesAttr().Set(indices, time=time)
    instancer.GetPositionsAttr().Set(positions, time=time)
    instancer.GetScalesAttr().Set(scales, time=time)

    # Centre at (0, 0, 0) and scale points
    adjusted_scale = 1. / (voxelgrid.size(0) - 1)
    scale_op = instancer.GetPrim().GetAttribute('xformOp:scale')
    if not scale_op:
        scale_op = UsdGeom.Xformable(instancer).AddScaleOp()
    scale_op.Set((adjusted_scale,) * 3, time=time)
    translate_op = instancer.GetPrim().GetAttribute('xformOp:translate')
    if not translate_op:
        UsdGeom.Xformable(instancer).AddTranslateOp()
    translate_op.Set((-0.5,) * 3, time=time)

    # Create a primvar to record the voxelgrid size
    prim = stage.GetPrimAtPath(instancer.GetPath())
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('grid_size', Sdf.ValueTypeNames.Int)
    pv.Set(voxelgrid.size(0))

    # Create a primvar to identify the poitn instancer as a Kaolin VoxelGrid
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('kaolin_type', Sdf.ValueTypeNames.String)
    pv.Set('VoxelGrid')

    return stage


def export_voxelgrid(file_path, voxelgrid, scene_path='/World/VoxelGrids/voxelgrid_0', time=None):
    r"""Export a single voxelgrid to a USD scene.

    Export a binary voxelgrid where occupied voxels are defined by non-zero values. The voxelgrid is
    represented by point instances of a cube centered at each occupied index coordinates. The voxelgrid
    will be scaled so that it fits within a unit cube. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        voxelgrid (torch.BoolTensor): Binary voxelgrid of shape ``(N, N, N)``.
        scene_path (str, optional): Absolute path of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrid('./new_stage.usd', voxelgrid)
    """
    if time is None:
        time = Usd.TimeCode.Default()
    stage = export_voxelgrids(file_path, [voxelgrid], [scene_path], times=[time])
    return stage


def export_voxelgrids(file_path, voxelgrids, scene_paths=None, times=None):
    r"""Export one or more voxelgrids to a USD scene.

    Export one or more binary voxelgrids where occupied voxels are defined by non-zero values. The voxelgrids are
    represented by point instances of a cube centered at each occupied index coordinates and scaled. The voxelgrids
    will be scaled so that it fits within a unit cube. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        voxelgrids (list of torch.BoolTensor): List of binary voxelgrid(s) of shape ``(N, N, N)``.
        scene_path (list of str, optional): Absolute path(s) of voxelgrid within the USD file scene.
            Must be a valid Sdf.Path. If no path is provided, a default path is used.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> voxelgrid_1 = torch.rand(32, 32, 32) > 0.5
        >>> voxelgrid_2 = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrids('./new_stage.usd', [voxelgrid_1, voxelgrid_2])
    """
    if scene_paths is None:
        scene_paths = [f'/World/VoxelGrids/voxelgrid_{i}' for i in range(len(voxelgrids))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)
    assert len(voxelgrids) == len(scene_paths)
    stage = create_stage(file_path)
    for scene_path, voxelgrid, time in zip(scene_paths, voxelgrids, times):
        add_voxelgrid(stage, voxelgrid, scene_path, time=time)
    stage.Save()

    return stage
